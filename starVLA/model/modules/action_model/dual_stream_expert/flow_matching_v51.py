# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenPI_v51 action model: state token inside the VLM prefix.

Differences from ``DualStreamFlowMatching`` (piv5):

- The robot state is embedded by ``state_vlm_proj`` into the VLM hidden size and
  written onto a ``<state>`` placeholder token that the framework inserts between
  the prompt and the latent-action bridge tokens, so the state is processed by
  the VLM parameters. Pure-video batches simply omit the token and the mrope
  positions continue without a gap.
- The expert prefix K/V cover the *full* sequence (prompt + state + LA bridge
  tokens) instead of being sliced to the prompt, so the action suffix attends to
  the LA queries' per-layer hidden states. Suffix RoPE positions are offset past
  the whole sequence accordingly.
- The expert suffix drops its own state token; it is actions-only.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flow_matching import DualStreamFlowMatching
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingV51(DualStreamFlowMatching):
    def __init__(self, global_config):
        super().__init__(global_config)
        vl_hidden_dim = int(self.qwenvl_with_expert.qwenvl.config.text_config.hidden_size)
        # Projects the robot state onto the `<state>` placeholder token in the
        # VLM embedding space. The token id is injected by the framework after it
        # expands the tokenizer.
        self.state_vlm_proj = nn.Linear(self.max_state_dim, vl_hidden_dim)
        self.state_token_id: Optional[int] = None

    # -- prefix (chat-template context + state token) ---------------------
    def embed_prefix(self, inputs: dict, num_latent_tokens: int = 0, state: Optional[Tensor] = None) -> dict:
        """Prefix embeddings with the state token written into the VLM sequence.

        The parent call uses ``num_latent_tokens=0`` so ``prompt_len`` /
        ``prompt_pad_masks`` cover the full sequence: expert prefix K/V then
        include the state and LA bridge tokens, and suffix positions are offset
        past them. ``latent_start`` marks where the bridge tokens begin for the
        latent-loss readout.
        """
        prefix = super().embed_prefix(inputs, 0)
        input_ids = inputs["input_ids"]
        if self.state_token_id is not None:
            state_mask = input_ids == self.state_token_id
            if bool(state_mask.any()):
                if state is None:
                    raise ValueError("inputs contain <state> tokens but no state tensor was provided.")
                if state.ndim == 3:
                    state = state.squeeze(1)
                state_emb = self.state_vlm_proj(state.float()).to(prefix["embs"].dtype)
                embs = prefix["embs"]
                prefix["embs"] = embs.masked_scatter(state_mask.unsqueeze(-1).expand_as(embs), state_emb)
        prefix["latent_start"] = input_ids.shape[1] - num_latent_tokens
        return prefix

    def encode_prefix_hidden(self, inputs: dict, num_latent_tokens: int = 0, state: Optional[Tensor] = None):
        prefix = self.embed_prefix(inputs, num_latent_tokens, state=state)
        last_hidden, layer_inputs = self.qwenvl_with_expert.encode_prefix_layers(
            inputs_embeds=prefix["embs"],
            attention_mask=prefix["pad_masks"],
            position_ids=prefix["position_ids"],
            visual_pos_masks=prefix["visual_pos_masks"],
            deepstack_visual_embeds=prefix["deepstack_visual_embeds"],
        )
        return prefix, last_hidden, layer_inputs

    def encode_prefix_context(self, inputs: dict, num_latent_tokens: int = 0, state: Optional[Tensor] = None):
        """Encode the VLM prefix and build per-layer K/V over the full sequence."""
        prefix, last_hidden, layer_inputs = self.encode_prefix_hidden(inputs, num_latent_tokens, state=state)
        return prefix, last_hidden, self.build_prefix_kvs(prefix, layer_inputs)

    # -- suffix (noisy actions + time; state lives in the prefix) ---------
    def embed_suffix(self, state, noisy_actions, timestep):
        # v51: the state is conditioned through the VLM prefix, so the suffix is
        # actions-only. ``state`` is accepted for signature compatibility.
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        # Expert path is fp32.
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb_rep = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb_rep], dim=-1)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        embs = action_time_emb
        pad_masks = torch.ones((bsize, action_time_dim), device=device, dtype=torch.bool)
        # pi0 block semantics: the first action opens a block.
        att_masks = torch.zeros((bsize, action_time_dim), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        return time_emb, embs, pad_masks, att_masks

    # -- LA: bridge-token hidden only -------------------------------------
    def encode_prefix(self, inputs: dict, num_latent_tokens: int, state: Optional[Tensor] = None) -> Tensor:
        prefix, last_hidden, _ = self.encode_prefix_hidden(inputs, num_latent_tokens, state=state)
        return last_hidden[:, prefix["latent_start"] :]

    # -- training ---------------------------------------------------------
    def forward(
        self,
        inputs: dict,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        num_latent_tokens: int = 0,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
    ) -> dict:
        prefix, last_hidden, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens, state=state)
        action_loss = self.flow_matching_loss(
            prefix, prefix_kvs, state, actions, action_mask, noise=noise, time=time, num_repeats=num_repeats
        )
        out = {"action_loss": action_loss}
        if num_latent_tokens > 0:
            out["latent_hidden"] = last_hidden[:, prefix["latent_start"] :]
        return out

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Optional[Tensor],
        noise: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        device = inputs["input_ids"].device
        bsize = state.shape[0] if state is not None else inputs["input_ids"].shape[0]

        if noise is None:
            noise = torch.randn(
                (bsize, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32
            )
        else:
            noise = noise.float()

        prefix, _, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens=0, state=state)

        # See the matching comment in the parent ``forward``: force real fp32
        # compute for the expert's autocast-eligible ops.
        with torch.autocast("cuda", dtype=torch.float32):
            dt = -1.0 / self.num_steps
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_out = self.run_expert(
                    prefix, prefix_kvs, state, x_t, expanded_time,
                    attention_implementation=attention_implementation,
                )
                v_t = self.action_out_proj(suffix_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t
