# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Flow-matching action model on causal-prefix + prefix-KV-conditioned expert.

Prefix = stock causal Qwen3-VL (images + language, optionally with latent-action
bridge tokens). Suffix = state + noisy action tokens on the Qwen2 expert.
Training and inference share one path: encode prefix once, recompute per-layer
K/V, then run the expert (Euler steps only re-run the expert).
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dual_stream import QwenVLWithExpert
from .joint_attention import create_sinusoidal_pos_embedding, sample_beta


class DualStreamFlowMatching(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        qwenvl_cfg = global_config.framework.qwenvl
        action_cfg = global_config.framework.action_model
        expert_cfg = action_cfg.action_expert

        self.qwenvl_with_expert = QwenVLWithExpert(
            base_vlm=str(qwenvl_cfg.base_vlm),
            expert_hidden_size=int(expert_cfg.hidden_size),
            expert_intermediate_size=int(expert_cfg.intermediate_size),
            expert_num_attention_heads=int(expert_cfg.num_attention_heads),
            expert_num_key_value_heads=int(expert_cfg.num_key_value_heads),
            expert_head_dim=int(expert_cfg.head_dim),
            num_expert_layers=expert_cfg.get("num_layers", None),
            adanorm_time=bool(expert_cfg.adanorm_time),
            final_norm_adanorm=bool(expert_cfg.final_norm_adanorm),
            attn_implementation=str(qwenvl_cfg.get("attn_implementation", "flash_attention_2")),
            expert_attention=str(action_cfg.get("expert_attention", "flex")),
            freeze_vision_encoder=bool(qwenvl_cfg.get("freeze_vision_encoder", False)),
            train_expert_only=bool(qwenvl_cfg.get("train_expert_only", False)),
        )

        self.proj_width = int(expert_cfg.hidden_size)
        self.n_action_steps = int(action_cfg.action_horizon)
        self.max_action_dim = int(action_cfg.action_dim)
        self.max_state_dim = int(action_cfg.state_dim)
        self.num_steps = int(action_cfg.num_inference_timesteps)
        self.noise_beta_alpha = float(action_cfg.noise_beta_alpha)
        self.noise_beta_beta = float(action_cfg.noise_beta_beta)
        self.adanorm_time = bool(expert_cfg.adanorm_time)
        self.loss_type = str(action_cfg.get("loss_type", "fm"))
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 1))

        # Expert-side projections stay in fp32 with the expert.
        self.state_proj = nn.Linear(self.max_state_dim, self.proj_width)
        self.action_in_proj = nn.Linear(self.max_action_dim, self.proj_width)
        self.action_out_proj = nn.Linear(self.proj_width, self.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.proj_width * 2, self.proj_width)
        self.action_time_mlp_out = nn.Linear(self.proj_width, self.proj_width)

    @property
    def image_token_id(self) -> int:
        return int(self.qwenvl_with_expert.qwenvl.config.image_token_id)

    # -- time -------------------------------------------------------------
    def sample_time(self, bsize, device):
        time_beta = sample_beta(self.noise_beta_alpha, self.noise_beta_beta, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # -- prefix (chat-template context) -----------------------------------
    def embed_prefix(self, inputs: dict, num_latent_tokens: int = 0) -> dict:
        """Build prefix embeddings from Qwen3-VL processor outputs.

        ``inputs`` carries ``input_ids`` / ``attention_mask`` / ``pixel_values`` /
        ``image_grid_thw``; ``num_latent_tokens`` counts bridge tokens already
        appended to the tail of ``input_ids``.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        embs = self.qwenvl_with_expert.embed_language_tokens(input_ids)
        image_embeds, deepstack_visual_embeds = self.qwenvl_with_expert.embed_image(
            pixel_values.to(dtype=embs.dtype), image_grid_thw
        )

        visual_pos_masks = input_ids == self.image_token_id
        image_mask = visual_pos_masks.unsqueeze(-1).expand_as(embs)
        embs = embs.masked_scatter(image_mask, image_embeds.to(device=embs.device, dtype=embs.dtype))

        pad_masks = attention_mask.bool()
        position_ids = self.qwenvl_with_expert.build_position_ids(
            input_ids, attention_mask.long(), image_grid_thw=image_grid_thw
        )

        prompt_len = input_ids.shape[1] - num_latent_tokens
        # Bridge tokens must not shift the suffix RoPE positions.
        prompt_pad_masks = pad_masks[:, :prompt_len].clone()

        return dict(
            embs=embs,
            pad_masks=pad_masks,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            prompt_len=prompt_len,
            prompt_pad_masks=prompt_pad_masks,
        )

    def encode_prefix_context(self, inputs: dict, num_latent_tokens: int = 0):
        """Encode the VLM prefix and build per-layer prompt K/V for the expert."""
        prefix = self.embed_prefix(inputs, num_latent_tokens)
        last_hidden, layer_inputs = self.qwenvl_with_expert.encode_prefix_layers(
            inputs_embeds=prefix["embs"],
            attention_mask=prefix["pad_masks"],
            position_ids=prefix["position_ids"],
            visual_pos_masks=prefix["visual_pos_masks"],
            deepstack_visual_embeds=prefix["deepstack_visual_embeds"],
        )
        prefix_kvs = self.qwenvl_with_expert.build_prefix_kv(
            layer_inputs,
            prefix["position_ids"],
            prefix["prompt_len"],
        )
        return prefix, last_hidden, prefix_kvs

    # -- suffix (state + noisy actions + time) ----------------------------
    def embed_suffix(self, state, noisy_actions, timestep):
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
        # Expert path is fp32.
        state = state.float()
        noisy_actions = noisy_actions.float()
        timestep = timestep.float()

        state_emb = self.state_proj(state)

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb_rep = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb_rep], dim=-1)
        action_time_emb = F.silu(self.action_time_mlp_in(action_time_emb))
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        # pi0 block semantics: state opens a block, first action opens another.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True
        return time_emb, embs, pad_masks, att_masks

    # -- positions --------------------------------------------------------
    @staticmethod
    def _build_suffix_position_ids(prefix_position_ids, prompt_pad_masks, suffix_pad_masks):
        """3D mrope ids for the suffix, offset past the prompt (bridge excluded)."""
        prompt_position_ids = prefix_position_ids[..., : prompt_pad_masks.shape[1]]
        valid_prefix_pos = prompt_position_ids.masked_fill(~prompt_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        return suffix_1d.unsqueeze(0).expand(3, -1, -1)

    # -- shared expert step -----------------------------------------------
    def run_expert(self, prefix, prefix_kvs, state, x_t, timestep) -> Tensor:
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        suffix_out = self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=time_embs if self.adanorm_time else None,
        )
        return suffix_out[:, -x_t.shape[1] :]

    # -- repeat helpers ---------------------------------------------------
    @staticmethod
    def _expand_prefix_for_repeats(prefix: dict, prefix_kvs, repeats: int):
        """Tile prefix K/V / masks along batch so the expert sees ``r`` diffusion copies.

        The VLM prefix is encoded once; only the tensors the expert reads are expanded.
        """
        if repeats <= 1:
            return prefix, prefix_kvs
        # position_ids: [3, B, L] -> repeat batch axis
        position_ids = prefix["position_ids"].repeat(1, repeats, 1)
        prompt_pad_masks = prefix["prompt_pad_masks"].repeat(repeats, 1)
        expanded_prefix = {
            **prefix,
            "position_ids": position_ids,
            "prompt_pad_masks": prompt_pad_masks,
        }
        expanded_kvs = [
            (key.repeat(repeats, 1, 1, 1), value.repeat(repeats, 1, 1, 1)) for key, value in prefix_kvs
        ]
        return expanded_prefix, expanded_kvs

    # -- LA: bridge-token hidden only -------------------------------------
    def encode_prefix(self, inputs: dict, num_latent_tokens: int) -> Tensor:
        prefix, last_hidden, _ = self.encode_prefix_context(inputs, num_latent_tokens)
        return last_hidden[:, prefix["prompt_len"] :]

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
        """Flow-matching loss with optional diffusion repeats.

        The VLM prefix is encoded once at batch ``B``. When ``num_repeats=r>1``,
        actions / state / prefix K/V are tiled to ``B*r`` for the expert, matching
        QwenPI_v4's cheap repeat (VLM not re-run).
        """
        actions = actions.float()
        action_mask = action_mask.float()
        device = actions.device
        repeats = int(self.repeated_diffusion_steps if num_repeats is None else num_repeats)
        if repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {repeats}.")

        prefix, last_hidden, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens)
        if actions.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"actions batch {actions.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]} before repeats."
            )

        # The expert is designed to run in fp32 (see the explicit ``.float()``
        # calls throughout embed_suffix/run_expert), but the trainer wraps the
        # whole forward in ``autocast(bfloat16)`` and DeepSpeed casts all
        # parameters to bf16, so those casts alone are silently overridden back
        # to bf16 compute. Nesting an ``autocast(float32)`` island here forces
        # the expert's autocast-eligible ops (matmul/linear) to actually compute
        # in fp32, matching QwenPI_v4's DiT precision handling.
        with torch.autocast("cuda", dtype=torch.float32):
            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
                if state is not None:
                    state = state.repeat(repeats, 1)

            if state is None:
                state = torch.zeros(actions.shape[0], self.max_state_dim, device=device, dtype=torch.float32)

            if noise is None:
                noise = torch.randn(actions.shape, device=device, dtype=torch.float32)
            else:
                noise = noise.float()
            if time is None:
                time = self.sample_time(actions.size(0), device)
            else:
                time = time.float()

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert(expert_prefix, expert_kvs, state, x_t, time)
            v_t = self.action_out_proj(suffix_out).float()

            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

        out = {"action_loss": action_loss}
        if num_latent_tokens > 0:
            out["latent_hidden"] = last_hidden[:, prefix["prompt_len"] :]
        return out

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, inputs: dict, state: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                (bsize, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32
            )
        else:
            noise = noise.float()

        prefix, _, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens=0)

        # See the matching comment in ``forward``: force real fp32 compute for
        # the expert's autocast-eligible ops instead of silently inheriting
        # whatever ambient autocast dtype the caller established.
        with torch.autocast("cuda", dtype=torch.float32):
            dt = -1.0 / self.num_steps
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                suffix_out = self.run_expert(prefix, prefix_kvs, state, x_t, expanded_time)
                v_t = self.action_out_proj(suffix_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t
