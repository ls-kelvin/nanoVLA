# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv33_LA foresight: query-first two-pass with stop-gradient on queries.

Logical suffix order is ``[query(N) | state(1) | action(chunk)]`` with pi0
block-causal semantics, implemented as two expert passes on top of the V31
KV-cache plumbing:

  pass 1: [query(N)]              single bidirectional block, no AdaRMSNorm FiLM
  pass 2: [state(1), action(chunk)] block-causal, time-FiLM, attends to
                                  [prefix | query(detached)] K/V
  latent-only: [query(N)]         fully bidirectional, no FiLM, no repeats

Pass 1 is unmodulated (plain RMSNorm, not AdaRMSNorm at t=0). Its per-layer
K/V are detached before being concatenated onto the prefix cache, so the
action loss never backpropagates into the learnable query tokens: queries are
trained only by their own latent readout loss (embedding MSE or soft_kl).
Pass 2 keeps the sampled-timestep FiLM (``ada_cond=time``) for state+action.

Pass 2 has two attention blocks, so it cannot use flash. Pass 1 is a single
bidirectional block and follows ``latent_expert_attention``.
"""

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import Tensor

from .flow_matching_foresight_v31 import DualStreamFlowMatchingForesightV31
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingForesightV33(DualStreamFlowMatchingForesightV31):
    """V31 two-pass foresight with query-first order and detached query K/V."""

    # -- pass 1: [query(N)] -------------------------------------------------
    def encode_query_kv(
        self,
        prefix: dict,
        prefix_kvs,
        attention_implementation: Optional[str] = None,
    ):
        """Pass 1: encode ``[query]`` once. Returns query hidden + its K/V cache.

        No AdaRMSNorm FiLM (unmodulated RMSNorm, not t=0). The returned suffix
        K/V still carry gradients (the latent readout loss backprops through
        them); callers must detach before reusing them as pass-2 prefix.
        """
        bsize = prefix["prompt_pad_masks"].shape[0]
        device = prefix["prompt_pad_masks"].device
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_learnable_only(
            bsize, device
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        suffix_out, suffix_kvs = self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=None,
            attention_implementation=attention_implementation or self.latent_expert_attention,
            return_suffix_kvs=True,
        )
        return suffix_out, suffix_kvs, suffix_pad_masks

    # -- pass 2: [state(1), action_time(chunk)] -----------------------------
    def embed_suffix_state_action(self, state: Tensor, noisy_actions: Tensor, timestep: Tensor):
        """State + action suffix. Two pi0 blocks (not flash-compatible)."""
        if state.ndim == 3:
            state = state.squeeze(1)
        bsize = state.shape[0]
        device = state.device
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
        action_len = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        suffix_len = 1 + action_len
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        # Blocks: state | action (first token of each opens a block).
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        att_masks[:, 1] = True
        return time_emb, embs, pad_masks, att_masks

    def run_expert_state_action(
        self,
        prefix: dict,
        prefix_kvs,
        prefix_pad_masks: Tensor,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        attention_implementation: Optional[str] = None,
        position_offset: Optional[int] = None,
    ) -> Tensor:
        """Pass 2: ``[state | action]`` attending to ``[prefix | query]`` K/V.

        ``prefix`` still holds the original VLM ``position_ids`` / prompt pads so
        RoPE continues after the query block (offset by ``num_learnable_tokens``;
        pass ``position_offset=0`` when no query K/V precede this suffix).
        ``prefix_kvs`` / ``prefix_pad_masks`` are the concatenated cache from
        pass 1 (query K/V already detached by the caller).

        Attention is block-causal (state | action), so flash is rejected.
        """
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_state_action(
            state, x_t, timestep
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        if position_offset is None:
            position_offset = self.num_learnable_tokens
        suffix_position_ids = suffix_position_ids + position_offset
        suffix_out = self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=time_embs if self.adanorm_time else None,
            attention_implementation=self._resolve_block_causal_attention(attention_implementation),
        )
        return suffix_out[:, -x_t.shape[1] :]

    @staticmethod
    def _detach_kvs(kvs):
        return [(key.detach(), value.detach()) for key, value in kvs]

    # -- training: joint ----------------------------------------------------
    def flow_matching_loss_joint_foresight(
        self,
        prefix: dict,
        prefix_kvs,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        latent_targets: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Query pass at ``B``, then state+action flow-matching at ``B*r``.

        Query K/V are detached before pass 2, so the action loss does not
        backpropagate into the learnable query tokens; queries are trained
        only by the latent readout loss (which never sees state/action).
        """
        actions = actions.float()
        action_mask = action_mask.float()
        latent_targets = latent_targets.float()
        self._check_latent_token_count(latent_targets)
        device = actions.device
        bsize = actions.shape[0]
        repeats = int(self.repeated_diffusion_steps if num_repeats is None else num_repeats)
        if repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {repeats}.")

        if actions.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"actions batch {actions.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]} before repeats."
            )
        if latent_targets.shape[0] != bsize:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match actions batch {bsize}."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            query_out, query_kvs, query_pad = self.encode_query_kv(
                prefix, prefix_kvs, attention_implementation=attention_implementation
            )
            latent_loss = self._latent_loss_from_learnable(
                query_out, latent_targets, valid_mask=latent_valid_mask
            )

            # Stop gradient: later blocks attend to query K/V but never
            # backprop into the query tokens.
            combined_kvs = self._cat_kvs(prefix_kvs, self._detach_kvs(query_kvs))
            combined_pad = torch.cat([prefix["prompt_pad_masks"], query_pad], dim=1)

            if state is None:
                state = torch.zeros(
                    bsize, self.max_state_dim, device=device, dtype=torch.float32
                )
            elif state.ndim == 3:
                state = state.squeeze(1)
            if state.shape[0] != bsize:
                raise ValueError(
                    f"state batch {state.shape[0]} must match actions batch {bsize} before repeats."
                )

            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
                state = state.repeat(repeats, 1)

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

            expert_prefix = self._expand_prefix_dict(prefix, repeats)
            action_out = self.run_expert_state_action(
                expert_prefix,
                self._repeat_kvs(combined_kvs, repeats),
                combined_pad.repeat(repeats, 1) if repeats > 1 else combined_pad,
                state,
                x_t,
                time,
                attention_implementation=attention_implementation,
            )

            v_t = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

        return action_loss, latent_loss

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        inputs: dict,
        state: Tensor,
        noise: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Euler sampling; query K/V are encoded once, then reused (detached)."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = torch.randn(
                (bsize, self.n_action_steps, self.max_action_dim), device=device, dtype=torch.float32
            )
        else:
            noise = noise.float()

        prefix, _, prefix_kvs = self.encode_prefix_context(inputs, num_latent_tokens=0)

        with torch.autocast("cuda", dtype=torch.float32):
            _, query_kvs, query_pad = self.encode_query_kv(
                prefix, prefix_kvs, attention_implementation=attention_implementation
            )
            combined_kvs = self._cat_kvs(prefix_kvs, self._detach_kvs(query_kvs))
            combined_pad = torch.cat([prefix["prompt_pad_masks"], query_pad], dim=1)

            dt = -1.0 / self.num_steps
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                action_out = self.run_expert_state_action(
                    prefix,
                    combined_kvs,
                    combined_pad,
                    state,
                    x_t,
                    expanded_time,
                    attention_implementation=attention_implementation,
                )
                v_t = self.action_out_proj(action_out).float()
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t
