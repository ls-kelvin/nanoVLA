# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv32_LA foresight: WMv3 joint learnable tokens, sampled-time latent cond.

Two changes on top of ``DualStreamFlowMatchingForesight`` (WMv3):

1. The latent-only path drops the fixed learned ``foresight_cond`` embedding
   and instead samples a flow-matching timestep with ``sample_time`` (same
   Beta schedule as the action stream) whose sinusoidal embedding serves as
   the AdaRMSNorm condition -- matching how the joint path already conditions
   latent tokens on the (sampled) action timestep.
2. Latent supervision uses all ``repeated_diffusion_steps`` copies in both
   paths instead of only the first ``B`` rows (joint) or a single pass
   (latent-only). Each repeat row draws its own timestep, so under
   ``adanorm_time`` every copy is a distinct conditioning sample.

``sample_latent`` stays deterministic by conditioning on ``t = 0``.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .flow_matching_foresight import DualStreamFlowMatchingForesight
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingForesightV32(DualStreamFlowMatchingForesight):
    """WMv3 foresight with sampled-time latent cond and repeated latent loss."""

    def __init__(self, global_config):
        super().__init__(global_config)
        # Replaced by sampled-t sinusoidal embeddings; keep it out of the state dict.
        del self.foresight_cond

    # -- latent-only conditioning ------------------------------------------
    def _latent_time_cond(
        self, bsize: int, device: torch.device, timestep: Optional[Tensor]
    ) -> Optional[Tensor]:
        if not self.adanorm_time:
            return None
        if timestep is None:
            timestep = self.sample_time(bsize, device)
        return create_sinusoidal_pos_embedding(
            timestep.float(), self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

    # -- expert runners -----------------------------------------------------
    def run_expert_learnable_only(
        self,
        prefix: dict,
        prefix_kvs,
        timestep: Optional[Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Latent-only expert forward; AdaRMSNorm cond is a sampled-t embedding."""
        bsize = prefix["prompt_pad_masks"].shape[0]
        device = prefix["prompt_pad_masks"].device
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_learnable_only(
            bsize, device
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix["position_ids"], prefix["prompt_pad_masks"], suffix_pad_masks
        )
        return self.qwenvl_with_expert.run_expert(
            suffix_embs=suffix_embs,
            prefix_kvs=prefix_kvs,
            suffix_position_ids=suffix_position_ids,
            prefix_pad_masks=prefix["prompt_pad_masks"],
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            ada_cond=self._latent_time_cond(bsize, device, timestep),
            attention_implementation=attention_implementation or self.latent_expert_attention,
        )

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
        """One joint expert pass -> ``(action_loss, latent_loss)``.

        Unlike WMv3, the latent loss uses all ``B * repeats`` rows: each
        repeat row has its own sampled timestep in the AdaRMSNorm condition,
        so the learnable readouts are not identical across rows.
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
            if repeats > 1:
                actions = actions.repeat(repeats, 1, 1)
                action_mask = action_mask.repeat(repeats, 1, 1)
                if state is not None:
                    state = state.repeat(repeats, 1)

            if state is None:
                state = torch.zeros(
                    actions.shape[0], self.max_state_dim, device=device, dtype=torch.float32
                )

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
            suffix_out = self.run_expert_foresight(
                expert_prefix,
                expert_kvs,
                state,
                x_t,
                time,
                attention_implementation=attention_implementation,
            )
            learnable_out, action_out = self._split_foresight_outputs(
                suffix_out, self.num_learnable_tokens, x_t.shape[1]
            )

            v_t = self.action_out_proj(action_out).float()
            if self.loss_type == "L1_fm":
                losses = F.l1_loss(u_t, v_t, reduction="none")
            else:
                losses = F.mse_loss(u_t, v_t, reduction="none")
            action_loss = (losses * action_mask).sum() / action_mask.sum().clamp_min(1.0)

            if repeats > 1:
                latent_targets = latent_targets.repeat(repeats, 1, 1)
                if latent_valid_mask is not None:
                    latent_valid_mask = latent_valid_mask.repeat(repeats)
            latent_loss = self._latent_loss_from_learnable(
                learnable_out, latent_targets, valid_mask=latent_valid_mask
            )

        return action_loss, latent_loss

    # -- training: latent-only ----------------------------------------------
    def flow_matching_loss_latent_only(
        self,
        prefix: dict,
        prefix_kvs,
        latent_targets: Tensor,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
        latent_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Single learnable-token readout with one independently sampled timestep.

        ``repeated_diffusion_steps`` is used by the joint action path, but the
        latent-only path intentionally never repeats.
        """
        latent_targets = latent_targets.float()
        self._check_latent_token_count(latent_targets)

        if latent_targets.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]}."
            )
        # Keep the argument for call-site compatibility, but latent-only must
        # remain a single pass regardless of the action repeat setting.
        repeats = 1

        with torch.autocast("cuda", dtype=torch.float32):
            if repeats > 1:
                latent_targets = latent_targets.repeat(repeats, 1, 1)
                if latent_valid_mask is not None:
                    latent_valid_mask = latent_valid_mask.repeat(repeats)
            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert_learnable_only(
                expert_prefix, expert_kvs, attention_implementation=attention_implementation
            )
            latent_loss = self._latent_loss_from_learnable(
                suffix_out, latent_targets, valid_mask=latent_valid_mask
            )

        return latent_loss

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def sample_latent(self, prefix: dict, prefix_kvs) -> Tensor:
        """Single-pass foresight readout, deterministically conditioned on t = 0."""
        bsize = prefix["prompt_pad_masks"].shape[0]
        device = prefix["prompt_pad_masks"].device
        with torch.autocast("cuda", dtype=torch.float32):
            timestep = torch.zeros(bsize, dtype=torch.float32, device=device)
            suffix_out = self.run_expert_learnable_only(prefix, prefix_kvs, timestep=timestep)
            if self.foresight_latent_loss_type == "embedding":
                return self.learnable_to_latent_proj(suffix_out).float()
            return self.learnable_to_logits_proj(suffix_out).float()
