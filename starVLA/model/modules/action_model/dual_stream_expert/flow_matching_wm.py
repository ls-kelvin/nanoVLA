# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Decoupled latent-action flow-matching branch on top of ``DualStreamFlowMatching``.

Mirrors the "world-model" idea from ``QwenWM_LA``
(``LayerwiseMetaqueryWMFlowmatchingActionHead``): latent and action are two
**independent** flow-matching streams that share the same VLM prefix and the
same Qwen2 expert weights, but each runs its own forward pass with its own
noise/velocity and its own input/output projections. Unlike ``QwenWM_LA``
(which conditions on cross-attention over VLM hidden states), the latent
stream here reads the VLM the same way the action stream does: per-layer
prefix K/V fed to the expert (see ``dual_stream.QwenVLWithExpert``).

The latent suffix has no state/action block-causal structure -- it is a
single free-attention block, analogous to ``LayerwiseMetaqueryWMFlowmatchingActionHead``'s
latent stream (``self_attention_mask=None``).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .flow_matching import DualStreamFlowMatching
from .joint_attention import create_sinusoidal_pos_embedding


class DualStreamFlowMatchingWM(DualStreamFlowMatching):
    """``DualStreamFlowMatching`` plus an independent latent-action flow-matching branch."""

    def __init__(self, global_config):
        super().__init__(global_config)
        action_cfg = global_config.framework.action_model

        latent_action_dim = action_cfg.get("latent_action_dim", None)
        if latent_action_dim is None:
            raise ValueError(
                "DualStreamFlowMatchingWM requires framework.action_model.latent_action_dim "
                "(set by the framework before constructing the action model)."
            )
        self.latent_action_dim = int(latent_action_dim)
        # Default to sdpa (same as the action stream); "flash" is kept as an
        # opt-in, less-battle-tested option (see joint_attention.flash_attention_dense).
        self.latent_expert_attention = str(action_cfg.get("latent_expert_attention", "sdpa"))

        # Independent projections; the latent stream never shares weights with
        # the action stream (own encoder/decoder, own time-conditioning MLP).
        self.latent_in_proj = nn.Linear(self.latent_action_dim, self.proj_width)
        self.latent_time_mlp_in = nn.Linear(self.proj_width * 2, self.proj_width)
        self.latent_time_mlp_out = nn.Linear(self.proj_width, self.proj_width)
        self.latent_out_proj = nn.Linear(self.proj_width, self.latent_action_dim)

    # -- suffix (noisy latent tokens + time) -------------------------------
    def embed_suffix_latent(self, noisy_latent: Tensor, timestep: Tensor):
        """Embed the latent-only suffix. No state token, no block-causal split."""
        bsize = noisy_latent.shape[0]
        device = noisy_latent.device
        noisy_latent = noisy_latent.float()
        timestep = timestep.float()

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=4e-3, max_period=4.0, device=device
        ).to(dtype=torch.float32)

        latent_emb = self.latent_in_proj(noisy_latent)
        time_emb_rep = time_emb[:, None, :].expand(-1, latent_emb.shape[1], -1)
        latent_time_emb = torch.cat([latent_emb, time_emb_rep], dim=-1)
        latent_time_emb = F.silu(self.latent_time_mlp_in(latent_time_emb))
        latent_time_emb = self.latent_time_mlp_out(latent_time_emb)
        latent_len = latent_time_emb.shape[1]

        pad_masks = torch.ones((bsize, latent_len), device=device, dtype=torch.bool)
        # Single attention block (att_masks all-zero-but-first): every latent
        # token attends to every other latent token, i.e. fully bidirectional
        # self-attention -- mirrors the WM latent stream's "free self-attn".
        att_masks = torch.zeros((bsize, latent_len), device=device, dtype=torch.bool)
        att_masks[:, 0] = True
        return time_emb, latent_time_emb, pad_masks, att_masks

    # -- shared expert step (latent stream) --------------------------------
    def run_expert_latent(
        self,
        prefix: dict,
        prefix_kvs,
        x_t: Tensor,
        timestep: Tensor,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_latent(
            x_t, timestep
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
            attention_implementation=attention_implementation or self.latent_expert_attention,
        )
        return suffix_out

    # -- training (latent stream) -------------------------------------------
    def flow_matching_loss_latent(
        self,
        prefix: dict,
        prefix_kvs,
        latent_targets: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        num_repeats: Optional[int] = None,
        attention_implementation: Optional[str] = None,
    ) -> Tensor:
        """Flow-matching MSE loss for the independent latent stream.

        Structurally mirrors :meth:`DualStreamFlowMatching.flow_matching_loss`
        (same repeated-diffusion-step handling via ``_expand_prefix_for_repeats``),
        but has no action mask / state (latent targets have no padding-dim
        semantics: every codebook dim is always valid).
        """
        latent_targets = latent_targets.float()
        device = latent_targets.device
        repeats = int(self.repeated_diffusion_steps if num_repeats is None else num_repeats)
        if repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {repeats}.")

        if latent_targets.shape[0] != prefix["prompt_pad_masks"].shape[0]:
            raise ValueError(
                f"latent_targets batch {latent_targets.shape[0]} must match prefix batch "
                f"{prefix['prompt_pad_masks'].shape[0]} before repeats."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            if repeats > 1:
                latent_targets = latent_targets.repeat(repeats, 1, 1)

            if noise is None:
                noise = torch.randn(latent_targets.shape, device=device, dtype=torch.float32)
            else:
                noise = noise.float()
            if time is None:
                time = self.sample_time(latent_targets.size(0), device)
            else:
                time = time.float()

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * latent_targets
            u_t = noise - latent_targets

            expert_prefix, expert_kvs = self._expand_prefix_for_repeats(prefix, prefix_kvs, repeats)
            suffix_out = self.run_expert_latent(
                expert_prefix, expert_kvs, x_t, time, attention_implementation=attention_implementation
            )
            v_t = self.latent_out_proj(suffix_out).float()

            latent_loss = F.mse_loss(u_t, v_t)

        return latent_loss
