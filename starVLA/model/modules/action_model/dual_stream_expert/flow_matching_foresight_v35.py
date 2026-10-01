# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv35_LA foresight: WMv32 + prefix-attention dropout for noisy actions.

Same architecture as WMv32. The only change is a structural dropout on the
joint training path: with probability ``action_model.prefix_drop_prob``, a
sample's noisy-action query rows cannot attend to any prefix K/V (images /
text / state prompt), while the state and learnable (latent foresight) rows
keep prefix access. The drop decision is sampled per original sample and
shared by all ``repeated_diffusion_steps`` repeat rows.

With ``prefix_drop_prob = 1.0`` the action stream never sees the prefix
directly and can only reach it through the learnable (latent foresight)
bottleneck; set ``action_model.inference_prefix_drop: true`` so
``sample_actions`` masks the prefix as well and inference matches training.
Otherwise dropout is training-only: ``sample_actions`` / ``sample_latent``
and the latent-only path are unaffected.
"""

from typing import Optional

import torch
from torch import Tensor

from .flow_matching_foresight_v32 import DualStreamFlowMatchingForesightV32


class DualStreamFlowMatchingForesightV35(DualStreamFlowMatchingForesightV32):
    """WMv32 with prefix-attention dropout on noisy-action rows."""

    def __init__(self, global_config):
        super().__init__(global_config)
        action_cfg = global_config.framework.action_model
        # Probability of hiding all prefix K/V from a sample's noisy-action
        # query rows during joint training. 1.0 = the action stream never
        # attends to the prefix directly (latent-bottleneck ablation).
        self.prefix_drop_prob = float(action_cfg.get("prefix_drop_prob", 0.0))
        if not 0.0 <= self.prefix_drop_prob <= 1.0:
            raise ValueError(
                f"framework.action_model.prefix_drop_prob must be in [0, 1], got {self.prefix_drop_prob}."
            )
        # Also mask prefix from action rows during sample_actions, so inference
        # matches training when prefix_drop_prob = 1.0.
        self.inference_prefix_drop = bool(action_cfg.get("inference_prefix_drop", False))

    def _sample_prefix_drop_mask(self, bsize: int, device: torch.device) -> Optional[Tensor]:
        """Per-sample drop decisions; None disables the dropout path entirely."""
        if self.prefix_drop_prob <= 0.0 or not self.training:
            return None
        return torch.rand(bsize, device=device) < self.prefix_drop_prob

    def _inference_prefix_drop_mask(self, bsize: int, device: torch.device) -> Optional[Tensor]:
        """All-True drop mask when inference-time prefix drop is configured."""
        if not self.inference_prefix_drop:
            return None
        return torch.ones(bsize, dtype=torch.bool, device=device)
