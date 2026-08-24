# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv34_LA: QwenWMv33_LA plus ground-truth codebook latent tokens.

Logical suffix ``[query | latent | state | action]``: nothing after the query
block attends to query (its stop-gradient becomes vacuous), while state and
action attend to latent tokens built from the Sharla codebook mapped to the
expert hidden size. Training teacher-forces ground-truth latents; inference
predicts them with the query pass first (two-stage).
"""

from pathlib import Path
from typing import Optional

import torch

from starVLA.model.framework.VLM4A.QwenWMv33_LA import Qwen_WMv33_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingForesightV34
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWMv34_LA")
class Qwen_WMv34_LA(Qwen_WMv33_LA):
    """v5 prefix-KV expert + codebook-latent-conditioned foresight."""

    _framework_name = "QwenWMv34_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV34

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.action_model.set_latent_codebook(self._load_sharla_codebook())
        logger.info("%s injected Sharla codebook into the action model.", self._framework_name)

    def _load_sharla_codebook(self) -> torch.Tensor:
        """Return the Sharla quantizer codebook ``[codebook_size, latent_dim]``."""
        if self.latent_action_encoder is not None:
            return self.latent_action_encoder.model.quantizer.codebook.weight.detach().float()

        # The encoder may be skipped when cached soft distributions are used;
        # fall back to reading just the codebook from the Sharla checkpoint.
        sharla_cfg = self.latent_action_cfg.get("sharla", {}) or {}
        ckpt_path = sharla_cfg.get("ckpt_path", None)
        if ckpt_path is None or str(ckpt_path) in ("", "null", "None"):
            raise ValueError(
                f"{self._framework_name} requires the Sharla codebook but the latent-action "
                "encoder is not loaded and framework.latent_action.sharla.ckpt_path is unset."
            )
        path = Path(str(ckpt_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Sharla tokenizer checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        for key in ("quantizer.codebook.weight", "model.quantizer.codebook.weight"):
            if key in state_dict:
                return state_dict[key].detach().float()
        raise KeyError(
            f"No quantizer codebook found in Sharla checkpoint {path}; "
            f"available keys sample: {sorted(state_dict.keys())[:10]}"
        )
