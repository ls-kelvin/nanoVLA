# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv35_LA: QwenWMv32_LA with prefix-attention dropout on noisy actions.

Same architecture as WMv32; during joint training, each sample independently
(with probability ``framework.action_model.prefix_drop_prob``) hides all
prefix K/V from its noisy-action query rows. State and learnable (latent
foresight) rows keep prefix access; inference is unchanged.
"""

from starVLA.model.framework.VLM4A.QwenWMv32_LA import Qwen_WMv32_LA
from starVLA.model.modules.action_model.dual_stream_expert import (
    DualStreamFlowMatchingForesightV35,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenWMv35_LA")
class Qwen_WMv35_LA(Qwen_WMv32_LA):
    """WMv32 + prefix-attention dropout for noisy-action rows."""

    _framework_name = "QwenWMv35_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV35
