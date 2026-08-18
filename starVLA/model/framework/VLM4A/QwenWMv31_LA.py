# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv31_LA: QwenWMv3_LA with two-pass state/latent KV foresight.

State and learnable tokens are encoded once; action flow-matching reuses
their per-layer K/V. This is the architecture introduced after the original
joint-suffix QwenWMv3_LA, split out so old WMv3 checkpoints keep loading.
"""

from starVLA.model.framework.VLM4A.QwenWMv3_LA import Qwen_WMv3_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingForesightV31
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenWMv31_LA")
class Qwen_WMv31_LA(Qwen_WMv3_LA):
    """v5 prefix-KV expert + two-pass learnable-token foresight for Sharla latents."""

    _framework_name = "QwenWMv31_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV31
