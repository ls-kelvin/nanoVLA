# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv32_LA: QwenWMv3_LA with sampled-time latent conditioning.

The latent-only foresight path drops the fixed ``foresight_cond`` embedding
and samples a flow-matching timestep like the action stream; latent
supervision uses all ``repeated_diffusion_steps`` copies in both the joint
and latent-only paths. Split out so WMv3 checkpoints keep loading.
"""

from starVLA.model.framework.VLM4A.QwenWMv3_LA import Qwen_WMv3_LA
from starVLA.model.modules.action_model.dual_stream_expert import (
    DualStreamFlowMatchingForesightV32,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenWMv32_LA")
class Qwen_WMv32_LA(Qwen_WMv3_LA):
    """v5 prefix-KV expert + sampled-time latent conditioning for Sharla latents."""

    _framework_name = "QwenWMv32_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV32
