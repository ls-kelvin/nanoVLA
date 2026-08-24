# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv33_LA: query-first two-pass foresight with stop-gradient on queries.

Logical suffix ``[query | state | action]`` (query moved ahead of state),
pi0 block-causal, implemented as two expert passes: an unmodulated query pass
(plain RMSNorm, not AdaRMSNorm at t=0) whose per-layer K/V are detached and
concatenated onto the prefix cache, then a time-conditioned state+action
pass. The action loss never backpropagates into the query tokens.
"""

from starVLA.model.framework.VLM4A.QwenWMv3_LA import Qwen_WMv3_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingForesightV33
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenWMv33_LA")
class Qwen_WMv33_LA(Qwen_WMv3_LA):
    """v5 prefix-KV expert + query-first detached-KV foresight for Sharla latents."""

    _framework_name = "QwenWMv33_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV33
