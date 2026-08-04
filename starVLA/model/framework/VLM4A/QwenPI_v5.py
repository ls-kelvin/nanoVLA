# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenPI_v5 Framework
===================
Causal Qwen3-VL prefix (flash_attention_2) + Qwen2 action expert conditioned on
per-layer prefix K/V, trained with flow matching.

Unlike QwenPI_v4 -- which reads per-layer hidden states out of a stock VLM and
feeds them to a separate cross-attention DiT -- v5 recomputes each VLM layer's
K/V from its real input and lets the matching expert layer attend over
``[prefix_prompt, suffix]`` with flex_attention (pi0 state/action block mask).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatching
from starVLA.model.modules.vlm.QWen3 import build_qwen3_vl_inputs
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenPI_v5DefaultConfig:
    name: str = "QwenPI_v5"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 28,
            "freeze_vision_encoder": False,
            "train_expert_only": False,
            "gradient_checkpointing": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 16,
            "state_dim": 16,
            "action_horizon": 32,
            "num_inference_timesteps": 10,
            # Prefix is encoded once; r copies only expand expert-side K/V + actions.
            "repeated_diffusion_steps": 8,
            "loss_type": "fm",
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            # "flex" | "sdpa"; sdpa is for mask equivalence checks / fallback.
            "expert_attention": "flex",
            # Per-dim action validity mask (len == action_dim); invalid joints are
            # excluded from the flow-matching loss.
            "action_dim_mask": None,
            "action_expert": {
                # Heads / head_dim MUST match the Qwen3-VL backbone so prefix K/V
                # concatenate cleanly with expert K/V; hidden / intermediate are
                # the expert's own. Expert weights run in fp32.
                "hidden_size": 768,
                "intermediate_size": 2752,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "num_layers": None,
                "adanorm_time": True,
                "final_norm_adanorm": False,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v5")
class Qwen_PI_v5(baseframework):
    """Causal VLM prefix + prefix-KV-conditioned flow-matching action expert."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenPI_v5DefaultConfig, config)
        qwenvl_cfg = self.config.framework.qwenvl
        action_cfg = self.config.framework.action_model

        self.action_model = DualStreamFlowMatching(global_config=self.config)

        vlm_hf_cfg = self.action_model.qwenvl_with_expert.qwenvl.config
        qwenvl_cfg.vl_hidden_dim = int(vlm_hf_cfg.text_config.hidden_size)
        qwenvl_cfg.num_vl_layers = int(vlm_hf_cfg.text_config.num_hidden_layers)

        self.processor = AutoProcessor.from_pretrained(str(qwenvl_cfg.base_vlm))
        self.processor.tokenizer.padding_side = "left"
        vla_data_cfg = self.config.datasets.vla_data
        self.cot_prompt = vla_data_cfg.get("CoT_prompt", None) if "CoT_prompt" in vla_data_cfg else None

        self.action_horizon = int(action_cfg.action_horizon)
        self.action_dim = int(action_cfg.action_dim)
        self.state_dim = int(action_cfg.state_dim)
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 1))

        action_dim_mask = action_cfg.get("action_dim_mask", None)
        if action_dim_mask is not None:
            mask = torch.tensor([float(value) for value in action_dim_mask], dtype=torch.float32)
            if mask.numel() != self.action_dim:
                raise ValueError(
                    f"action_model.action_dim_mask must have length {self.action_dim}, got {mask.numel()}."
                )
            self.register_buffer("action_dim_mask", mask, persistent=False)
        else:
            self.action_dim_mask = None

        if bool(qwenvl_cfg.get("gradient_checkpointing", True)):
            self.action_model.qwenvl_with_expert.qwenvl.gradient_checkpointing_enable(
                {"use_reentrant": False}
            )

    # ------------------------------------------------------------------ #
    # VLM inputs
    # ------------------------------------------------------------------ #
    def _build_vlm_inputs(self, batch_images: List, instructions: List[str], prebuilt_inputs=None) -> dict:
        """Use dataloader-side processor outputs when available, else build them here.

        Returns a fresh mapping either way: callers append latent bridge tokens to
        ``input_ids`` in place, which must not leak back into the batch payload.
        """
        device = self.action_model.qwenvl_with_expert.qwenvl.device
        if prebuilt_inputs is None:
            prebuilt_inputs = build_qwen3_vl_inputs(
                self.processor, batch_images, instructions, cot_prompt=self.cot_prompt
            )
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in prebuilt_inputs.items()
        }

    # ------------------------------------------------------------------ #
    # tensor plumbing
    # ------------------------------------------------------------------ #
    def _prepare_state(self, examples: List[dict], device) -> Optional[torch.Tensor]:
        if "state" not in examples[0]:
            return None
        state = torch.from_numpy(np.array([example["state"] for example in examples])).float()
        if state.ndim == 3:
            state = state[:, -1, :]
        return self._pad_last_dim(state, self.state_dim).to(device)

    def _prepare_actions(self, examples: List[dict], device):
        actions = torch.from_numpy(np.array([example["action"] for example in examples])).float()
        actions = actions[:, -self.action_horizon :, :]
        valid_dim = actions.shape[-1]
        actions = self._pad_last_dim(actions, self.action_dim).to(device)

        action_mask = torch.zeros_like(actions)
        action_mask[..., :valid_dim] = 1.0
        if self.action_dim_mask is not None:
            action_mask = action_mask * self.action_dim_mask.to(device=device, dtype=action_mask.dtype)
        return actions, action_mask

    @staticmethod
    def _pad_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        current = tensor.shape[-1]
        if current > target_dim:
            raise ValueError(f"Data dim {current} exceeds the configured dim {target_dim}.")
        if current == target_dim:
            return tensor
        pad = torch.zeros(*tensor.shape[:-1], target_dim - current, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=-1)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        inputs = self._build_vlm_inputs(
            batch_images, instructions, prebuilt_inputs=examples[0].get("vlm_inputs", None)
        )
        device = inputs["input_ids"].device

        state = self._prepare_state(examples, device)
        actions, action_mask = self._prepare_actions(examples, device)

        if state is None:
            state = torch.zeros(actions.shape[0], self.state_dim, device=device, dtype=actions.dtype)

        out = self.action_model(
            inputs,
            state,
            actions,
            action_mask,
            num_repeats=self.repeated_diffusion_steps,
        )
        return {"action_loss": out["action_loss"]}

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        inputs = self._build_vlm_inputs(batch_images, instructions)
        device = inputs["input_ids"].device

        state = self._prepare_state(examples, device)
        if state is None:
            state = torch.zeros(len(examples), self.state_dim, device=device, dtype=torch.float32)

        # Training reaches the model through the accelerate/DeepSpeed wrapper, which
        # runs forward under autocast; inference is called on the unwrapped module,
        # so the same autocast has to be re-established for low-precision weights.
        param_dtype = self.action_model.state_proj.weight.dtype
        with torch.autocast(
            device.type,
            dtype=param_dtype,
            enabled=param_dtype in (torch.bfloat16, torch.float16),
        ):
            pred_actions = self.action_model.sample_actions(inputs, state)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
