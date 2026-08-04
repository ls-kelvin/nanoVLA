# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenPI_v4 Framework
===================
QwenVL + layer-wise cross-DiT flow-matching action head.

This variant keeps the QwenMetaQuery action-head shape: DiT inner hidden size can
stay smaller than the VLM hidden size, and cross_attention_dim is the true VLM
hidden size. Unlike QwenMetaQuery, it does not append or extract metaquery
tokens. The full non-padding QwenVL prompt sequence is used directly as the
cross-attention context for the action model.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.LayerwiseMetaqueryFM_ActionHeader import (
    LayerwiseMetaqueryFlowmatchingActionHead,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenPI_v4DefaultConfig:
    name: str = "QwenPI_v4"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 28,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 16,
            "state_dim": 16,
            "action_horizon": 32,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 0,
            "repeated_diffusion_steps": 16,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 32,
            "diffusion_model_cfg": {
                "input_embedding_dim": 1024,
                "attention_head_dim": 32,
                "num_attention_heads": 32,
                "norm_type": "norm",
                "dropout": 0.0,
                "final_dropout": False,
                "interleave_self_attention": True,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v4")
class Qwen_PI_v4(baseframework):
    """VLM + prompt-sequence-conditioned layer-wise flow-matching action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenPI_v4DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        action_cfg = self.config.framework.action_model
        dit_cfg = action_cfg.diffusion_model_cfg
        dit_cfg.cross_attention_dim = llm_hidden_size
        dit_cfg.num_layers = num_vl_layers
        dit_cfg.output_dim = int(action_cfg.action_dim)

        self.action_model = LayerwiseMetaqueryFlowmatchingActionHead(global_config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.action_horizon = int(action_cfg.action_horizon)
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 16))

    def _build_vlm_inputs(self, batch_images: List, instructions: List[str], prebuilt_inputs=None):
        """Use dataloader-side processor outputs when available, else build them here.

        Returns a fresh mapping either way: callers append latent bridge tokens to
        ``input_ids`` in place, which must not leak back into the batch payload.
        """
        if prebuilt_inputs is None:
            return self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        device = self.qwen_vl_interface.model.device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in prebuilt_inputs.items()
        }

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str], prebuilt_inputs=None
    ) -> tuple:
        """Run QwenVL and return layer-wise prompt hidden states plus padding mask."""
        inputs = self._build_vlm_inputs(batch_images, instructions, prebuilt_inputs)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                # Only ``hidden_states`` are consumed; keep the vocab projection to a
                # single position instead of running it over the whole sequence.
                logits_to_keep=1,
            )
            vl_embs_list = list(outputs.hidden_states[-self.num_action_dit_layers :])
        return vl_embs_list, attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        vl_embs_list, attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions, prebuilt_inputs=examples[0].get("vlm_inputs", None)
        )
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            r = self.repeated_diffusion_steps
            actions_target = actions_target.repeat(r, 1, 1)
            # ``vl_embs_list`` / ``attention_mask`` stay at the un-repeated batch.
            # The action head folds the ``r`` diffusion repeats into the DiT query
            # sequence, so cross-attention K/V are projected once instead of once
            # per repeat.

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(r, 1, 1)

            action_loss = self.action_model(
                vl_embs_list,
                actions_target,
                state_repeated,
                encoder_attention_mask=attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state_t,
                encoder_attention_mask=attention_mask,
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
