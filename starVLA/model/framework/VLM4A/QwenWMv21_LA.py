# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv21_LA: QwenWMv2_LA with a joint state|LA|action denoising suffix.

WMv2 runs latent and action as two independent flow-matching streams (separate
forward passes, separate noise/timesteps). WMv21 instead places both in ONE
suffix and denoises them together (see ``DualStreamFlowMatchingJointLA``):

  joint suffix: [state(1), latent(N, noisy), action(chunk, noisy)]

Block-causal attention: state opens block 1, the first latent token opens
block 2 (covering latent AND action), so state attends to itself only while
latent and action attend to each other bidirectionally (and to state). Both
streams share the same flow-matching timestep, so training is a single expert
pass and inference is a single Euler loop with a shared number of denoising
steps.

Latent-only streams (robot types excluded from action training) keep WMv2's
independent latent suffix, mirroring WMv3's joint/latent-only split.
"""

from typing import List, Optional

import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenWMv2_LA import Qwen_WMv2_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingJointLA
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWMv21_LA")
class Qwen_WMv21_LA(Qwen_WMv2_LA):
    """v5 prefix-KV expert + joint latent/action flow-matching denoising suffix."""

    def _build_action_model(self) -> DualStreamFlowMatchingJointLA:
        """Same latent_dim wiring as WMv2, but build the joint-LA action model."""
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        latent_dim = self.latent_action_cfg.get("latent_dim", None)
        if latent_dim is None:
            raise ValueError(
                "framework.latent_action.latent_dim is required for QwenWMv21_LA (e.g. the Sharla "
                "codebook_dim). It cannot be auto-inferred from the encoder here because that would "
                "require constructing the action model (which owns the VLM) twice."
            )
        self.config.framework.action_model.latent_action_dim = int(latent_dim)
        return DualStreamFlowMatchingJointLA(global_config=self.config)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _compute_stream_losses(
        self,
        prefix: dict,
        prefix_kvs,
        examples: List[dict],
        device,
        compute_action: bool,
        compute_latent: bool,
    ):
        """Joint latent+action denoising when both heads are needed; WMv2 fallbacks otherwise."""
        action_dit_loss = None
        latent_action_loss = None
        action_train_batch_size = 0

        if compute_action and compute_latent:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            latent_targets = self._make_continuous_targets(examples, device, torch.float32)
            action_dit_loss, latent_action_loss = self.action_model.flow_matching_loss_joint(
                prefix,
                prefix_kvs,
                state,
                actions,
                action_mask,
                latent_targets,
                num_repeats=self.repeated_diffusion_steps,
            )
        elif compute_action:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            action_dit_loss = self.action_model.flow_matching_loss(
                prefix, prefix_kvs, state, actions, action_mask, num_repeats=self.repeated_diffusion_steps
            )
        elif compute_latent:
            latent_targets = self._make_continuous_targets(examples, device, torch.float32)
            latent_action_loss = self.action_model.flow_matching_loss_latent(
                prefix, prefix_kvs, latent_targets, num_repeats=self.repeated_diffusion_steps
            )
        return action_dit_loss, latent_action_loss, action_train_batch_size

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _inference_latent_token_count(self, examples: List[dict]) -> int:
        """Latent token count for joint sampling: (frame pairs) * latent_query_num.

        Mirrors the dataloader's offset construction
        (``hdf5_la_dataset`` / ``lerobot_la_datasets``): offsets span
        ``range(0, horizon+1, stride)`` plus an optional terminal frame, and
        each consecutive pair yields ``latent_query_num`` tokens. An explicit
        ``framework.latent_action.inference_latent_tokens`` overrides the
        derivation.
        """
        override = self.latent_action_cfg.get("inference_latent_tokens", None)
        if override is not None:
            return int(override)

        la_cfg = self.config.datasets.vla_data.get("latent_action", {}) or {}
        robot_type = examples[0].get("robot_type", None)
        robot_type = str(robot_type) if robot_type is not None else None

        stride = int(la_cfg.get("stride", 4))
        horizon = la_cfg.get("horizon", None)
        horizon = int(horizon) if horizon is not None else self.action_horizon
        if robot_type:
            stride_overrides = la_cfg.get("stride_overrides", None) or {}
            robot_stride = stride_overrides.get(robot_type, None)
            if robot_stride is not None:
                stride = int(robot_stride)
            horizon_overrides = la_cfg.get("horizon_overrides", None) or {}
            robot_horizon = horizon_overrides.get(robot_type, None) or {}
            if robot_horizon.get("frame_stride", None) is not None:
                stride = int(robot_horizon["frame_stride"])
            if robot_horizon.get("horizon", None) is not None:
                horizon = int(robot_horizon["horizon"])

        offsets = list(range(0, horizon + 1, stride))
        if bool(la_cfg.get("include_terminal_frame", True)) and offsets[-1] != horizon:
            offsets.append(horizon)
        pair_count = len(offsets) - 1
        if pair_count < 1:
            raise ValueError(
                f"Resolved latent-action horizon/stride give no frame pairs: horizon={horizon}, stride={stride}."
            )
        return pair_count * self.latent_query_num

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Joint Euler sampling: latent and action denoise together (shared steps)."""
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

        num_latent_tokens = self._inference_latent_token_count(examples)

        # Training reaches the model through the accelerate/DeepSpeed wrapper, which
        # runs forward under autocast; inference is called on the unwrapped module,
        # so the same autocast has to be re-established for low-precision weights.
        param_dtype = self.action_model.state_proj.weight.dtype
        with torch.autocast(
            device.type,
            dtype=param_dtype,
            enabled=param_dtype in (torch.bfloat16, torch.float16),
        ):
            pred_actions, pred_latent = self.action_model.sample_actions(
                inputs,
                state,
                num_latent_tokens=num_latent_tokens,
                return_latent=True,
            )
        return {
            "normalized_actions": pred_actions.detach().float().cpu().numpy(),
            "predicted_latent": pred_latent.detach().float().cpu().numpy(),
        }
