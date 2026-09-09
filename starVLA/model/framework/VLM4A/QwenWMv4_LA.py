# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv4_LA: QwenWMv32_LA with world-model queries feeding a Wan2.1 DiT.

Joint suffix is ``[state | LA query | WM query | action]``. WM query hidden
states condition ``WanVideoBranch``. There is no separate latent-action
inference path; ``predict_action`` keeps both query groups in the suffix.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms.functional import to_tensor

from starVLA.model.framework.VLM4A.QwenWMv32_LA import Qwen_WMv32_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingForesightV4
from starVLA.model.modules.vlm.QWen3 import merge_qwen3_vl_inputs
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    return getattr(cfg, key, default)


@FRAMEWORK_REGISTRY.register("QwenWMv4_LA")
class Qwen_WMv4_LA(Qwen_WMv32_LA):
    """v5 prefix-KV expert + WMv32 latent cond + WAN video branch."""

    _framework_name = "QwenWMv4_LA"
    _action_model_cls = DualStreamFlowMatchingForesightV4

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        wan_cfg = _cfg_get(self.config.framework, "wan", {}) or {}
        self.wan_cfg = wan_cfg
        self.video_loss_weight = float(_cfg_get(wan_cfg, "video_loss_weight", 1.0))
        image_size = _cfg_get(wan_cfg, "video_image_size", [256, 256])
        self.wan_image_size = tuple(int(v) for v in image_size)
        logger.info(
            "%s wan enabled=%s wan_video=%s num_wm_queries=%s video_loss_weight=%s",
            self._framework_name,
            bool(_cfg_get(wan_cfg, "enabled", True)),
            self.action_model.wan_video is not None,
            int(_cfg_get(wan_cfg, "num_wm_queries", 32)),
            self.video_loss_weight,
        )

    def _ensure_wan_defaults(self) -> None:
        from omegaconf import OmegaConf

        extras = {
            "enabled": True,
            "wan_model_path": None,
            "num_wm_queries": 32,
            "video_image_size": [256, 256],
            "video_keys": ["cam_high"],
            "video_loss_weight": 1.0,
            "freeze_dit": False,
            "num_inference_steps": 10,
            "flow_shift": 5.0,
            "gradient_checkpointing": True,
        }
        current = self.config.framework.get("wan", {}) or {}
        self.config.framework.wan = OmegaConf.merge(OmegaConf.create(extras), current)

    def _build_action_model(self):
        self._ensure_wan_defaults()
        return super()._build_action_model()

    def _frame_to_wan_tensor(self, frame) -> torch.Tensor:
        """Convert one PIL/ndarray frame to ``(3, H, W)`` in [-1, 1]."""
        if isinstance(frame, torch.Tensor):
            image = frame.detach().float()
            if image.ndim == 3 and image.shape[0] not in (1, 3):
                image = image.permute(2, 0, 1)
            if image.max() > 1.5:
                image = image / 255.0
        else:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            frame = frame.convert("RGB")
            width, height = self.wan_image_size[1], self.wan_image_size[0]
            if frame.size != (width, height):
                frame = frame.resize((width, height))
            image = to_tensor(frame)
        if tuple(image.shape[-2:]) != self.wan_image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.wan_image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        return image * 2.0 - 1.0

    def _make_wm_frames(self, examples: List[dict], device) -> Optional[torch.Tensor]:
        if self.action_model.wan_video is None:
            return None
        if examples[0].get("wm_cached_latents") is not None:
            return None
        pixels = examples[0].get("wm_pixel_values")
        if pixels is not None:
            if (
                pixels.dtype != torch.uint8 or pixels.ndim != 5
                or pixels.shape[0] != len(examples) or pixels.shape[2] != 3
                or tuple(pixels.shape[-2:]) != self.wan_image_size
            ):
                raise ValueError("wm_pixel_values must be uint8 (B, T, 3, H, W) at WAN image size.")
            # Transfer before converting: 4x fewer H2D bytes than float32.
            pixels = pixels.to(device=device, non_blocking=True)
            if pixels.device.type == "cpu":
                return pixels.float().div_(255.0).mul_(2.0).sub_(1.0)
            # CPU and CUDA division round differently. A 256-value lookup
            # table retains CPU to_tensor's exact fp32 normalization for every
            # uint8 value, without transferring a full float32 video batch.
            # Keep this runtime cache outside buffers so model.to(bfloat16)
            # cannot round it; rebuild lazily if the model changes devices.
            cached = getattr(self, "_wan_pixel_lut", None)
            if cached is None or cached[0] != pixels.device:
                table = torch.arange(256, device="cpu", dtype=torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
                cached = (pixels.device, table.to(pixels.device))
                self._wan_pixel_lut = cached
            return cached[1][pixels.long()]
        stacked = []
        lengths = set()
        for example in examples:
            frames = example.get("wm_frames", None)
            if frames is None:
                raise RuntimeError(
                    "WAN video branch is enabled but sample is missing 'wm_frames'. "
                    "Set datasets.vla_data.wan.enabled=true."
                )
            if any(frame is None for frame in frames):
                raise RuntimeError("wm_frames contains a missing frame; WAN frame packing failed.")
            lengths.add(len(frames))
            stacked.append(torch.stack([self._frame_to_wan_tensor(frame) for frame in frames], dim=0))
        if len(lengths) != 1:
            raise ValueError(f"wm_frames length must match within a batch, got {sorted(lengths)}.")
        return torch.stack(stacked, dim=0).to(device=device, dtype=torch.float32)

    def _scale_loss_for_global_mean(self, loss: torch.Tensor, local_count: int) -> torch.Tensor:
        """Same DDP weighting as WMv2, without a GPU-to-host scalar read."""
        if not dist.is_available() or not dist.is_initialized():
            return loss
        count = torch.tensor(float(local_count), device=loss.device, dtype=loss.dtype)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        scale = dist.get_world_size() * float(local_count) / count.clamp_min(1.0)
        return loss * scale

    def _assemble_losses(
        self,
        action_dit_loss: torch.Tensor | None,
        latent_action_loss: torch.Tensor | None,
        video_loss: torch.Tensor | None,
        action_train_batch_size: int,
    ) -> dict:
        total_loss = None
        if action_dit_loss is not None:
            total_loss = action_dit_loss * float(self.latent_action_cfg.action_loss_weight)
        if latent_action_loss is not None:
            latent_weight = float(self.latent_action_cfg.get("latent_loss_weight", 1.0))
            weighted_latent = latent_action_loss * latent_weight
            total_loss = weighted_latent if total_loss is None else total_loss + weighted_latent
        if video_loss is not None:
            weighted_video = video_loss * float(self.video_loss_weight)
            total_loss = weighted_video if total_loss is None else total_loss + weighted_video
        if total_loss is None:
            raise RuntimeError("No loss was computed. Check loss_mode, latent_action train flags, and wan.")

        out = {
            "total_loss": total_loss,
            "action_loss": total_loss,
            "action_train_batch_size": action_train_batch_size,
        }
        if action_dit_loss is not None:
            out["action_dit_loss"] = action_dit_loss
            out["action_dis_loss"] = action_dit_loss
        if latent_action_loss is not None:
            out["latent_action_loss"] = latent_action_loss
        if video_loss is not None:
            out["video_loss"] = video_loss
        return out

    def _cache_metrics(self, examples, compute_latent):
        if not compute_latent or self.action_model.wan_video is None:
            return {}
        return {"wan_cache_hits": len(examples) if examples[0].get("wm_cached_latents") is not None else 0,
                "wan_cache_samples": len(examples)}

    def _compute_stream_losses(
        self,
        prefix: dict,
        prefix_kvs,
        examples: List[dict],
        device,
        compute_action: bool,
        compute_latent: bool,
    ):
        action_dit_loss = None
        latent_action_loss = None
        video_loss = None
        action_train_batch_size = 0
        latent_valid_mask = None
        latent_valid_count = 0
        if compute_latent and self.foresight_latent_loss_type == "soft_kl":
            latent_valid_mask = self._collect_la_padding_mask(examples, device)
            latent_valid_count = sum(not bool(example.get("la_padded", False)) for example in examples)

        wm_frames = self._make_wm_frames(examples, device)
        wm_latents = examples[0].get("wm_cached_latents")
        wm_cache_keys = examples[0].get("wm_cache_keys")

        if compute_action and compute_latent:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            latent_targets = self._make_latent_targets(examples, device)
            action_dit_loss, latent_action_loss, video_loss = self.action_model.flow_matching_loss_joint_foresight(
                prefix,
                prefix_kvs,
                state,
                actions,
                action_mask,
                latent_targets,
                wm_frames=wm_frames,
                wm_latents=wm_latents,
                wm_cache_keys=wm_cache_keys,
                num_repeats=self.repeated_diffusion_steps,
                latent_valid_mask=latent_valid_mask,
            )
            if latent_valid_mask is not None:
                latent_action_loss = self._scale_loss_for_global_mean(
                    latent_action_loss, latent_valid_count
                )
        elif compute_action:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            action_dit_loss = self.action_model.flow_matching_loss(
                prefix, prefix_kvs, state, actions, action_mask, num_repeats=self.repeated_diffusion_steps
            )
        elif compute_latent:
            latent_targets = self._make_latent_targets(examples, device)
            latent_action_loss, video_loss = self.action_model.flow_matching_loss_latent_only(
                prefix,
                prefix_kvs,
                latent_targets,
                wm_frames=wm_frames,
                wm_latents=wm_latents,
                wm_cache_keys=wm_cache_keys,
                latent_valid_mask=latent_valid_mask,
            )
            if latent_valid_mask is not None:
                latent_action_loss = self._scale_loss_for_global_mean(
                    latent_action_loss, latent_valid_count
                )

        return action_dit_loss, latent_action_loss, video_loss, action_train_batch_size

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        loss_mode = str(kwargs.pop("loss_mode", "joint")).lower()
        if loss_mode == "dual":
            return self._forward_dual(examples, kwargs.pop("dual_loss_modes", None))

        compute_action, compute_latent = self._resolve_loss_flags(loss_mode)

        instructions = [example["lang"] for example in examples]
        inputs = self._build_vlm_inputs(
            [example["image"] for example in examples],
            instructions,
            prebuilt_inputs=examples[0].get("vlm_inputs", None),
        )
        device = inputs["input_ids"].device

        prefix, _, prefix_kvs = self.action_model.encode_prefix_context(inputs, num_latent_tokens=0)
        prefix_kvs = self._maybe_detach_prefix_kvs(prefix_kvs)
        self._print_training_sample_once(examples, instructions)

        action_dit_loss, latent_action_loss, video_loss, action_train_batch_size = self._compute_stream_losses(
            prefix, prefix_kvs, examples, device, compute_action, compute_latent
        )
        out = self._assemble_losses(
            action_dit_loss, latent_action_loss, video_loss, action_train_batch_size
        )
        out.update(self._cache_metrics(examples, compute_latent))
        return out

    def _forward_dual(self, batches: dict, dual_loss_modes: dict | None) -> dict:
        if not isinstance(batches, dict):
            raise TypeError(f"loss_mode='dual' expects a dict of streams, got {type(batches)!r}.")
        modes = dual_loss_modes or {}
        names = [name for name in ("latent", "action") if batches.get(name)]
        if len(names) < 2:
            raise ValueError(f"loss_mode='dual' needs both streams, got {names}.")

        streams = []
        for name in names:
            examples = batches[name]
            loss_mode = str(modes.get(name, "joint")).lower()
            compute_action, compute_latent = self._resolve_loss_flags(loss_mode)
            instructions = [example["lang"] for example in examples]
            inputs = self._build_vlm_inputs(
                [example["image"] for example in examples],
                instructions,
                prebuilt_inputs=examples[0].get("vlm_inputs", None),
            )
            streams.append(
                {
                    "name": name,
                    "examples": examples,
                    "instructions": instructions,
                    "loss_mode": loss_mode,
                    "compute_action": compute_action,
                    "compute_latent": compute_latent,
                    "inputs": inputs,
                }
            )

        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        merged = merge_qwen3_vl_inputs([stream["inputs"] for stream in streams], pad_token_id)
        prefix, _, layer_inputs = self.action_model.encode_prefix_hidden(merged, num_latent_tokens=0)
        device = prefix["position_ids"].device

        row = 0
        outputs = {}
        for stream in streams:
            batch_size = stream["inputs"]["input_ids"].shape[0]
            rows = slice(row, row + batch_size)
            row += batch_size

            stream_prefix = {
                "position_ids": prefix["position_ids"][:, rows],
                "prompt_pad_masks": prefix["prompt_pad_masks"][rows],
                "prompt_len": prefix["prompt_len"],
            }
            prefix_kvs = self.action_model.build_prefix_kvs(prefix, layer_inputs, rows)
            prefix_kvs = self._maybe_detach_prefix_kvs(prefix_kvs)
            self._print_training_sample_once(stream["examples"], stream["instructions"])

            action_dit_loss, latent_action_loss, video_loss, action_train_batch_size = (
                self._compute_stream_losses(
                    stream_prefix,
                    prefix_kvs,
                    stream["examples"],
                    device,
                    stream["compute_action"],
                    stream["compute_latent"],
                )
            )
            outputs[stream["name"]] = self._assemble_losses(
                action_dit_loss, latent_action_loss, video_loss, action_train_batch_size
            )
            outputs[stream["name"]].update(self._cache_metrics(stream["examples"], stream["compute_latent"]))

        return {"streams": outputs}
