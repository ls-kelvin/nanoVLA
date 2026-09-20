# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv32_LA_Mem: QwenWMv32_LA with history latent-action memory tokens.

History is no longer prepended to the VLM as main-view images
(``datasets.vla_data.history_frame``). Instead the dataset packs
``la_hist_frames`` -- ``history_stride + 1`` frames spaced one latent-action
stride apart and ending at the current frame -- and the frozen Sharla encoder
turns each consecutive pair into soft-quantized latent-action embeddings
(``weights @ F.normalize(codebook)``, i.e. ``encode_continuous``). A learnable
projection maps them to the VLM hidden size; the tokens are appended to the
tail of the VLM prefix -- right before the expert's state token in the
attention layout -- and enter the prefix K/V that state/action attend to.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenWMv32_LA import Qwen_WMv32_LA
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWMv32_LA_Mem")
class Qwen_WMv32_LA_Mem(Qwen_WMv32_LA):
    """WMv32 foresight + Sharla history latent-action memory in the VLM prefix."""

    _framework_name = "QwenWMv32_LA_Mem"

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)
        self.la_mem_proj = nn.Linear(self.action_model.latent_action_dim, hidden_size)
        logger.info(
            "%s history la_mem tokens: %d per history pair (vl_hidden_dim=%d)",
            self._framework_name,
            self.latent_query_num,
            hidden_size,
        )

    # ------------------------------------------------------------------ #
    # history latent-action memory tokens
    # ------------------------------------------------------------------ #
    def _build_hist_frame_pairs(self, examples: List[dict]):
        frame_pairs = []
        counts = []
        for example in examples:
            frames = example.get("la_hist_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    f"{self._framework_name} requires `la_hist_frames` with at least two frames. "
                    "Set datasets.vla_data.latent_action.history_stride >= 1 (episode-leading "
                    "steps are clamped to the first frame, so history is always available)."
                )
            counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
        if len(set(counts)) != 1:
            raise ValueError(f"la_mem history pair counts must match within a batch, got {counts}.")
        return frame_pairs, counts[0]

    def _build_mem_embs(self, examples: List[dict], device) -> torch.Tensor:
        """History latent actions -> ``[B, h*query_num, vl_hidden_dim]`` prefix tokens."""
        if self.latent_action_encoder is None:
            raise RuntimeError(
                f"{self._framework_name} requires the Sharla latent-action encoder to build "
                "history memory tokens (load_latent_action_encoder must stay enabled)."
            )
        frame_pairs, pair_count = self._build_hist_frame_pairs(examples)
        embeddings = self.latent_action_encoder.encode_continuous(frame_pairs)
        if embeddings.ndim != 3:
            raise ValueError(
                f"{self.latent_action_backend} encode_continuous must return [pairs, query, latent_dim], "
                f"got {tuple(embeddings.shape)}."
            )
        if embeddings.shape[0] != len(frame_pairs):
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {embeddings.shape[0]} frame pairs, "
                f"but {len(frame_pairs)} were provided."
            )
        if embeddings.shape[1] != self.latent_query_num:
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {embeddings.shape[1]} query tokens, "
                f"but latent_query_num={self.latent_query_num}."
            )
        embeddings = embeddings.reshape(
            len(examples), pair_count * self.latent_query_num, embeddings.shape[2]
        )
        # encode_continuous runs under inference_mode; clone so la_mem_proj can backprop.
        embeddings = embeddings.clone().to(device=device, dtype=self.la_mem_proj.weight.dtype)
        return self.la_mem_proj(embeddings)

    # ------------------------------------------------------------------ #
    # prefix hooks (see QwenWMv2_LA)
    # ------------------------------------------------------------------ #
    def _encode_training_prefix(self, inputs: dict, examples: List[dict], device):
        mem_embs = self._build_mem_embs(examples, device)
        prefix, _, prefix_kvs = self.action_model.encode_prefix_context(
            inputs, num_latent_tokens=0, extra_embs=mem_embs
        )
        return prefix, prefix_kvs

    def _encode_merged_prefix(self, merged_inputs: dict, stream_examples: List[List[dict]]):
        device = merged_inputs["input_ids"].device
        mem_embs = [self._build_mem_embs(examples, device) for examples in stream_examples]
        token_counts = {emb.shape[1] for emb in mem_embs}
        if len(token_counts) != 1:
            raise ValueError(
                f"la_mem token counts must match across merged streams, got {sorted(token_counts)}."
            )
        merged_mem = torch.cat(mem_embs, dim=0)
        prefix, _, layer_inputs = self.action_model.encode_prefix_hidden(
            merged_inputs, num_latent_tokens=0, extra_embs=merged_mem
        )
        return prefix, layer_inputs

    def _forward_dual(self, batches: dict, dual_loss_modes: dict | None) -> dict:
        """Fall back to per-stream forwards when history token counts differ."""
        names = [name for name in ("latent", "action") if batches.get(name)]
        if len(names) >= 2:
            counts = set()
            for name in names:
                frames = batches[name][0].get("la_hist_frames", None)
                counts.add(None if frames is None else len(frames))
            if len(counts) != 1:
                modes = dual_loss_modes or {}
                return {
                    "streams": {
                        name: self.forward(
                            batches[name], loss_mode=str(modes.get(name, "joint")).lower()
                        )
                        for name in names
                    }
                }
        return super()._forward_dual(batches, dual_loss_modes)

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

        mem_embs = self._build_mem_embs(examples, device)

        # See QwenPI_v5.predict_action: re-establish autocast for low-precision weights.
        param_dtype = self.action_model.state_proj.weight.dtype
        with torch.autocast(
            device.type,
            dtype=param_dtype,
            enabled=param_dtype in (torch.bfloat16, torch.float16),
        ):
            pred_actions = self.action_model.sample_actions(inputs, state, extra_embs=mem_embs)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
