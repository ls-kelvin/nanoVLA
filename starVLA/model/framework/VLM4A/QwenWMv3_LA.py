# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv3_LA: QwenWMv2_LA with InternVLA-style learnable-token foresight.

Replaces the independent latent flow-matching branch with joint learnable
tokens that share one expert forward with state + noisy actions.

Latent supervision is switchable via ``framework.latent_action.loss_type``:

  - ``embedding`` (default): MSE on Sharla continuous embeddings
  - ``soft_kl``: KL against Sharla soft codebook distributions (PI_v4/v5 style)
"""

from typing import List, Optional

import torch

from starVLA.model.framework.VLM4A.QwenWMv2_LA import Qwen_WMv2_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingForesight
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWMv3_LA")
class Qwen_WMv3_LA(Qwen_WMv2_LA):
    """v5 prefix-KV expert + joint learnable-token foresight for Sharla latents."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.foresight_latent_loss_type = self._get_foresight_latent_loss_type()
        self.codebook_size = int(self.latent_action_cfg.get("codebook_size", 256))
        episode_cache_dir = self.latent_action_cfg.get("episode_cache_dir", None)
        if episode_cache_dir is None:
            try:
                episode_cache_dir = self.config.datasets.vla_data.latent_action.get(
                    "episode_cache_dir", None
                )
            except Exception:
                episode_cache_dir = None
        self.use_cached_soft_distribution = (
            self.foresight_latent_loss_type == "soft_kl"
            and episode_cache_dir is not None
            and str(episode_cache_dir) not in ("", "null", "None")
        )
        if self.foresight_latent_loss_type == "soft_kl":
            if self.latent_action_encoder is not None and hasattr(
                self.latent_action_encoder, "codebook_size"
            ):
                encoder_size = int(self.latent_action_encoder.codebook_size)
                if encoder_size != self.codebook_size:
                    raise ValueError(
                        f"latent_action.codebook_size={self.codebook_size} does not match "
                        f"encoder codebook_size={encoder_size}."
                    )
            logger.info(
                "QwenWMv3_LA foresight latent loss_type=soft_kl (codebook_size=%d, cached=%s)",
                self.codebook_size,
                self.use_cached_soft_distribution,
            )
        else:
            logger.info("QwenWMv3_LA foresight latent loss_type=embedding")

    def _ensure_latent_action_defaults(self) -> None:
        from omegaconf import OmegaConf

        super()._ensure_latent_action_defaults()
        extras = {
            # embedding: MSE on continuous Sharla embeddings (default, preserves WMv3 v1 behavior)
            # soft_kl: KL on codebook soft distributions (PI_v4/v5 style)
            "loss_type": "embedding",
            "codebook_size": 256,
            "episode_cache_dir": None,
            "sharla": {
                "kl_eps": 1.0e-8,
            },
        }
        current = self.config.framework.get("latent_action", {})
        self.config.framework.latent_action = OmegaConf.merge(
            OmegaConf.create(extras),
            current,
        )
        # Re-merge sharla nested defaults without wiping user keys.
        sharla_defaults = {"kl_eps": 1.0e-8}
        sharla_current = self.config.framework.latent_action.get("sharla", {}) or {}
        self.config.framework.latent_action.sharla = OmegaConf.merge(
            OmegaConf.create(sharla_defaults),
            sharla_current,
        )

    def _get_foresight_latent_loss_type(self) -> str:
        loss_type = str(self.config.framework.latent_action.get("loss_type", "embedding")).lower()
        if loss_type not in {"embedding", "soft_kl"}:
            raise ValueError(
                f"framework.latent_action.loss_type must be 'embedding' or 'soft_kl', got {loss_type!r}."
            )
        if loss_type == "soft_kl":
            backend = str(self.config.framework.latent_action.get("backend", "sharla")).lower()
            if backend != "sharla":
                raise ValueError(
                    "QwenWMv3_LA soft_kl currently requires latent_action.backend='sharla', "
                    f"got {backend!r}."
                )
        return loss_type

    def _build_action_model(self) -> DualStreamFlowMatchingForesight:
        """Same latent_dim wiring as WMv2, but build the foresight action model."""
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        latent_dim = self.latent_action_cfg.get("latent_dim", None)
        if latent_dim is None:
            raise ValueError(
                "framework.latent_action.latent_dim is required for QwenWMv3_LA (e.g. the Sharla "
                "codebook_dim). It cannot be auto-inferred from the encoder here because that would "
                "require constructing the action model (which owns the VLM) twice."
            )
        self.config.framework.action_model.latent_action_dim = int(latent_dim)
        loss_type = self._get_foresight_latent_loss_type()
        if loss_type == "soft_kl":
            codebook_size = self.latent_action_cfg.get("codebook_size", None)
            if codebook_size is None:
                raise ValueError(
                    "framework.latent_action.codebook_size is required when loss_type='soft_kl'."
                )
            self.config.framework.action_model.codebook_size = int(codebook_size)
        return DualStreamFlowMatchingForesight(global_config=self.config)

    def _load_latent_norm_stats(self, latent_action_dim: int) -> None:
        """Skip embedding norm stats when foresight uses soft_kl (no embedding targets)."""
        if self._get_foresight_latent_loss_type() == "soft_kl":
            return
        super()._load_latent_norm_stats(latent_action_dim)

    # ------------------------------------------------------------------ #
    # latent targets
    # ------------------------------------------------------------------ #
    def _collect_la_padding_mask(self, examples: List[dict], device) -> torch.Tensor:
        flags = [bool(example.get("la_padded", False)) for example in examples]
        return torch.tensor(
            [0.0 if padded else 1.0 for padded in flags], dtype=torch.float32, device=device
        )

    def _make_soft_targets(self, examples: List[dict], device) -> torch.Tensor:
        """Soft codebook distributions ``[B, N, C]`` for foresight soft_kl."""
        if self.use_cached_soft_distribution and examples and "la_cached_distribution" in examples[0]:
            return self._load_cached_soft_targets(examples, device)

        if self.latent_action_encoder is None:
            raise RuntimeError(
                "Latent-action encoder is not initialized and no la_cached_distribution was provided. "
                "Set datasets.vla_data.latent_action.episode_cache_dir or load the Sharla encoder."
            )
        if not hasattr(self.latent_action_encoder, "encode_distribution"):
            raise RuntimeError(
                f"{self.latent_action_backend} encoder must implement encode_distribution()."
            )

        frame_pairs, counts = self._build_latent_frame_pairs(examples)
        distributions = self.latent_action_encoder.encode_distribution(frame_pairs)
        if distributions.ndim != 3:
            raise ValueError(
                f"{self.latent_action_backend} encode_distribution must return "
                f"[pairs, query, codebook], got {tuple(distributions.shape)}."
            )
        if distributions.shape[0] != len(frame_pairs):
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {distributions.shape[0]} frame pairs, "
                f"but {len(frame_pairs)} were provided."
            )
        if distributions.shape[1] != self.latent_query_num:
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {distributions.shape[1]} query tokens, "
                f"but latent_query_num={self.latent_query_num}."
            )
        if distributions.shape[-1] != self.codebook_size:
            raise ValueError(
                f"Soft distribution codebook size {distributions.shape[-1]} != "
                f"latent_action.codebook_size={self.codebook_size}."
            )
        pair_count = counts[0]
        return distributions.reshape(
            len(examples), pair_count * self.latent_query_num, self.codebook_size
        ).to(device=device, dtype=torch.float32)

    def _load_cached_soft_targets(self, examples: List[dict], device) -> torch.Tensor:
        cached_list = []
        for example in examples:
            cached = example.get("la_cached_distribution", None)
            if cached is None:
                raise RuntimeError(
                    "use_cached_soft_distribution is enabled but sample is missing "
                    "'la_cached_distribution'. Set datasets.vla_data.latent_action.episode_cache_dir."
                )
            if not isinstance(cached, torch.Tensor):
                cached = torch.as_tensor(cached, dtype=torch.float32)
            if cached.ndim != 2:
                raise ValueError(
                    f"la_cached_distribution must be [num_tokens, codebook_size], got {tuple(cached.shape)}."
                )
            if cached.shape[-1] != self.codebook_size:
                raise ValueError(
                    f"Cached soft distribution codebook size {cached.shape[-1]} != "
                    f"latent_action.codebook_size={self.codebook_size}."
                )
            cached_list.append(cached.float())
        token_counts = {item.shape[0] for item in cached_list}
        if len(token_counts) != 1:
            raise ValueError(
                f"Cached soft distribution token counts must match within a batch, got {sorted(token_counts)}."
            )
        return torch.stack(cached_list, dim=0).to(device=device, dtype=torch.float32)

    def _make_latent_targets(self, examples: List[dict], device) -> torch.Tensor:
        if self.foresight_latent_loss_type == "soft_kl":
            return self._make_soft_targets(examples, device)
        return self._make_continuous_targets(examples, device, torch.float32)

    def _compute_stream_losses(
        self,
        prefix: dict,
        prefix_kvs,
        examples: List[dict],
        device,
        compute_action: bool,
        compute_latent: bool,
    ):
        """Joint foresight when both heads are needed; otherwise single-path fallbacks."""
        action_dit_loss = None
        latent_action_loss = None
        action_train_batch_size = 0
        latent_valid_mask = None
        if compute_latent and self.foresight_latent_loss_type == "soft_kl":
            latent_valid_mask = self._collect_la_padding_mask(examples, device)

        if compute_action and compute_latent:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            latent_targets = self._make_latent_targets(examples, device)
            action_dit_loss, latent_action_loss = self.action_model.flow_matching_loss_joint_foresight(
                prefix,
                prefix_kvs,
                state,
                actions,
                action_mask,
                latent_targets,
                num_repeats=self.repeated_diffusion_steps,
                latent_valid_mask=latent_valid_mask,
            )
            if latent_valid_mask is not None:
                latent_action_loss = self._scale_loss_for_global_mean(
                    latent_action_loss, int(latent_valid_mask.sum().item())
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
            latent_action_loss = self.action_model.flow_matching_loss_latent_only(
                prefix, prefix_kvs, latent_targets, latent_valid_mask=latent_valid_mask
            )
            if latent_valid_mask is not None:
                latent_action_loss = self._scale_loss_for_global_mean(
                    latent_action_loss, int(latent_valid_mask.sum().item())
                )

        return action_dit_loss, latent_action_loss, action_train_batch_size
