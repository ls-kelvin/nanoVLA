"""QwenMetaQuery with latent-action auxiliary prediction head.

Replaces the cache-split design with a lightweight LatentPredictor that uses
N x Qwen3VLTextDecoderLayer blocks to predict latent-action codes from the
metaquery last hidden states.  The latent prediction is purely an auxiliary
training loss — inference is identical to vanilla QwenMetaQuery.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenMetaQuery import Qwen_MetaQuery
from starVLA.model.modules.latent_action import build_latent_action_encoder
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
#  LatentPredictor — Qwen3VL TextDecoderLayer causal prediction head
# ---------------------------------------------------------------------------
class LatentPredictor(nn.Module):
    """Predict latent-action codes from metaquery last hidden states.

    Architecture: learnable query tokens are appended after the 64 metaquery
    hidden states, then processed by N causal Qwen3VL TextDecoderLayer blocks.
    The query positions at the output are classified into codebook indices.

    When ``codes_per_query > 1`` (e.g. villax backend), each query position
    predicts multiple VQ codes via a wider classifier head.
    """

    def __init__(
        self,
        vlm_text_config,
        num_latent_codes: int,
        codebook_size: int,
        num_blocks: int = 2,
        codes_per_query: int = 1,
    ):
        super().__init__()
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextRMSNorm,
            Qwen3VLTextRotaryEmbedding,
        )

        self.num_latent_codes = num_latent_codes
        self.codebook_size = codebook_size
        self.codes_per_query = codes_per_query
        hidden_size = vlm_text_config.hidden_size

        self.queries = nn.Parameter(
            torch.randn(num_latent_codes, hidden_size) * vlm_text_config.initializer_range
        )

        predictor_config = copy.deepcopy(vlm_text_config)
        predictor_config.num_hidden_layers = num_blocks
        if not hasattr(predictor_config, "_attn_implementation"):
            predictor_config._attn_implementation = "eager"

        self.blocks = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config=predictor_config, layer_idx=i) for i in range(num_blocks)]
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=predictor_config)
        self.final_norm = Qwen3VLTextRMSNorm(hidden_size, eps=vlm_text_config.rms_norm_eps)
        self.classifier = nn.Linear(hidden_size, codes_per_query * codebook_size)

    def forward(self, meta_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta_hidden: ``[B, num_metaqueries, H]`` from VLM last layer.
        Returns:
            If codes_per_query == 1:
                logits: ``[B, num_latent_codes, codebook_size]``.
            If codes_per_query > 1:
                logits: ``[B, num_latent_codes, codes_per_query, codebook_size]``.
        """
        B = meta_hidden.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        hidden = torch.cat([meta_hidden, queries], dim=1)

        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(hidden, position_ids)

        for block in self.blocks:
            hidden = block(hidden, position_embeddings=position_embeddings)

        latent_hidden = hidden[:, -self.num_latent_codes :, :]
        logits = self.classifier(self.final_norm(latent_hidden))

        if self.codes_per_query > 1:
            logits = logits.reshape(B, self.num_latent_codes, self.codes_per_query, self.codebook_size)

        return logits


# ---------------------------------------------------------------------------
#  Default config
# ---------------------------------------------------------------------------
@dataclass
class QwenMetaQueryLADefaults:
    latent_action: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "backend": "unit",
            "loss_type": "auto",
            "groot_tokenizer_path": None,
            "dinov2_path_override": None,
            "num_bridge_tokens": 8,
            "num_codebooks": 2,
            "codebook_size": 128,
            "label_smoothing": 0.0,
            "loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "action_train_robot_types": None,
            "image_size": [224, 224],
            "predictor": {"num_blocks": 2},
            "univla": {
                "ckpt_path": None,
                "model_dim": 768,
                "latent_dim": 128,
                "patch_size": 14,
                "enc_blocks": 12,
                "dec_blocks": 12,
                "num_heads": 12,
                "dropout": 0.0,
                "image_size": [224, 224],
                "strict_load": True,
            },
            "softvq": {
                "config_path": None,
                "ckpt_path": None,
                "image_size": [256, 256],
                "strict_load": False,
                "kl_eps": 1e-8,
            },
            "villax": {
                "ckpt_path": None,
                "image_size": [224, 224],
                "d_t": 8,
            },
        }
    )


# ---------------------------------------------------------------------------
#  Framework
# ---------------------------------------------------------------------------
@FRAMEWORK_REGISTRY.register("QwenMetaQuery_LA")
class Qwen_MetaQuery_LA(Qwen_MetaQuery):
    """QwenMetaQuery with a latent-action auxiliary prediction head.

    Training adds a LatentPredictor (N x Qwen3VL TextDecoderLayer) on top of
    the metaquery last hidden states to predict latent-action codes as an
    auxiliary CE/KL loss.  Inference is identical to vanilla QwenMetaQuery.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        self.latent_action_enabled = bool(self.latent_action_cfg.get("enabled", True))
        self.latent_action_backend = str(self.latent_action_cfg.get("backend", "unit")).lower()

        self.latent_action_encoder = None
        self.num_bridge_tokens = int(self.latent_action_cfg.num_bridge_tokens)
        self.num_codebooks = int(self.latent_action_cfg.num_codebooks)
        self.codebook_size = int(self.latent_action_cfg.codebook_size)
        self.latent_action_loss_type = self._get_latent_action_loss_type()
        self.label_smoothing = float(self.latent_action_cfg.get("label_smoothing", 0.0))

        self.latent_predictor = None
        self.codes_per_query = self.num_bridge_tokens if self.latent_action_backend == "villax" else 1
        if self.latent_action_enabled:
            self.latent_action_encoder = build_latent_action_encoder(self.config)
            num_latent_codes = self._compute_num_latent_codes()
            num_blocks = int(self.latent_action_cfg.get("predictor", {}).get("num_blocks", 2))

            vlm_hf_cfg = self.qwen_vl_interface.model.config
            text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)

            self.latent_predictor = LatentPredictor(
                vlm_text_config=text_cfg,
                num_latent_codes=num_latent_codes,
                codebook_size=self.codebook_size,
                num_blocks=num_blocks,
                codes_per_query=self.codes_per_query,
            )
            logger.info(
                "LatentPredictor: %d blocks, %d latent codes, codebook_size=%d, "
                "codes_per_query=%d, params=%.2fM",
                num_blocks,
                num_latent_codes,
                self.codebook_size,
                self.codes_per_query,
                sum(p.numel() for p in self.latent_predictor.parameters()) / 1e6,
            )

        self._printed_training_sample = False

    # ------------------------------------------------------------------ #
    # Config helpers
    # ------------------------------------------------------------------ #
    def _ensure_latent_action_defaults(self) -> None:
        from omegaconf import OmegaConf

        current = self.config.framework.get("latent_action", {})
        backend = str(current.get("backend", QwenMetaQueryLADefaults().latent_action["backend"])).lower()
        backend_defaults = {}
        if backend == "univla":
            backend_defaults = {
                "loss_type": "ce",
                "num_bridge_tokens": 4,
                "num_codebooks": 1,
                "codebook_size": 16,
            }
        elif backend == "softvq":
            backend_defaults = {
                "loss_type": "soft_kl",
                "num_bridge_tokens": 1,
                "num_codebooks": 1,
                "codebook_size": 64,
            }
        elif backend in {"unit", "groot_unit"}:
            backend_defaults = {"loss_type": "ce"}
        elif backend == "villax":
            backend_defaults = {
                "loss_type": "ce",
                "num_bridge_tokens": 4,
                "num_codebooks": 1,
                "codebook_size": 32,
            }

        defaults = OmegaConf.create(QwenMetaQueryLADefaults().latent_action)
        self.config.framework.latent_action = OmegaConf.merge(
            defaults, OmegaConf.create(backend_defaults), current
        )

    def _get_latent_action_loss_type(self) -> str:
        loss_type = str(self.latent_action_cfg.get("loss_type", "auto")).lower()
        if loss_type == "auto":
            if self.latent_action_backend in ("univla", "unit", "groot_unit", "villax"):
                loss_type = "ce"
            elif self.latent_action_backend == "softvq":
                loss_type = "soft_kl"
            else:
                loss_type = "ce"
        if loss_type == "ar_lm" or loss_type == "bridge_ce":
            loss_type = "ce"
        if loss_type not in {"ce", "soft_kl"}:
            raise ValueError(f"latent_action.loss_type must be 'ce' or 'soft_kl', got {loss_type!r}.")
        if loss_type == "soft_kl" and self.latent_action_backend != "softvq":
            raise ValueError("Use latent_action.backend='softvq' with latent_action.loss_type='soft_kl'.")
        return loss_type

    def _compute_num_latent_codes(self) -> int:
        la_data_cfg = self.config.datasets.vla_data.latent_action
        stride = int(la_data_cfg.get("stride", 4))
        la_horizon = int(self.config.framework.action_model.action_horizon)
        offsets = list(range(0, la_horizon + 1, stride))
        if bool(la_data_cfg.get("include_terminal_frame", True)) and offsets[-1] != la_horizon:
            offsets.append(la_horizon)
        num_windows = len(offsets) - 1

        if self.latent_action_backend == "villax":
            d_t = int(self.latent_action_cfg.get("villax", {}).get("d_t", 8))
            num_latent_codes = num_windows * (d_t - 1)
        else:
            num_latent_codes = num_windows * self.num_bridge_tokens

        if num_latent_codes <= 0:
            raise ValueError(
                f"Computed num_latent_codes={num_latent_codes} (num_windows={num_windows}, "
                f"backend={self.latent_action_backend}). Check stride/action_horizon."
            )
        return num_latent_codes

    # ------------------------------------------------------------------ #
    # State dict: exclude frozen latent encoder
    # ------------------------------------------------------------------ #
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        keys_to_remove = [k for k in sd if k.startswith("latent_action_encoder.")]
        for k in keys_to_remove:
            del sd[k]
        return sd

    # ------------------------------------------------------------------ #
    # Ground-truth target generation (kept from old design)
    # ------------------------------------------------------------------ #
    def _make_visual_only_vq_targets(self, examples: List[dict]) -> torch.LongTensor:
        if self.latent_action_encoder is None:
            raise RuntimeError("Latent-action encoder is not initialized.")

        frame_pairs = []
        pair_counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    "QwenMetaQuery_LA requires `la_frames` with at least two frames. "
                    "Use datasets.vla_data.dataset_py=lerobot_la_datasets and enable latent_action."
                )
            pair_counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))

        if len(set(pair_counts)) != 1:
            raise ValueError(
                "QwenMetaQuery_LA requires the same number of latent-action frame pairs per batch "
                f"because targets are batched without padding, got pair counts {pair_counts}."
            )
        pair_count = pair_counts[0]

        vq_indices = self.latent_action_encoder.encode(frame_pairs)
        if vq_indices.ndim == 2:
            vq_indices = vq_indices.unsqueeze(-1)
        if vq_indices.shape[0] != len(frame_pairs):
            raise ValueError(
                f"Latent-action encoder returned {vq_indices.shape[0]} pairs, "
                f"but {len(frame_pairs)} frame pairs were provided."
            )
        if vq_indices.shape[1] != self.num_bridge_tokens:
            raise ValueError(
                f"Latent-action encoder returned {vq_indices.shape[1]} VQ tokens, "
                f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if vq_indices.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"Latent-action encoder returned {vq_indices.shape[-1]} codebooks, "
                f"but latent_action.num_codebooks={self.num_codebooks}."
            )
        if vq_indices.numel() > 0:
            max_idx = int(vq_indices.max().item())
            min_idx = int(vq_indices.min().item())
            if min_idx < 0 or max_idx >= self.codebook_size:
                raise ValueError(
                    f"Latent-action indices must be in [0, {self.codebook_size}), "
                    f"got min={min_idx}, max={max_idx}."
                )
        vq_indices = vq_indices.reshape(len(examples), pair_count * self.num_bridge_tokens, self.num_codebooks)
        return vq_indices.to(self.metaquery_suffix_ids.device)

    def _make_villax_vq_targets(self, examples: List[dict]) -> torch.LongTensor:
        """Generate VQ targets using villa-x clip-based encoding.

        la_frames for villax is a list of clips (list of frame lists per window).
        Returns targets shaped ``[B, num_queries, codes_per_query]``.
        """
        if self.latent_action_encoder is None:
            raise RuntimeError("Latent-action encoder is not initialized.")

        all_clips = []
        window_counts = []
        for example in examples:
            clips = example.get("la_frames", None)
            if clips is None or not isinstance(clips, list) or len(clips) == 0:
                raise ValueError(
                    "QwenMetaQuery_LA (villax) requires `la_frames` as a list of clips. "
                    "Set datasets.vla_data.latent_action.full_window=true."
                )
            if not isinstance(clips[0], list):
                raise ValueError(
                    "QwenMetaQuery_LA (villax) requires `la_frames` to be list of lists "
                    "(each inner list is a window's frames). Got a flat list instead. "
                    "Set datasets.vla_data.latent_action.full_window=true."
                )
            window_counts.append(len(clips))
            all_clips.extend(clips)

        if len(set(window_counts)) != 1:
            raise ValueError(
                "QwenMetaQuery_LA (villax) requires the same number of windows per batch, "
                f"got window counts {window_counts}."
            )

        vq_indices = self.latent_action_encoder.encode_clips(all_clips)
        # vq_indices: [total_clips, d_t-1, num_learned_tokens]

        num_windows = window_counts[0]
        d_t_minus_1 = vq_indices.shape[1]
        codes_per_query = vq_indices.shape[2]
        num_queries = num_windows * d_t_minus_1

        if vq_indices.numel() > 0:
            max_idx = int(vq_indices.max().item())
            min_idx = int(vq_indices.min().item())
            if min_idx < 0 or max_idx >= self.codebook_size:
                raise ValueError(
                    f"villa-x VQ indices must be in [0, {self.codebook_size}), "
                    f"got min={min_idx}, max={max_idx}."
                )

        # Reshape: [total_clips, d_t-1, codes] → [B, num_windows*(d_t-1), codes]
        vq_indices = vq_indices.reshape(len(examples), num_queries, codes_per_query)
        return vq_indices.to(self.metaquery_suffix_ids.device)

    def _make_softvq_targets(self, examples: List[dict]) -> torch.Tensor:
        if self.latent_action_encoder is None:
            raise RuntimeError("Latent-action encoder is not initialized.")
        if not hasattr(self.latent_action_encoder, "encode_distribution"):
            raise RuntimeError("SoftVQ latent-action encoder must implement encode_distribution().")

        frame_pairs = []
        pair_counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    "QwenMetaQuery_LA soft_kl requires `la_frames` with at least two frames. "
                    "Use datasets.vla_data.dataset_py=lerobot_la_datasets and enable latent_action."
                )
            pair_counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))

        if len(set(pair_counts)) != 1:
            raise ValueError(
                "QwenMetaQuery_LA requires the same number of SoftVQ frame pairs per batch "
                f"because targets are batched without padding, got pair counts {pair_counts}."
            )
        pair_count = pair_counts[0]

        distributions = self.latent_action_encoder.encode_distribution(frame_pairs)
        if distributions.ndim != 3:
            raise ValueError(
                f"SoftVQ encoder must return [num_pairs, tokens, codebook], got {tuple(distributions.shape)}."
            )
        if distributions.shape[0] != len(frame_pairs):
            raise ValueError(
                f"SoftVQ encoder returned {distributions.shape[0]} pairs, "
                f"but {len(frame_pairs)} frame pairs were provided."
            )
        if distributions.shape[1] != self.num_bridge_tokens:
            raise ValueError(
                f"SoftVQ encoder returned {distributions.shape[1]} tokens, "
                f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if distributions.shape[-1] != self.codebook_size:
            raise ValueError(
                f"SoftVQ encoder returned codebook size {distributions.shape[-1]}, "
                f"but latent_action.codebook_size={self.codebook_size}."
            )
        distributions = distributions.reshape(len(examples), pair_count * self.num_bridge_tokens, self.codebook_size)
        return distributions.to(self.metaquery_suffix_ids.device)

    def _collect_la_padding_mask(self, examples: List[dict]) -> torch.Tensor:
        flags = [bool(example.get("la_padded", False)) for example in examples]
        return torch.tensor([0.0 if padded else 1.0 for padded in flags], dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Cross-embodiment action training selection (kept from old design)
    # ------------------------------------------------------------------ #
    def _get_action_train_robot_types(self) -> set[str] | None:
        robot_types = self.latent_action_cfg.get("action_train_robot_types", None)
        if robot_types is None:
            return None
        if isinstance(robot_types, str):
            robot_types = robot_types.strip()
            if robot_types == "" or robot_types.lower() in {"none", "null", "all"}:
                return None
            if robot_types.startswith("[") and robot_types.endswith("]"):
                robot_types = robot_types[1:-1]
            robot_types = [item.strip().strip("'\"") for item in robot_types.split(",")]

        allowed = {str(rt) for rt in robot_types if str(rt)}
        return allowed if allowed else None

    def _select_action_training_batch(
        self,
        examples: List[dict],
        meta_embs: List[torch.Tensor],
    ) -> tuple[List[dict], List[torch.Tensor]]:
        action_train_robot_types = self._get_action_train_robot_types()
        if action_train_robot_types is None:
            return examples, meta_embs

        selected_indices = [
            idx
            for idx, example in enumerate(examples)
            if str(example.get("robot_type", "")) in action_train_robot_types
        ]
        if len(selected_indices) == len(examples):
            return examples, meta_embs
        if not selected_indices:
            return [], []

        index_tensor = torch.tensor(selected_indices, device=meta_embs[-1].device, dtype=torch.long)
        selected_examples = [examples[idx] for idx in selected_indices]
        selected_meta_embs = [hidden.index_select(0, index_tensor) for hidden in meta_embs]
        return selected_examples, selected_meta_embs

    # ------------------------------------------------------------------ #
    # Loss helpers
    # ------------------------------------------------------------------ #
    def _action_loss_from_meta_embs(
        self,
        examples: List[dict],
        meta_embs: List[torch.Tensor],
    ) -> torch.Tensor:
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        base_hidden = meta_embs[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            r = self.repeated_diffusion_steps
            actions_target = actions_target.repeat(r, 1, 1)
            meta_embs_r = [h.repeat(r, 1, 1) for h in meta_embs]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(r, 1, 1)

            return self.action_model(meta_embs_r, actions_target, state_repeated)

    def _compute_latent_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute latent prediction loss.

        Args:
            logits:  ``[B, N, codebook_size]`` (codes_per_query=1) or
                     ``[B, N, codes_per_query, codebook_size]`` (codes_per_query>1)
            targets: ``[B, N, num_codebooks]`` (discrete CE, codes_per_query=1) or
                     ``[B, N, codes_per_query]`` (discrete CE, codes_per_query>1) or
                     ``[B, N, codebook_size]`` (soft KL)
            valid_mask: ``[B]`` per-sample mask (1=keep, 0=skip) for soft_kl.
        """
        if self.latent_action_loss_type == "ce":
            if self.codes_per_query > 1:
                # logits: [B, N, codes_per_query, codebook_size]
                # targets: [B, N, codes_per_query]
                loss = F.cross_entropy(
                    logits.reshape(-1, self.codebook_size).float(),
                    targets.reshape(-1).long().to(logits.device),
                    label_smoothing=self.label_smoothing,
                )
            else:
                target_indices = targets[..., 0].long()
                loss = F.cross_entropy(
                    logits.reshape(-1, self.codebook_size).float(),
                    target_indices.reshape(-1).to(logits.device),
                    label_smoothing=self.label_smoothing,
                )
            return {"latent_action_loss": loss, "latent_ce_loss": loss}
        elif self.latent_action_loss_type == "soft_kl":
            eps = float(self.latent_action_cfg.get("softvq", {}).get("kl_eps", 1e-8))
            target_probs = targets.float().clamp_min(eps)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
            log_probs = F.log_softmax(logits.float(), dim=-1)

            kl_per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=1)

            if valid_mask is None:
                valid_mask = torch.ones_like(kl_per_sample)
            else:
                valid_mask = valid_mask.to(kl_per_sample.device, dtype=kl_per_sample.dtype)
            denom = valid_mask.sum().clamp_min(1.0)
            kl_loss = (kl_per_sample * valid_mask).sum() / denom

            valid_count = int(valid_mask.sum().item())
            kl_loss = self._scale_action_loss_for_global_mean(kl_loss, valid_count)

            entropy = -(target_probs * target_probs.log()).sum(dim=-1).mean(dim=1)
            entropy = (entropy * valid_mask).sum() / denom
            return {
                "latent_action_loss": kl_loss,
                "latent_softvq_kl_loss": kl_loss,
                "latent_softvq_target_entropy": entropy,
            }
        else:
            raise ValueError(f"Unknown latent_action_loss_type: {self.latent_action_loss_type}")

    def _scale_action_loss_for_global_mean(
        self,
        loss: torch.Tensor,
        local_count: int,
    ) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return loss

        count = torch.tensor(float(local_count), device=loss.device, dtype=loss.dtype)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0:
            return loss * 0.0
        return loss * (dist.get_world_size() * float(local_count) / count)

    # ------------------------------------------------------------------ #
    # Debug logging
    # ------------------------------------------------------------------ #
    def _print_training_sample_once(
        self,
        examples: List[dict],
        instructions: list[str],
        gt_indices: torch.LongTensor | None = None,
    ) -> None:
        if self._printed_training_sample or not self.training:
            return
        if not logger.is_rank_zero():
            return

        example = examples[0]
        episode_index = example.get("episode_index", None)
        step_index = example.get("step_index", None)
        episode_path = example.get("episode_path", None)
        la_frame_offsets = example.get("la_frame_offsets", None)

        gt_info = None
        if gt_indices is not None:
            gt_info = gt_indices[0].detach().cpu().tolist()

        predictor_info = "None"
        if self.latent_predictor is not None:
            predictor_info = (
                f"{len(self.latent_predictor.blocks)} blocks, "
                f"{self.latent_predictor.num_latent_codes} codes, "
                f"codebook_size={self.latent_predictor.codebook_size}"
            )

        logger.info(
            "First training sample (episode_index=%s, step_index=%s, episode_path=%s):\n"
            "  instruction: %s\n"
            "  la_frame_offsets: %s\n"
            "  latent predictor: %s\n"
            "  GT latent indices (first sample): %s",
            episode_index,
            step_index,
            episode_path,
            instructions[0],
            la_frame_offsets,
            predictor_info,
            gt_info,
        )
        self._printed_training_sample = True

    # ------------------------------------------------------------------ #
    # Train / inference
    # ------------------------------------------------------------------ #
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        meta_embs = self._encode_metaquery_hidden_states(batch_images, instructions)

        action_examples, action_meta_embs = self._select_action_training_batch(examples, meta_embs)
        action_train_batch_size = len(action_examples)
        if action_examples:
            action_dit_loss = self._action_loss_from_meta_embs(action_examples, action_meta_embs)
        else:
            action_dit_loss = self._action_loss_from_meta_embs(
                examples[:1],
                [hidden[:1] for hidden in meta_embs],
            ) * 0.0
        action_dit_loss = self._scale_action_loss_for_global_mean(action_dit_loss, action_train_batch_size)

        latent_outputs: dict = {}
        latent_action_loss = None
        if self.latent_action_enabled and self.latent_predictor is not None:
            if self.latent_action_loss_type == "soft_kl":
                gt_targets = self._make_softvq_targets(examples)
                la_valid_mask = self._collect_la_padding_mask(examples)
                gt_indices_for_log = gt_targets.argmax(dim=-1).unsqueeze(-1)
            elif self.latent_action_backend == "villax":
                gt_targets = self._make_villax_vq_targets(examples)
                la_valid_mask = None
                gt_indices_for_log = gt_targets
            else:
                gt_targets = self._make_visual_only_vq_targets(examples)
                la_valid_mask = None
                gt_indices_for_log = gt_targets

            self._print_training_sample_once(examples, instructions, gt_indices_for_log)

            meta_last = meta_embs[-1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_logits = self.latent_predictor(meta_last)

            latent_outputs = self._compute_latent_loss(pred_logits, gt_targets, la_valid_mask)
            latent_action_loss = latent_outputs["latent_action_loss"]

        total_loss = action_dit_loss * float(self.latent_action_cfg.get("action_loss_weight", 1.0))
        if latent_action_loss is not None:
            total_loss = total_loss + latent_action_loss * float(self.latent_action_cfg.get("loss_weight", 1.0))

        out = {
            "total_loss": total_loss,
            "action_loss": total_loss,
            "action_dit_loss": action_dit_loss,
            "action_train_batch_size": action_train_batch_size,
        }
        out.update(latent_outputs)
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        meta_embs = self._encode_metaquery_hidden_states(batch_images, instructions)
        base_hidden = meta_embs[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(meta_embs, state_t)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


# ---------------------------------------------------------------------------
#  Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/Robotwin/train_files/starvla_metaquery_la_univla_robotwin.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUG_MODE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct"

    # Disable latent encoder for the smoke test (needs real checkpoint)
    cfg.framework.latent_action.enabled = False
    cfg.framework.latent_action.action_train_robot_types = None

    model = Qwen_MetaQuery_LA(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    action_dim = int(cfg.framework.action_model.action_dim)
    state_dim = int(cfg.framework.action_model.state_dim)
    horizon = int(cfg.framework.action_model.action_horizon)
    sample = {
        "action": np.random.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16),
    }
    sample2 = dict(sample)
    sample2["lang"] = "Another fake instruction for testing."
    batch = [sample, sample2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model(batch)
    loss = out["action_loss"]
    print(f"Action Loss: {loss.item()}")
    loss.backward()
    dit_grad = sum(p.grad.norm().item() for p in model.action_model.model.parameters() if p.grad is not None)
    emb_grad = model.qwen_vl_interface.model.get_input_embeddings().weight.grad
    emb_grad_norm = emb_grad.norm().item() if emb_grad is not None else 0.0
    print(f"[grad] action DiT grad-norm sum = {dit_grad:.4e} | input-embedding grad-norm = {emb_grad_norm:.4e}")
    assert dit_grad > 0, "Action DiT received zero gradient!"

    predict_output = model.predict_action([sample])
    print(f"Predicted normalized actions shape: {predict_output['normalized_actions'].shape}")
    print("Finished")
