"""QwenPI_v4 with bridge-token latent-action supervision."""

from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenPI_v4 import Qwen_PI_v4
from starVLA.model.modules.latent_action import build_latent_action_encoder
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenPI_v4_LA")
class Qwen_PI_v4_LA(Qwen_PI_v4):
    """QwenPI_v4 plus latent-action loss on appended bridge-token positions."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        load_latent_action_encoder = bool(kwargs.pop("load_latent_action_encoder", True))
        super().__init__(config=config, **kwargs)
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        self.latent_action_private_cfg = self.latent_action_cfg.qwenpi_v4_la
        self.latent_action_enabled = bool(self.latent_action_cfg.get("enabled", True))
        self.latent_action_backend = str(self.latent_action_cfg.get("backend", "unit")).lower()
        self.train_latent_action = bool(self.latent_action_cfg.get("train_latent", True))
        self.train_continuous_action = bool(self.latent_action_cfg.get("train_action", True))
        if not self.train_latent_action and not self.train_continuous_action:
            raise ValueError("At least one of latent_action.train_latent or latent_action.train_action must be true.")

        self.num_bridge_tokens = int(self.latent_action_cfg.num_bridge_tokens)
        self.num_codebooks = int(self.latent_action_cfg.num_codebooks)
        self.codebook_size = int(self.latent_action_cfg.codebook_size)
        self.latent_action_loss_type = self._get_latent_action_loss_type()
        self.label_smoothing = float(self.latent_action_cfg.get("label_smoothing", 0.0))
        self.detach_vl_embs_for_action_head = bool(
            self.latent_action_cfg.get("detach_vl_embs_for_action_head", False)
        )

        self.latent_action_token_ids: list[int] = []
        self.latent_action_encoder = None
        self.latent_head = None
        episode_cache_dir = self.latent_action_cfg.get("episode_cache_dir", None)
        if episode_cache_dir is None:
            try:
                episode_cache_dir = self.config.datasets.vla_data.latent_action.get(
                    "episode_cache_dir", None
                )
            except Exception:
                episode_cache_dir = None
        self.use_cached_soft_distribution = (
            episode_cache_dir is not None and str(episode_cache_dir) not in ("", "null", "None")
        )
        cached_indices_path = self.latent_action_cfg.get("cached_indices_path", None)
        self.use_cached_indices = (
            cached_indices_path is not None and str(cached_indices_path) not in ("", "null", "None")
        )
        skip_encoder = self.use_cached_soft_distribution or self.use_cached_indices
        if self.latent_action_enabled:
            self._expand_latent_action_bridge_tokens()
            if self.train_latent_action and load_latent_action_encoder and not skip_encoder:
                self.latent_action_encoder = build_latent_action_encoder(self.config)
                if self.latent_action_backend == "villax":
                    self._sync_villax_shape_from_encoder()
            elif self.use_cached_soft_distribution:
                logger.info("Using episode soft_kl cache from: %s", episode_cache_dir)
            elif self.use_cached_indices:
                logger.info("Using cached latent-action indices from: %s", cached_indices_path)
            hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)
            dropout = float(self.latent_action_private_cfg.get("head_dropout", 0.0))
            self.latent_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.num_codebooks * self.codebook_size),
            )

        self._printed_training_sample = False

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [key for key in state_dict if key.startswith("latent_action_encoder.")]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

    def _ensure_latent_action_defaults(self) -> None:
        from omegaconf import OmegaConf

        defaults = {
            "enabled": True,
            "backend": "unit",
            "loss_type": "auto",
            "train_latent": True,
            "train_action": True,
            "latent_loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "detach_vl_embs_for_action_head": False,
            "action_train_robot_types": None,
            "num_bridge_tokens": 8,
            "num_codebooks": 2,
            "codebook_size": 128,
            "label_smoothing": 0.0,
            "ce_loss_weights": None,
            "strict_num_bridge_tokens": True,
            "groot_tokenizer_path": None,
            "dinov2_path_override": None,
            "image_size": [224, 224],
            "cached_indices_path": None,
            "episode_cache_dir": None,
            "qwenpi_v4_la": {
                "token_format": "<latent_action_{i}>",
                "head_dropout": 0.0,
            },
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
                "config_path": (
                    "starVLA/model/modules/latent_action/softvq_joint/"
                    "0614_joint_aloha_xrecon_scratch.yaml"
                ),
                "ckpt_path": (
                    "/inspire/qb-ilm/project/qproject-fundationmodel/public/jjc/exp/softla/"
                    "0614_joint_aloha_xrecon_scratch/checkpoints/partial_step_100000.pt"
                ),
                "image_size": [224, 224],
                "strict_load": True,
                "kl_eps": 1e-8,
            },
            "sharla": {
                "config_path": None,
                "ckpt_path": None,
                "image_size": [224, 224],
                "strict_load": True,
                "kl_eps": 1e-8,
            },
            "villax": {
                "ckpt_path": None,
                "image_size": [224, 224],
                "num_learned_tokens": 4,
            },
        }

        current = self.config.framework.get("latent_action", {})
        backend = str(current.get("backend", defaults["backend"])).lower()
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
                "strict_num_bridge_tokens": True,
            }
        elif backend == "sharla":
            backend_defaults = {
                "loss_type": "soft_kl",
                "num_bridge_tokens": 8,
                "num_codebooks": 1,
                "codebook_size": 256,
                "strict_num_bridge_tokens": True,
            }
        elif backend in {"unit", "groot_unit"}:
            backend_defaults = {"loss_type": "ce"}
        elif backend == "villax":
            backend_defaults = {
                "loss_type": "ce",
                "num_bridge_tokens": 7,
                "num_codebooks": 1,
                "codebook_size": 32,
                "villax": {"num_learned_tokens": 4},
            }

        self.config.framework.latent_action = OmegaConf.merge(
            OmegaConf.create(defaults),
            OmegaConf.create(backend_defaults),
            current,
        )

    def _get_latent_action_loss_type(self) -> str:
        loss_type = str(self.latent_action_cfg.get("loss_type", "auto")).lower()
        if loss_type == "auto":
            loss_type = "soft_kl" if self.latent_action_backend in {"softvq", "sharla"} else "ce"
        if loss_type not in {"ce", "soft_kl"}:
            raise ValueError(f"latent_action.loss_type must be 'ce' or 'soft_kl', got {loss_type!r}.")
        if loss_type == "soft_kl" and self.latent_action_backend not in {"softvq", "sharla"}:
            raise ValueError(
                "Use latent_action.backend='softvq' or 'sharla' with latent_action.loss_type='soft_kl'."
            )
        return loss_type

    def _expand_latent_action_bridge_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_format = str(self.latent_action_private_cfg.token_format)
        tokens = [token_format.format(i=i) for i in range(self.num_bridge_tokens)]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

        token_ids = []
        for token in tokens:
            ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Latent-action bridge token `{token}` must map to one token id, got {ids}.")
            token_ids.append(int(ids[0]))
        self.latent_action_token_ids = token_ids
        self.register_buffer(
            "latent_action_bridge_token_ids",
            torch.tensor(token_ids, dtype=torch.long),
            persistent=False,
        )

    def _sync_villax_shape_from_encoder(self) -> None:
        villax_num_tokens = int(self.latent_action_encoder.num_learned_tokens)
        villax_codebook_size = int(self.latent_action_encoder.n_codes)
        if self.num_codebooks != villax_num_tokens:
            logger.warning(
                "Overriding latent_action.num_codebooks=%d with villa-x checkpoint num_learned_tokens=%d.",
                self.num_codebooks,
                villax_num_tokens,
            )
            self.num_codebooks = villax_num_tokens
            self.latent_action_cfg.num_codebooks = villax_num_tokens
        if self.codebook_size != villax_codebook_size:
            logger.warning(
                "Overriding latent_action.codebook_size=%d with villa-x checkpoint n_codes=%d.",
                self.codebook_size,
                villax_codebook_size,
            )
            self.codebook_size = villax_codebook_size
            self.latent_action_cfg.codebook_size = villax_codebook_size

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
        allowed = {str(robot_type) for robot_type in robot_types if str(robot_type)}
        return allowed if allowed else None

    def _select_action_training_batch(
        self,
        examples: List[dict],
        vl_embs_list: list[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> tuple[List[dict], list[torch.Tensor], torch.Tensor | None]:
        action_train_robot_types = self._get_action_train_robot_types()
        if action_train_robot_types is None:
            return examples, vl_embs_list, attention_mask

        selected_indices = [
            idx
            for idx, example in enumerate(examples)
            if str(example.get("robot_type", "")) in action_train_robot_types
        ]
        if len(selected_indices) == len(examples):
            return examples, vl_embs_list, attention_mask
        if not selected_indices:
            return [], [], None

        index_tensor = torch.tensor(selected_indices, device=vl_embs_list[-1].device, dtype=torch.long)
        selected_examples = [examples[idx] for idx in selected_indices]
        selected_vl_embs_list = [hidden.index_select(0, index_tensor) for hidden in vl_embs_list]
        selected_attention_mask = (
            attention_mask.index_select(0, index_tensor) if attention_mask is not None else None
        )
        return selected_examples, selected_vl_embs_list, selected_attention_mask

    def _collect_latent_window_counts(self, examples: List[dict]) -> list[int]:
        counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None:
                raise ValueError(
                    "QwenPI_v4_LA requires `la_frames`. Use "
                    "datasets.vla_data.dataset_py=lerobot_la_datasets or hdf5_la_dataset "
                    "and enable datasets.vla_data.latent_action."
                )
            if self.latent_action_backend == "villax":
                if not isinstance(frames, list) or len(frames) == 0 or not isinstance(frames[0], list):
                    raise ValueError(
                        "QwenPI_v4_LA (villax) requires `la_frames` to be a list of clips. "
                        "Set datasets.vla_data.latent_action.full_window=true."
                    )
                counts.append(len(frames))
            else:
                if len(frames) < 2:
                    raise ValueError(f"`la_frames` must contain at least 2 frames, got {len(frames)}.")
                counts.append(len(frames) - 1)
        if len(set(counts)) != 1:
            raise ValueError(
                "QwenPI_v4_LA requires the same latent-action window count per batch "
                f"because bridge-token suffixes are batched without padding, got {counts}."
            )
        return counts

    def _build_latent_frame_pairs(self, examples: List[dict]):
        frame_pairs = []
        counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    "QwenPI_v4_LA requires `la_frames` with at least two frames for latent-action targets."
                )
            counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
        if len(set(counts)) != 1:
            raise ValueError(f"Latent-action frame-pair counts must match within a batch, got {counts}.")
        return frame_pairs, counts

    def _build_latent_suffix(self, window_count: int, batch_size: int, device: torch.device) -> torch.LongTensor:
        bridge_ids = self.latent_action_bridge_token_ids.to(device)
        suffix = bridge_ids.repeat(int(window_count))
        return suffix.unsqueeze(0).expand(batch_size, -1)

    def _encode_vl_hidden_states_with_latent_tokens(
        self,
        batch_images: List,
        instructions: list[str],
        window_count: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor, dict]:
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        suffix = self._build_latent_suffix(window_count, batch_size, input_ids.device)
        inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        if attention_mask is not None:
            suffix_mask = torch.ones(
                suffix.shape,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            inputs["attention_mask"] = torch.cat([attention_mask, suffix_mask], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = [
                hidden[:, :prompt_len, :]
                for hidden in outputs.hidden_states[-self.num_action_dit_layers :]
            ]
            latent_hidden = outputs.hidden_states[-1][:, prompt_len:, :]

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        return vl_embs_list, attention_mask, latent_hidden, inputs

    def _make_discrete_targets(self, examples: List[dict], instructions: list[str]) -> torch.LongTensor:
        if self.use_cached_indices and "la_cached_indices" in examples[0]:
            return self._load_cached_targets(examples)

        if self.latent_action_encoder is None:
            raise RuntimeError("Latent-action encoder is not initialized.")

        if self.latent_action_backend == "villax":
            targets = self._make_villax_targets(examples)
            return targets.to(self.latent_action_bridge_token_ids.device)

        frame_pairs, counts = self._build_latent_frame_pairs(examples)
        pair_instructions = []
        for instruction, count in zip(instructions, counts):
            pair_instructions.extend([instruction] * count)

        indices = self.latent_action_encoder.encode(frame_pairs, pair_instructions)
        if indices.ndim == 1:
            indices = indices[:, None, None]
        elif indices.ndim == 2:
            indices = indices.unsqueeze(-1)
        elif indices.ndim != 3:
            raise ValueError(f"Latent-action encoder must return [pairs,tokens,codes], got {tuple(indices.shape)}.")

        self._validate_discrete_target_shape(indices, len(frame_pairs))
        pair_count = counts[0]
        return indices.reshape(len(examples), pair_count * self.num_bridge_tokens, self.num_codebooks).to(
            self.latent_action_bridge_token_ids.device
        )

    def _load_cached_targets(self, examples: List[dict]) -> torch.LongTensor:
        """Load pre-computed latent-action indices from cached data."""
        cached_list = []
        for example in examples:
            cached = example.get("la_cached_indices", None)
            if cached is None:
                raise RuntimeError(
                    "use_cached_indices is enabled but sample is missing 'la_cached_indices'. "
                    "Ensure datasets.vla_data.latent_action.cached_indices_path is set correctly."
                )
            if not isinstance(cached, torch.Tensor):
                cached = torch.tensor(cached, dtype=torch.long)
            cached_list.append(cached)
        indices = torch.stack(cached_list, dim=0)
        return indices.to(self.latent_action_bridge_token_ids.device)

    def _make_villax_targets(self, examples: List[dict]) -> torch.LongTensor:
        all_clips = []
        window_counts = []
        for example in examples:
            clips = example.get("la_frames", None)
            if clips is None or not isinstance(clips, list) or len(clips) == 0 or not isinstance(clips[0], list):
                raise ValueError(
                    "QwenPI_v4_LA (villax) requires `la_frames` as a list of clips. "
                    "Set datasets.vla_data.latent_action.full_window=true."
                )
            window_counts.append(len(clips))
            all_clips.extend(clips)
        if len(set(window_counts)) != 1:
            raise ValueError(f"VillaX window counts must match within a batch, got {window_counts}.")

        indices = self.latent_action_encoder.encode_clips(all_clips)
        if indices.shape[1] != self.num_bridge_tokens:
            raise ValueError(
                f"villa-x encoder returned {indices.shape[1]} transitions, "
                f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if indices.shape[2] != self.num_codebooks:
            raise ValueError(
                f"villa-x encoder returned {indices.shape[2]} codebooks, "
                f"but latent_action.num_codebooks={self.num_codebooks}."
            )
        self._validate_code_range(indices)
        return indices.reshape(len(examples), window_counts[0] * self.num_bridge_tokens, self.num_codebooks)

    def _validate_discrete_target_shape(self, indices: torch.Tensor, num_pairs: int) -> None:
        if indices.shape[0] != num_pairs:
            raise ValueError(
                f"Latent-action encoder returned {indices.shape[0]} frame pairs, but {num_pairs} were provided."
            )
        if indices.shape[1] != self.num_bridge_tokens:
            raise ValueError(
                f"Latent-action encoder returned {indices.shape[1]} bridge tokens, "
                f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if indices.shape[2] != self.num_codebooks:
            raise ValueError(
                f"Latent-action encoder returned {indices.shape[2]} codebooks, "
                f"but latent_action.num_codebooks={self.num_codebooks}."
            )
        self._validate_code_range(indices)

    def _validate_code_range(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        max_idx = int(indices.max().item())
        min_idx = int(indices.min().item())
        if min_idx < 0 or max_idx >= self.codebook_size:
            raise ValueError(
                f"Latent-action indices must be in [0, {self.codebook_size}), got min={min_idx}, max={max_idx}."
            )

    def _make_soft_targets(self, examples: List[dict]) -> torch.Tensor:
        if self.use_cached_soft_distribution and "la_cached_distribution" in examples[0]:
            return self._load_cached_soft_targets(examples)

        if self.latent_action_encoder is None:
            raise RuntimeError("Latent-action encoder is not initialized.")
        if not hasattr(self.latent_action_encoder, "encode_distribution"):
            raise RuntimeError(
                f"{self.latent_action_backend} latent-action encoder must implement encode_distribution()."
            )

        frame_pairs, counts = self._build_latent_frame_pairs(examples)
        distributions = self.latent_action_encoder.encode_distribution(frame_pairs)
        if distributions.ndim != 3:
            raise ValueError(
                f"{self.latent_action_backend} encoder must return [pairs,tokens,codebook], "
                f"got {tuple(distributions.shape)}."
            )
        if distributions.shape[0] != len(frame_pairs):
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {distributions.shape[0]} frame pairs, "
                f"but {len(frame_pairs)} were provided."
            )
        if bool(self.latent_action_cfg.get("strict_num_bridge_tokens", True)):
            if distributions.shape[1] != self.num_bridge_tokens:
                raise ValueError(
                    f"{self.latent_action_backend} encoder returned {distributions.shape[1]} bridge tokens, "
                    f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
                )
        elif distributions.shape[1] < self.num_bridge_tokens:
            raise ValueError(
                f"{self.latent_action_backend} encoder returned {distributions.shape[1]} bridge tokens, "
                f"fewer than latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if distributions.shape[-1] != self.codebook_size:
            raise ValueError(
                f"SoftVQ encoder returned codebook size {distributions.shape[-1]}, "
                f"but latent_action.codebook_size={self.codebook_size}."
            )
        distributions = distributions[:, : self.num_bridge_tokens]
        pair_count = counts[0]
        return distributions.reshape(len(examples), pair_count * self.num_bridge_tokens, self.codebook_size).to(
            self.latent_action_bridge_token_ids.device
        )

    def _load_cached_soft_targets(self, examples: List[dict]) -> torch.Tensor:
        """Load soft_kl teacher weights from episode cache samples."""
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
            cached_list.append(cached.float())
        stacked = torch.stack(cached_list, dim=0)
        expected_tokens = stacked.shape[1]
        if expected_tokens % max(self.num_bridge_tokens, 1) != 0:
            raise ValueError(
                f"Cached soft distribution token count {expected_tokens} is not divisible by "
                f"num_bridge_tokens={self.num_bridge_tokens}."
            )
        if stacked.shape[-1] != self.codebook_size:
            raise ValueError(
                f"Cached soft distribution codebook size {stacked.shape[-1]} != "
                f"latent_action.codebook_size={self.codebook_size}."
            )
        return stacked.to(self.latent_action_bridge_token_ids.device)

    def _collect_la_padding_mask(self, examples: List[dict]) -> torch.Tensor:
        flags = [bool(example.get("la_padded", False)) for example in examples]
        return torch.tensor([0.0 if padded else 1.0 for padded in flags], dtype=torch.float32)

    def _latent_logits(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        if self.latent_head is None:
            raise RuntimeError("Latent-action head is not initialized.")
        logits = self.latent_head(latent_hidden)
        return logits.reshape(latent_hidden.shape[0], latent_hidden.shape[1], self.num_codebooks, self.codebook_size)

    def _compute_latent_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        targets = targets.detach()
        if valid_mask is not None:
            valid_mask = valid_mask.detach()

        if self.latent_action_loss_type == "ce":
            if targets.ndim != 3:
                raise ValueError(f"CE targets must be [B,N,num_codebooks], got {tuple(targets.shape)}.")
            if logits.shape[:3] != targets.shape:
                raise ValueError(f"Latent logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}.")

            weights_cfg = self.latent_action_cfg.get("ce_loss_weights", None)
            if weights_cfg is None:
                weights = [1.0] * self.num_codebooks
            else:
                weights = [float(weight) for weight in weights_cfg]
                if len(weights) != self.num_codebooks:
                    raise ValueError(
                        f"latent_action.ce_loss_weights must have length {self.num_codebooks}, got {len(weights)}."
                    )

            per_codebook_losses = []
            for codebook_idx, weight in enumerate(weights):
                loss = F.cross_entropy(
                    logits[:, :, codebook_idx, :].reshape(-1, self.codebook_size).float(),
                    targets[:, :, codebook_idx].reshape(-1).long().to(logits.device),
                    label_smoothing=self.label_smoothing,
                )
                per_codebook_losses.append(loss * weight)
            ce_loss = torch.stack(per_codebook_losses).sum() / max(sum(weights), 1e-8)
            return {"latent_action_loss": ce_loss, "latent_ce_loss": ce_loss}

        if self.latent_action_loss_type == "soft_kl":
            if self.num_codebooks != 1:
                raise ValueError("soft_kl requires latent_action.num_codebooks=1.")
            logits = logits.squeeze(2)
            if logits.shape != targets.shape:
                raise ValueError(
                    f"Latent logits shape {tuple(logits.shape)} does not match target {tuple(targets.shape)}."
                )

            backend_cfg = self.latent_action_cfg.get(self.latent_action_backend, {})
            eps = float(backend_cfg.get("kl_eps", 1e-8))
            target_probs = targets.float().clamp_min(eps)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            kl_per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=1)

            if valid_mask is None:
                valid_mask = torch.ones_like(kl_per_sample)
            else:
                valid_mask = valid_mask.to(kl_per_sample.device, dtype=kl_per_sample.dtype)
            valid_count = int(valid_mask.sum().item())
            denom = valid_mask.sum().clamp_min(1.0)
            kl_loss = (kl_per_sample * valid_mask).sum() / denom
            kl_loss = self._scale_action_loss_for_global_mean(kl_loss, valid_count)

            entropy = -(target_probs * target_probs.log()).sum(dim=-1).mean(dim=1)
            entropy = (entropy * valid_mask).sum() / denom
            max_prob = target_probs.max(dim=-1).values.mean(dim=1)
            max_prob = (max_prob * valid_mask).sum() / denom
            return {
                "latent_action_loss": kl_loss,
                f"latent_{self.latent_action_backend}_kl_loss": kl_loss,
                f"latent_{self.latent_action_backend}_target_entropy": entropy,
                f"latent_{self.latent_action_backend}_target_max_prob": max_prob,
            }

        raise ValueError(f"Unknown latent_action_loss_type: {self.latent_action_loss_type}")

    def _action_loss_from_hidden(
        self,
        examples: List[dict],
        vl_embs_list: list[torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not examples or "action" not in examples[0]:
            raise ValueError("QwenPI_v4_LA action loss requires examples with `action`.")

        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        base_hidden = vl_embs_list[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            r = self.repeated_diffusion_steps
            actions_target = actions_target.repeat(r, 1, 1)
            if self.detach_vl_embs_for_action_head:
                vl_embs_list = [hidden.detach().repeat(r, 1, 1) for hidden in vl_embs_list]
            else:
                vl_embs_list = [hidden.repeat(r, 1, 1) for hidden in vl_embs_list]
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(r, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(r, 1, 1)

            return self.action_model(
                vl_embs_list,
                actions_target,
                state_repeated,
                encoder_attention_mask=attention_mask,
            )

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

    def _print_training_sample_once(self, examples: List[dict], instructions: list[str]) -> None:
        if self._printed_training_sample or not self.training or not logger.is_rank_zero():
            return
        example = examples[0]
        logger.info(
            "First QwenPI_v4_LA training sample (episode_index=%s, step_index=%s, episode_path=%s):\n"
            "  instruction: %s\n"
            "  la_frame_offsets: %s\n"
            "  num_bridge_tokens: %s\n"
            "  latent backend: %s",
            example.get("episode_index", None),
            example.get("step_index", None),
            example.get("episode_path", None),
            instructions[0],
            example.get("la_frame_offsets", None),
            self.num_bridge_tokens,
            self.latent_action_backend,
        )
        self._printed_training_sample = True

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        loss_mode = str(kwargs.pop("loss_mode", "joint")).lower()
        if loss_mode not in {"joint", "latent", "action"}:
            raise ValueError(f"loss_mode must be 'joint', 'latent', or 'action', got {loss_mode!r}.")

        compute_action = loss_mode in {"joint", "action"} and self.train_continuous_action
        compute_latent = loss_mode in {"joint", "latent"} and self.train_latent_action
        if loss_mode == "action" and not compute_action:
            raise RuntimeError("Action loss requested, but latent_action.train_action=false.")
        if loss_mode == "latent" and not compute_latent:
            raise RuntimeError("Latent loss requested, but latent_action.train_latent=false.")

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        latent_hidden = None
        if compute_latent:
            if not self.latent_action_enabled or self.latent_head is None:
                raise RuntimeError("Latent-action loss requested, but latent action is disabled or uninitialized.")
            window_count = self._collect_latent_window_counts(examples)[0]
            vl_embs_list, attention_mask, latent_hidden, _ = self._encode_vl_hidden_states_with_latent_tokens(
                batch_images,
                instructions,
                window_count,
            )
            self._print_training_sample_once(examples, instructions)
        else:
            vl_embs_list, attention_mask = self._encode_vl_hidden_states(batch_images, instructions)

        action_dit_loss = None
        action_train_batch_size = 0
        if compute_action:
            action_examples, action_vl_embs_list, action_attention_mask = self._select_action_training_batch(
                examples,
                vl_embs_list,
                attention_mask,
            )
            action_train_batch_size = len(action_examples)
            if action_examples:
                action_dit_loss = self._action_loss_from_hidden(
                    action_examples,
                    action_vl_embs_list,
                    action_attention_mask,
                )
            else:
                dummy_attention_mask = attention_mask[:1] if attention_mask is not None else None
                action_dit_loss = self._action_loss_from_hidden(
                    examples[:1],
                    [hidden[:1] for hidden in vl_embs_list],
                    dummy_attention_mask,
                ) * 0.0
            action_dit_loss = self._scale_action_loss_for_global_mean(action_dit_loss, action_train_batch_size)

        latent_outputs = {}
        latent_action_loss = None
        if compute_latent:
            if self.latent_action_loss_type == "soft_kl":
                targets = self._make_soft_targets(examples)
                valid_mask = self._collect_la_padding_mask(examples)
            else:
                targets = self._make_discrete_targets(examples, instructions)
                valid_mask = None
            logits = self._latent_logits(latent_hidden)
            latent_outputs = self._compute_latent_loss(logits, targets, valid_mask)
            latent_action_loss = latent_outputs["latent_action_loss"]

        total_loss = None
        if action_dit_loss is not None:
            total_loss = action_dit_loss * float(self.latent_action_cfg.action_loss_weight)
        if latent_action_loss is not None:
            latent_weight = float(self.latent_action_cfg.get("latent_loss_weight", self.latent_action_cfg.get("loss_weight", 1.0)))
            weighted_latent = latent_action_loss * latent_weight
            total_loss = weighted_latent if total_loss is None else total_loss + weighted_latent
        if total_loss is None:
            raise RuntimeError("No loss was computed. Check loss_mode and latent_action train flags.")

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
        out.update(latent_outputs)
        return out
