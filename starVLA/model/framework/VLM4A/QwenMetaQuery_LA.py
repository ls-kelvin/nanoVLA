"""QwenMetaQuery with pluggable latent-action auxiliary losses."""

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


def _make_bridge_tokens(num_tokens: int) -> list[str]:
    if num_tokens <= 0:
        raise ValueError(f"latent_action.num_bridge_tokens must be positive, got {num_tokens}.")
    return [f"<|bridge_{i}|>" for i in range(num_tokens)]


@dataclass
class QwenMetaQueryLADefaults:
    latent_action: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "backend": "unit",
            "loss_type": "auto",
            "mode": "ar",
            "groot_tokenizer_path": None,
            "dinov2_path_override": None,
            "num_bridge_tokens": 8,
            "num_codebooks": 2,
            "codebook_size": 128,
            "ce_loss_weights": [1.0, 0.5],
            "label_smoothing": 0.0,
            "loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "action_train_robot_types": None,
            "image_size": [224, 224],
            "strict_num_bridge_tokens": True,
            "token_format": "<robot_action_{i}>",
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
        }
    )


@FRAMEWORK_REGISTRY.register("QwenMetaQuery_LA")
class Qwen_MetaQuery_LA(Qwen_MetaQuery):
    """QwenMetaQuery plus a cache-split latent-action objective.

    Training runs one multimodal prefix prefill, then continues from the same
    KV cache with two independent assistant suffixes:
      - latent-action suffix for bridge CE (UniT) or AR LM loss (UniVLA).
      - Metaquery tokens for the original action DiT loss.

    Inference only runs the metaquery suffix.
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
        self.ce_loss_weights = list(self.latent_action_cfg.get("ce_loss_weights", [1.0] * self.num_codebooks))
        self.label_smoothing = float(self.latent_action_cfg.get("label_smoothing", 0.0))
        if self.latent_action_loss_type == "bridge_ce" and len(self.ce_loss_weights) != self.num_codebooks:
            raise ValueError(
                f"latent_action.ce_loss_weights length ({len(self.ce_loss_weights)}) "
                f"must match num_codebooks ({self.num_codebooks})."
            )

        self.latent_action_token_ids: list[int] = []
        self.latent_action_id_to_token: dict[int, str] = {}
        self.bridge_ce_predictors = nn.ModuleList()
        hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)

        if self.latent_action_enabled and self.latent_action_loss_type == "bridge_ce":
            self._expand_bridge_tokens()
            self.bridge_ce_predictors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                        nn.Linear(hidden_size, self.codebook_size),
                    )
                    for _ in range(self.num_codebooks)
                ]
            )
        elif self.latent_action_enabled and self.latent_action_loss_type == "ar_lm":
            self._expand_latent_action_tokens()

        if self.latent_action_enabled:
            self.latent_action_encoder = build_latent_action_encoder(self.config)

        self._printed_training_sample = False

    def _ensure_latent_action_defaults(self) -> None:
        from omegaconf import OmegaConf

        current = self.config.framework.get("latent_action", {})
        backend = str(current.get("backend", QwenMetaQueryLADefaults().latent_action["backend"])).lower()
        backend_defaults = {}
        if backend == "univla":
            backend_defaults = {
                "loss_type": "ar_lm",
                "num_bridge_tokens": 4,
                "num_codebooks": 1,
                "codebook_size": 16,
                "ce_loss_weights": [1.0],
            }
        elif backend in {"unit", "groot_unit"}:
            backend_defaults = {"loss_type": "bridge_ce"}

        defaults = OmegaConf.create(QwenMetaQueryLADefaults().latent_action)
        self.config.framework.latent_action = OmegaConf.merge(defaults, OmegaConf.create(backend_defaults), current)

    def _get_latent_action_loss_type(self) -> str:
        loss_type = str(self.latent_action_cfg.get("loss_type", "auto")).lower()
        if loss_type == "auto":
            loss_type = "ar_lm" if self.latent_action_backend == "univla" else "bridge_ce"
        if loss_type not in {"bridge_ce", "ar_lm"}:
            raise ValueError(f"latent_action.loss_type must be 'auto', 'bridge_ce', or 'ar_lm', got {loss_type!r}.")
        if loss_type == "ar_lm":
            mode = str(self.latent_action_cfg.get("mode", "ar")).lower()
            if mode != "ar":
                raise ValueError("QwenMetaQuery_LA ar_lm supports only latent_action.mode='ar'.")
            if self.num_codebooks != 1:
                raise ValueError("QwenMetaQuery_LA ar_lm requires latent_action.num_codebooks=1.")
        if loss_type == "bridge_ce" and self.latent_action_backend == "univla":
            raise ValueError("Use latent_action.loss_type='ar_lm' for backend='univla'.")
        return loss_type

    def _expand_bridge_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        tokens = _make_bridge_tokens(self.num_bridge_tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

        token_ids = []
        for token in tokens:
            ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Bridge token `{token}` must map to one token id, got {ids}.")
            token_ids.append(int(ids[0]))
        self.register_buffer(
            "bridge_suffix_ids",
            torch.tensor(token_ids, dtype=torch.long),
            persistent=False,
        )

    def _expand_latent_action_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_format = str(self.latent_action_cfg.token_format)
        tokens = [token_format.format(i=i) for i in range(self.codebook_size)]

        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

        token_ids = []
        for token in tokens:
            ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Latent-action token `{token}` must map to exactly one token id, got {ids}.")
            token_ids.append(int(ids[0]))

        sorted_ids = sorted(token_ids)
        if sorted_ids != list(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids))):
            raise ValueError(f"Latent-action token ids must be contiguous for label masking, got {sorted_ids}.")

        self.latent_action_token_ids = token_ids
        self.latent_action_id_to_token = {idx: tok for idx, tok in zip(token_ids, tokens)}
        self.qwen_vl_interface._ACTION_TOKEN_MIN = sorted_ids[0]
        self.qwen_vl_interface._ACTION_TOKEN_MAX = sorted_ids[-1]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [key for key in state_dict if key.startswith("latent_action_encoder.")]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

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
                f"because bridge suffixes are batched without padding, got pair counts {pair_counts}."
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

        if bool(self.latent_action_cfg.get("strict_num_bridge_tokens", True)):
            if vq_indices.shape[1] != self.num_bridge_tokens:
                raise ValueError(
                    f"Latent-action encoder returned {vq_indices.shape[1]} VQ tokens, "
                    f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
                )
        elif vq_indices.shape[1] < self.num_bridge_tokens:
            raise ValueError(
                f"Latent-action encoder returned {vq_indices.shape[1]} VQ tokens, "
                f"fewer than latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if vq_indices.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"Latent-action encoder returned {vq_indices.shape[-1]} codebooks, "
                f"but latent_action.num_codebooks={self.num_codebooks}."
            )
        if vq_indices.numel() > 0:
            max_index = int(vq_indices.max().item())
            min_index = int(vq_indices.min().item())
            if min_index < 0 or max_index >= self.codebook_size:
                raise ValueError(
                    f"Latent-action indices must be in [0, {self.codebook_size}), "
                    f"got min={min_index}, max={max_index}."
                )
        vq_indices = vq_indices[:, : self.num_bridge_tokens]
        vq_indices = vq_indices.reshape(
            len(examples),
            pair_count * self.num_bridge_tokens,
            self.num_codebooks,
        )
        return vq_indices.to(self.metaquery_suffix_ids.device)

    def _latent_indices_to_solutions(self, indices: torch.LongTensor) -> list[str]:
        if indices.ndim != 3 or indices.shape[-1] != 1:
            raise ValueError(f"ar_lm expects latent indices shaped [B, L, 1], got {tuple(indices.shape)}.")
        token_format = str(self.latent_action_cfg.token_format)
        solutions = []
        for sample_indices in indices[..., 0].detach().cpu().tolist():
            solutions.append("".join(token_format.format(i=int(idx)) for idx in sample_indices))
        return solutions

    def _latent_indices_to_token_ids(self, indices: torch.LongTensor) -> torch.LongTensor:
        if indices.ndim != 3 or indices.shape[-1] != 1:
            raise ValueError(f"ar_lm expects latent indices shaped [B, L, 1], got {tuple(indices.shape)}.")
        if not self.latent_action_token_ids:
            raise RuntimeError("Latent-action AR tokens are not initialized.")
        flat_indices = indices[..., 0].long()
        token_ids = torch.tensor(self.latent_action_token_ids, dtype=torch.long, device=flat_indices.device)
        return token_ids.index_select(0, flat_indices.reshape(-1)).reshape(flat_indices.shape)

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

    def _build_prefix_inputs(self, batch_images: List, instructions: List[str]) -> dict:
        return self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

    def _build_training_messages(self, example: dict, instruction: str) -> list[dict]:
        content = [{"type": "image", "image": img} for img in example["image"]]

        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
            prompt = cot_prompt.replace("{instruction}", instruction)
        else:
            prompt = instruction

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _render_training_prefix(self, example: dict, instruction: str) -> str:
        messages = self._build_training_messages(example, instruction)
        return self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_token_suffix(self, token_ids: torch.LongTensor) -> str:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        return "".join(tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in token_ids.tolist())

    def _format_bridge_gt_indices(
        self,
        bridge_gt_indices: torch.LongTensor | None,
        bridge_pair_count: int | None,
    ) -> list | None:
        if bridge_gt_indices is None:
            return None
        sample_indices = bridge_gt_indices[0].detach().cpu()
        if bridge_pair_count is None:
            bridge_pair_count = sample_indices.shape[0] // self.num_bridge_tokens
        return sample_indices.reshape(bridge_pair_count, self.num_bridge_tokens, self.num_codebooks).tolist()

    def _print_training_sample_once(
        self,
        examples: List[dict],
        instructions: list[str],
        bridge_pair_count: int | None = None,
        bridge_gt_indices: torch.LongTensor | None = None,
        latent_ar_solution: str | None = None,
        latent_ar_token_ids: torch.LongTensor | None = None,
    ) -> None:
        if self._printed_training_sample or not self.training:
            return
        if not logger.is_rank_zero():
            return

        example = examples[0]
        instruction = instructions[0]
        sample_text = self._render_training_prefix(example, instruction)
        metaquery_suffix = self._format_token_suffix(self.metaquery_suffix_ids)

        bridge_suffix = None
        if bridge_pair_count is not None and hasattr(self, "bridge_suffix_ids"):
            bridge_suffix = self._format_token_suffix(self.bridge_suffix_ids.repeat(bridge_pair_count))
        formatted_bridge_gt_indices = self._format_bridge_gt_indices(bridge_gt_indices, bridge_pair_count)
        formatted_ar_token_ids = latent_ar_token_ids[0].detach().cpu().tolist() if latent_ar_token_ids is not None else None

        episode_index = example.get("episode_index", None)
        step_index = example.get("step_index", None)
        episode_path = example.get("episode_path", None)
        la_frame_offsets = example.get("la_frame_offsets", None)

        if self.latent_action_loss_type == "ar_lm":
            logger.info(
                "First training sample input (episode_index=%s, step_index=%s, episode_path=%s):\n"
                "%s\n"
                "Metaquery suffix branch:\n%s\n"
                "Latent AR suffix branch (pair_count=%s, la_frame_offsets=%s):\n%s\n"
                "Latent AR token ids:\n%s\n"
                "Latent GT indices [pair][bridge_token][codebook]:\n%s",
                episode_index,
                step_index,
                episode_path,
                sample_text,
                metaquery_suffix,
                bridge_pair_count,
                la_frame_offsets,
                latent_ar_solution,
                formatted_ar_token_ids,
                formatted_bridge_gt_indices,
            )
        else:
            logger.info(
                "First training sample input (episode_index=%s, step_index=%s, episode_path=%s):\n"
                "%s\n"
                "Metaquery suffix branch:\n%s\n"
                "Bridge suffix branch (pair_count=%s, la_frame_offsets=%s):\n%s\n"
                "Bridge GT indices [pair][bridge_token][codebook]:\n%s",
                episode_index,
                step_index,
                episode_path,
                sample_text,
                metaquery_suffix,
                bridge_pair_count,
                la_frame_offsets,
                bridge_suffix,
                formatted_bridge_gt_indices,
            )
        self._printed_training_sample = True

    def _run_prefix_cache(self, batch_images: List, instructions: List[str], return_outputs: bool = False):
        prefix_inputs = self._build_prefix_inputs(batch_images, instructions)
        prefix_outputs = self.qwen_vl_interface(
            **prefix_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past_key_values = getattr(prefix_outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "QwenMetaQuery_LA requires use_cache=True to return past_key_values. "
                "Disable incompatible gradient-checkpointing/cache settings for this framework."
            )
        if return_outputs:
            return prefix_inputs, past_key_values, prefix_outputs
        return prefix_inputs, past_key_values

    @staticmethod
    def _branch_cache(past_key_values):
        """Per-branch cache that shares the prefix KV tensors but appends into its own copy.

        ``DynamicLayer.update`` reassigns ``self.keys = torch.cat([self.keys, new])`` (it does
        not mutate in place), so shallow-copied layers append into fresh slots while the prefix
        tensors stay shared with the original cache — gradients still flow back through them, and
        the two suffix branches never contaminate each other. ``copy.deepcopy`` is wrong here: it
        would detach the prefix tensors from the autograd graph and silence the VLM gradient.
        """
        branch = copy.copy(past_key_values)
        branch.layers = [copy.copy(layer) for layer in past_key_values.layers]
        return branch

    def _run_suffix_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        suffix_ids: torch.LongTensor,
    ):
        input_ids = prefix_inputs["input_ids"]
        batch_size = input_ids.shape[0]
        suffix = suffix_ids.to(input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self._run_suffix_batch_from_cache(prefix_inputs, past_key_values, suffix)

    def _run_suffix_batch_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        suffix: torch.LongTensor,
        output_hidden_states: bool = True,
    ):
        input_ids = prefix_inputs["input_ids"]
        attention_mask = prefix_inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        if suffix.ndim != 2 or suffix.shape[0] != batch_size:
            raise ValueError(f"Suffix ids must be shaped [batch, suffix_len], got {tuple(suffix.shape)}.")
        prefix_len = past_key_values.get_seq_length()
        suffix = suffix.to(input_ids.device)
        suffix_mask = torch.ones(
            suffix.shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        branch_attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        # Continue mRoPE from the prefill positions: cache_position lets Qwen3-VL compute
        # position_ids = arange(L) + (prefix_len + rope_deltas) instead of restarting at 0,
        # and keeps the causal mask sized to (prefix_len + suffix_len) = the branch cache length.
        cache_position = torch.arange(
            prefix_len, prefix_len + suffix.shape[1], device=suffix.device
        )
        return self.qwen_vl_interface(
            input_ids=suffix,
            attention_mask=branch_attention_mask,
            past_key_values=self._branch_cache(past_key_values),
            cache_position=cache_position,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _encode_metaquery_from_cache(self, prefix_inputs: dict, past_key_values) -> List[torch.Tensor]:
        outputs = self._run_suffix_from_cache(prefix_inputs, past_key_values, self.metaquery_suffix_ids)
        per_layer = [layer[:, 1:-1, :] for layer in outputs.hidden_states]
        return per_layer[-self.num_action_dit_layers :]

    def _bridge_loss_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        gt_indices: torch.LongTensor,
    ) -> dict:
        if gt_indices.shape[1] % self.num_bridge_tokens != 0:
            raise ValueError(
                f"Latent target length {gt_indices.shape[1]} is not divisible by "
                f"num_bridge_tokens={self.num_bridge_tokens}."
            )
        pair_count = gt_indices.shape[1] // self.num_bridge_tokens
        bridge_suffix_ids = self.bridge_suffix_ids.repeat(pair_count)
        outputs = self._run_suffix_from_cache(prefix_inputs, past_key_values, bridge_suffix_ids)
        bridge_features = outputs.hidden_states[-1]
        logits_list = [predictor(bridge_features) for predictor in self.bridge_ce_predictors]

        total_loss = bridge_features.new_tensor(0.0, dtype=torch.float32)
        out = {}
        gt_indices = gt_indices.to(bridge_features.device)
        for i, logits in enumerate(logits_list):
            targets = gt_indices[..., i]
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                label_smoothing=self.label_smoothing,
            )
            out[f"latent_ce_loss_codebook{i + 1}"] = ce_loss
            total_loss = total_loss + ce_loss * float(self.ce_loss_weights[i])
        out["latent_action_loss"] = total_loss
        return out

    def _ar_lm_loss_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        prefix_outputs,
        target_token_ids: torch.LongTensor,
    ) -> dict:
        target_token_ids = target_token_ids.to(prefix_inputs["input_ids"].device)
        suffix_outputs = self._run_suffix_batch_from_cache(
            prefix_inputs,
            past_key_values,
            target_token_ids,
            output_hidden_states=False,
        )
        prefix_logits = getattr(prefix_outputs, "logits", None)
        suffix_logits = getattr(suffix_outputs, "logits", None)
        if prefix_logits is None or suffix_logits is None:
            raise RuntimeError("QwenMetaQuery_LA ar_lm requires logits from prefix and suffix forwards.")

        shifted_logits = torch.cat([prefix_logits[:, -1:, :], suffix_logits[:, :-1, :]], dim=1)
        if shifted_logits.shape[:2] != target_token_ids.shape:
            raise ValueError(
                f"Shifted AR logits shape {tuple(shifted_logits.shape[:2])} "
                f"does not match target ids shape {tuple(target_token_ids.shape)}."
            )
        latent_action_loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
            target_token_ids.reshape(-1),
            label_smoothing=self.label_smoothing,
        )
        return {
            "latent_action_loss": latent_action_loss,
            "latent_ar_lm_loss": latent_action_loss,
        }

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
            meta_embs = [h.repeat(r, 1, 1) for h in meta_embs]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(r, 1, 1)

            return self.action_model(meta_embs, actions_target, state_repeated)

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

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        if self.latent_action_enabled and self.latent_action_loss_type == "ar_lm":
            prefix_inputs, past_key_values, prefix_outputs = self._run_prefix_cache(
                batch_images,
                instructions,
                return_outputs=True,
            )
        else:
            prefix_inputs, past_key_values = self._run_prefix_cache(batch_images, instructions)
            prefix_outputs = None
        meta_embs = self._encode_metaquery_from_cache(prefix_inputs, past_key_values)
        action_examples, action_meta_embs = self._select_action_training_batch(examples, meta_embs)
        action_train_batch_size = len(action_examples)
        if action_examples:
            action_dit_loss = self._action_loss_from_meta_embs(action_examples, action_meta_embs)
        else:
            action_dit_loss = self._action_loss_from_meta_embs(
                examples[:1],
                [hidden[:1] for hidden in meta_embs],
            ) * 0.0
        action_dit_loss = self._scale_action_loss_for_global_mean(
            action_dit_loss,
            action_train_batch_size,
        )

        latent_outputs = {}
        latent_action_loss = None
        if self.latent_action_enabled:
            gt_indices = self._make_visual_only_vq_targets(examples)
            bridge_pair_count = gt_indices.shape[1] // self.num_bridge_tokens
            if self.latent_action_loss_type == "bridge_ce":
                self._print_training_sample_once(examples, instructions, bridge_pair_count, gt_indices)
                latent_outputs = self._bridge_loss_from_cache(prefix_inputs, past_key_values, gt_indices)
            else:
                latent_solutions = self._latent_indices_to_solutions(gt_indices)
                target_token_ids = self._latent_indices_to_token_ids(gt_indices)
                self._print_training_sample_once(
                    examples,
                    instructions,
                    bridge_pair_count,
                    gt_indices,
                    latent_solutions[0] if latent_solutions else None,
                    target_token_ids,
                )
                latent_outputs = self._ar_lm_loss_from_cache(
                    prefix_inputs,
                    past_key_values,
                    prefix_outputs,
                    target_token_ids,
                )
            latent_action_loss = latent_outputs["latent_action_loss"]
        else:
            self._print_training_sample_once(examples, instructions)

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

        prefix_inputs, past_key_values = self._run_prefix_cache(batch_images, instructions)
        meta_embs = self._encode_metaquery_from_cache(prefix_inputs, past_key_values)
        base_hidden = meta_embs[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(meta_embs, state_t)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
