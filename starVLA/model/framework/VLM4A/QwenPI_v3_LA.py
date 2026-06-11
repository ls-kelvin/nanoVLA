"""QwenPI_v3 with pluggable latent-action generation."""

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.latent_action import build_latent_action_encoder
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenPI_v3_LA")
class Qwen_PI_v3_LA(Qwen_PI_v3):
    """General latent-action extension of QwenPI_v3.

    UniVLA is the first backend behind the common latent-action encoder
    interface.  The VLM tokenizer is expanded after ``base_vlm`` is loaded.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        self.latent_action_enabled = bool(self.latent_action_cfg.get("enabled", True))
        self.latent_action_backend = str(self.latent_action_cfg.get("backend", "univla")).lower()
        self.latent_action_mode = self._get_latent_action_mode()
        self.train_latent_action = bool(self.latent_action_cfg.get("train_latent", True))
        self.train_continuous_action = bool(self.latent_action_cfg.get("train_action", True))
        if not self.train_latent_action and not self.train_continuous_action:
            raise ValueError("At least one of latent_action.train_latent or latent_action.train_action must be true.")

        self.latent_action_token_ids: list[int] = []
        self.latent_action_id_to_token: dict[int, str] = {}
        self.latent_action_query_token_ids: list[int] = []
        self.softvq_input_proj = None
        self.softvq_output_proj = None

        if self.latent_action_backend == "softvq":
            # SoftVQ is a continuous soft-distribution objective; it does not use
            # the discrete latent-action token vocabulary or the la_mask attention
            # path. Instead it projects the target distribution into / out of the
            # LLM hidden space and aligns via KL (see _softvq_kl_loss_from_cache).
            self.num_bridge_tokens = int(self.latent_action_cfg.get("num_bridge_tokens", 1))
            self.num_codebooks = int(self.latent_action_cfg.get("num_codebooks", 1))
            self.codebook_size = int(self.latent_action_cfg.codebook_size)
            if self.num_codebooks != 1:
                raise ValueError("QwenPI_v3_LA softvq requires latent_action.num_codebooks=1.")
            hidden_size = int(self.config.framework.qwenvl.vl_hidden_dim)
            self.softvq_input_proj = nn.Linear(self.codebook_size, hidden_size)
            self.softvq_output_proj = nn.Linear(hidden_size, self.codebook_size)
        elif self.latent_action_enabled:
            self._expand_latent_action_tokens()

        self.latent_action_encoder = None
        if self.latent_action_enabled and self.train_latent_action:
            self.latent_action_encoder = build_latent_action_encoder(self.config)

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
            "backend": "univla",
            "train_latent": True,
            "train_action": True,
            "latent_loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "la_mask": True,
            "action_train_robot_types": None,
            "mode": "ar",
            "codebook_size": 16,
            "num_codes_per_pair": 4,
            "token_format": "<robot_action_{i}>",
            "query_token_format": "<robot_action_query_{i}>",
            "latent_generation_max_extra_tokens": None,
            # SoftVQ latent-action loss (vision-only KL). Mirrors QwenMetaQuery_LA:
            # the soft target distribution is projected into the LLM as suffix
            # embeddings and the suffix hidden states predict an aligned
            # categorical distribution via KL. These keys are only consumed when
            # backend == "softvq".
            "num_bridge_tokens": 1,
            "num_codebooks": 1,
            "strict_num_bridge_tokens": True,
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
        }
        current = self.config.framework.get("latent_action", {})
        backend = str(current.get("backend", defaults["backend"])).lower()
        backend_defaults = {}
        if backend == "softvq":
            backend_defaults = {
                "num_bridge_tokens": 1,
                "num_codebooks": 1,
                "codebook_size": 64,
            }
        self.config.framework.latent_action = OmegaConf.merge(
            OmegaConf.create(defaults),
            OmegaConf.create(backend_defaults),
            current,
        )

    def _get_latent_action_mode(self) -> str:
        mode = str(self.latent_action_cfg.get("mode", "ar")).lower()
        if mode not in {"ar", "query"}:
            raise ValueError(f"latent_action.mode must be 'ar' or 'query', got {mode!r}.")
        return mode

    def _expand_latent_action_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        codebook_size = int(self.latent_action_cfg.codebook_size)
        num_codes_per_pair = int(self.latent_action_cfg.num_codes_per_pair)
        token_format = str(self.latent_action_cfg.token_format)
        query_token_format = str(self.latent_action_cfg.query_token_format)
        tokens = [token_format.format(i=i) for i in range(codebook_size)]
        query_tokens = []
        if self.latent_action_mode == "query":
            query_tokens = [query_token_format.format(i=i) for i in range(num_codes_per_pair)]
            duplicates = set(tokens) & set(query_tokens)
            if duplicates:
                raise ValueError(f"Latent-action query tokens overlap with action tokens: {sorted(duplicates)}")

        tokenizer.add_special_tokens({"additional_special_tokens": tokens + query_tokens})
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

        query_token_ids = []
        for token in query_tokens:
            ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Latent-action query token `{token}` must map to exactly one token id, got {ids}.")
            query_token_ids.append(int(ids[0]))
        self.latent_action_query_token_ids = query_token_ids

    def _token_id_mask(self, input_ids: torch.LongTensor, token_ids: list[int]) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in token_ids:
            mask |= input_ids == int(token_id)
        return mask

    def _mask_latent_action_attention(
        self,
        attention_mask: torch.Tensor | None,
        input_ids: torch.LongTensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is None or input_ids is None:
            return attention_mask
        if not bool(self.latent_action_cfg.get("la_mask", True)):
            return attention_mask
        if not self.latent_action_token_ids:
            return attention_mask

        if self.latent_action_mode == "query":
            latent_token_mask = self._token_id_mask(input_ids, self.latent_action_query_token_ids)
        else:
            action_token_min = min(self.latent_action_token_ids)
            action_token_max = max(self.latent_action_token_ids)
            latent_token_mask = (input_ids >= action_token_min) & (input_ids <= action_token_max)
            
        if not latent_token_mask.any():
            return attention_mask
        
        latent_token_mask = latent_token_mask.cumsum(dim=1) > 0

        masked_attention = attention_mask.clone()
        return masked_attention.masked_fill(latent_token_mask, 0)

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
        backbone_attention_mask: torch.Tensor | None,
    ) -> tuple[List[dict], list[torch.Tensor], torch.Tensor | None]:
        action_train_robot_types = self._get_action_train_robot_types()
        if action_train_robot_types is None:
            return examples, vl_embs_list, backbone_attention_mask

        selected_indices = [
            idx
            for idx, example in enumerate(examples)
            if str(example.get("robot_type", "")) in action_train_robot_types
        ]
        if len(selected_indices) == len(examples):
            return examples, vl_embs_list, backbone_attention_mask
        if not selected_indices:
            return [], [], None

        index_tensor = torch.tensor(selected_indices, device=vl_embs_list[-1].device, dtype=torch.long)
        selected_examples = [examples[idx] for idx in selected_indices]
        selected_vl_embs_list = [hidden.index_select(0, index_tensor) for hidden in vl_embs_list]
        selected_attention_mask = (
            backbone_attention_mask.index_select(0, index_tensor)
            if backbone_attention_mask is not None
            else None
        )
        return selected_examples, selected_vl_embs_list, selected_attention_mask

    def _build_latent_frame_pairs(self, examples: List[dict]):
        flat_pairs = []
        counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None:
                raise ValueError(
                    "Missing `la_frames`. Use datasets.vla_data.dataset_py=lerobot_la_datasets "
                    "and datasets.vla_data.latent_action.enabled=true when training latent actions."
                )
            if len(frames) < 2:
                raise ValueError(f"`la_frames` must contain at least 2 frames, got {len(frames)}.")
            counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                flat_pairs.append((frames[i], frames[i + 1]))
        return flat_pairs, counts

    def _latent_indices_to_solutions(self, indices: torch.LongTensor, counts: list[int]) -> list[str]:
        if indices.ndim == 1:
            indices = indices[:, None]
        token_format = str(self.latent_action_cfg.token_format)
        solutions = []
        cursor = 0
        for count in counts:
            chunk = indices[cursor : cursor + count].reshape(-1).detach().cpu().tolist()
            cursor += count
            solutions.append("".join(token_format.format(i=int(idx)) for idx in chunk))
        return solutions

    def _make_latent_action_targets(
        self,
        examples: List[dict],
        instructions: list[str],
    ) -> tuple[torch.LongTensor, list[int], list[str]]:
        if self.latent_action_encoder is None:
            raise ValueError("Latent-action encoder is not initialized.")
        frame_pairs, counts = self._build_latent_frame_pairs(examples)
        pair_instructions = []
        for instruction, count in zip(instructions, counts):
            pair_instructions.extend([instruction] * count)
        indices = self.latent_action_encoder.encode(frame_pairs, pair_instructions)
        return indices, counts, self._latent_indices_to_solutions(indices, counts)

    def _make_latent_action_solutions(self, examples: List[dict], instructions: list[str]) -> list[str]:
        _, _, solutions = self._make_latent_action_targets(examples, instructions)
        return solutions

    def _make_latent_action_query_solutions(self, counts: list[int]) -> list[str]:
        query_token_format = str(self.latent_action_cfg.query_token_format)
        num_codes_per_pair = int(self.latent_action_cfg.num_codes_per_pair)
        query_block = "".join(query_token_format.format(i=i) for i in range(num_codes_per_pair))
        return [query_block * count for count in counts]

    def _infer_query_pair_count_from_config(self) -> int:
        data_la_cfg = self.config.datasets.vla_data.get("latent_action", {})
        stride = int(data_la_cfg.get("stride", 4))
        if stride <= 0:
            raise ValueError(f"datasets.vla_data.latent_action.stride must be positive, got {stride}.")

        offsets = list(range(0, self.action_horizon, stride))
        if not offsets:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}.")
        if bool(data_la_cfg.get("include_terminal_frame", True)) and offsets[-1] != self.action_horizon - 1:
            offsets.append(self.action_horizon - 1)
        return len(offsets) - 1

    def _get_query_pair_counts(self, examples: List[dict]) -> list[int]:
        inferred_count = self._infer_query_pair_count_from_config()
        return [inferred_count] * len(examples)

    def _latent_indices_to_token_ids(self, indices: torch.LongTensor, counts: list[int]) -> list[list[int]]:
        if indices.ndim == 1:
            indices = indices[:, None]

        token_ids_by_sample = []
        cursor = 0
        for count in counts:
            chunk = indices[cursor : cursor + count].reshape(-1).detach().cpu().tolist()
            cursor += count
            token_ids = []
            for idx in chunk:
                idx = int(idx)
                if idx < 0 or idx >= len(self.latent_action_token_ids):
                    raise ValueError(
                        f"Latent-action code index {idx} is outside codebook size {len(self.latent_action_token_ids)}."
                    )
                token_ids.append(self.latent_action_token_ids[idx])
            token_ids_by_sample.append(token_ids)
        return token_ids_by_sample

    def _build_query_labels(
        self,
        input_ids: torch.LongTensor,
        target_token_ids: list[list[int]],
    ) -> torch.LongTensor:
        labels = torch.full_like(input_ids, -100)
        query_token_mask = self._token_id_mask(input_ids, self.latent_action_query_token_ids)

        for batch_idx, targets in enumerate(target_token_ids):
            query_positions = torch.nonzero(query_token_mask[batch_idx], as_tuple=False).flatten()
            if query_positions.numel() != len(targets):
                raise ValueError(
                    f"Query token count mismatch for sample {batch_idx}: "
                    f"got {query_positions.numel()}, expected {len(targets)}."
                )
            if query_positions.numel() == 0:
                continue

            label_positions = query_positions + 1
            if int(label_positions[-1].item()) >= labels.size(1):
                raise ValueError("Query tokens must be followed by at least one token for shifted causal-LM loss.")
            labels[batch_idx, label_positions] = torch.tensor(
                targets,
                device=labels.device,
                dtype=labels.dtype,
            )

        return labels

    def _extract_query_latent_action_solutions(
        self,
        logits: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> list[str]:
        query_token_mask = self._token_id_mask(input_ids, self.latent_action_query_token_ids)
        action_token_ids = torch.tensor(self.latent_action_token_ids, device=logits.device, dtype=torch.long)
        solutions = []

        for batch_idx in range(input_ids.size(0)):
            query_positions = torch.nonzero(query_token_mask[batch_idx], as_tuple=False).flatten()
            if query_positions.numel() == 0:
                solutions.append("")
                continue
            query_logits = logits[batch_idx, query_positions].float().index_select(-1, action_token_ids)
            predicted_offsets = query_logits.argmax(dim=-1)
            predicted_ids = action_token_ids.index_select(0, predicted_offsets).detach().cpu().tolist()
            solutions.append("".join(self.latent_action_id_to_token[int(idx)] for idx in predicted_ids))
        return solutions

    def _build_training_messages(self, example: dict, instruction: str, latent_action: str | None) -> list[dict]:
        content = [{"type": "image", "image": img} for img in example["image"]]

        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
            prompt = cot_prompt.replace("{instruction}", instruction)
        else:
            prompt = instruction

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if latent_action is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": latent_action}]})
        return messages

    def _render_training_input(self, example: dict, instruction: str, latent_action: str | None) -> str:
        messages = self._build_training_messages(example, instruction, latent_action)
        return self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=latent_action is None,
        )

    def _print_training_sample_once(
        self,
        examples: List[dict],
        instructions: list[str],
        latent_solutions: list[str] | None,
    ) -> None:
        if self._printed_training_sample or not self.training:
            return
        if not logger.is_rank_zero():
            return

        example = examples[0]
        instruction = instructions[0]
        latent_action = latent_solutions[0] if latent_solutions is not None and len(latent_solutions) > 0 else None
        sample_text = self._render_training_input(example, instruction, latent_action)
        episode_index = example.get("episode_index", None)
        step_index = example.get("step_index", None)
        episode_path = example.get("episode_path", None)

        logger.info(
            "First training sample input (episode_index=%s, step_index=%s, episode_path=%s):\n%s",
            episode_index,
            step_index,
            episode_path,
            sample_text,
        )
        self._printed_training_sample = True

    def _should_use_latent_for_action(self) -> bool:
        return self.latent_action_enabled and not bool(self.latent_action_cfg.get("la_mask", True))

    def _encode_vl_hidden_states_with_solutions(
        self,
        batch_images: list,
        instructions: list[str],
        solutions: list[str] | None = None,
        latent_target_token_ids: list[list[int]] | None = None,
        labels: bool = False,
        project_for_action: bool = True,
        return_qwen_inputs: bool = False,
    ):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=solutions,
        )
        if labels and self.latent_action_mode == "query":
            if latent_target_token_ids is None:
                raise ValueError("latent_target_token_ids is required when latent_action.mode='query'.")
            qwen_inputs["labels"] = self._build_query_labels(qwen_inputs["input_ids"], latent_target_token_ids)
        if not labels and "labels" in qwen_inputs:
            qwen_inputs.pop("labels")
        attention_mask = qwen_inputs.get("attention_mask", None)
        if project_for_action:
            attention_mask = self._mask_latent_action_attention(attention_mask, qwen_inputs.get("input_ids", None))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if project_for_action:
                if self.action_model is None or self.num_action_dit_layers <= 0:
                    raise RuntimeError("Action model is not initialized. Set latent_action.train_action=true.")
                vl_embs_list = list(outputs.hidden_states[-self.num_action_dit_layers :])
                vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            else:
                vl_embs_list = [outputs.hidden_states[-1]]
        if return_qwen_inputs:
            return vl_embs_list, attention_mask, outputs, qwen_inputs
        return vl_embs_list, attention_mask, outputs

    def _continuous_action_loss(self, examples: List[dict], vl_embs_list: list, backbone_attention_mask):
        if self.action_model is None:
            raise RuntimeError("Action model is not initialized. Set latent_action.train_action=true.")
        actions = [example["action"] for example in examples]
        base_hidden = vl_embs_list[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            # In action-expert mode, feed the raw state (subset-aligned with
            # `examples`) to the Action DiT; in instruction mode it is already
            # encoded in the prompt, so the DiT gets no state token.
            state_repeated = None
            if self.state_mode == "action_expert" and examples and "state" in examples[0]:
                state = torch.tensor(
                    np.array([example["state"] for example in examples]),
                    device=base_hidden.device,
                    dtype=base_hidden.dtype,
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            return self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state=state_repeated,
                encoder_attention_mask=backbone_attention_mask,
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

    # ──────────────────────────────────────────────────────────────────
    #  SoftVQ latent-action path (KV-cache split, action/latent independent)
    # ──────────────────────────────────────────────────────────────────
    def _run_prefix_cache(self, batch_images: List, instructions: List[str]):
        """Single multimodal prefill that returns the KV cache and the per-layer
        prompt hidden states (the action branch reads these directly; the latent
        branch continues the suffix from the cache)."""
        prefix_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prefix_outputs = self.qwen_vl_interface(
                **prefix_inputs,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        past_key_values = getattr(prefix_outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(
                "QwenPI_v3_LA softvq requires use_cache=True to return past_key_values. "
                "Disable incompatible gradient-checkpointing/cache settings for this framework."
            )
        return prefix_inputs, past_key_values, prefix_outputs

    @staticmethod
    def _branch_cache(past_key_values):
        """Per-branch cache that shares the prefix KV tensors but appends into its
        own copy. ``DynamicLayer.update`` reassigns ``self.keys = torch.cat(...)``
        (not in place), so shallow-copied layers append into fresh slots while the
        prefix tensors stay shared with the original cache and gradients still flow
        back through them. ``copy.deepcopy`` would detach the prefix tensors."""
        branch = copy.copy(past_key_values)
        branch.layers = [copy.copy(layer) for layer in past_key_values.layers]
        return branch

    def _run_suffix_embeds_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        suffix_embeds: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        input_ids = prefix_inputs["input_ids"]
        attention_mask = prefix_inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        if suffix_embeds.ndim != 3 or suffix_embeds.shape[0] != batch_size:
            raise ValueError(f"Suffix embeds must be shaped [batch, suffix_len, hidden], got {tuple(suffix_embeds.shape)}.")
        prefix_len = past_key_values.get_seq_length()
        suffix_embeds = suffix_embeds.to(input_ids.device)
        suffix_mask = torch.ones(
            suffix_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        branch_attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        cache_position = torch.arange(
            prefix_len, prefix_len + suffix_embeds.shape[1], device=suffix_embeds.device
        )
        return self.qwen_vl_interface(
            inputs_embeds=suffix_embeds,
            attention_mask=branch_attention_mask,
            past_key_values=self._branch_cache(past_key_values),
            cache_position=cache_position,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

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
                    "QwenPI_v3_LA softvq requires `la_frames` with at least two frames. "
                    "Use datasets.vla_data.dataset_py=lerobot_la_datasets and enable latent_action."
                )
            pair_counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))

        if len(set(pair_counts)) != 1:
            raise ValueError(
                "QwenPI_v3_LA softvq requires the same number of SoftVQ frame pairs per batch "
                "because suffix embeddings are batched without padding, got pair counts "
                f"{pair_counts}. Tune latent_action.horizon_overrides so every embodiment "
                "yields the same pair count."
            )
        pair_count = pair_counts[0]

        distributions = self.latent_action_encoder.encode_distribution(frame_pairs)
        if distributions.ndim != 3:
            raise ValueError(f"SoftVQ encoder must return [num_pairs, tokens, codebook], got {tuple(distributions.shape)}.")
        if distributions.shape[0] != len(frame_pairs):
            raise ValueError(
                f"SoftVQ encoder returned {distributions.shape[0]} pairs, "
                f"but {len(frame_pairs)} frame pairs were provided."
            )
        if bool(self.latent_action_cfg.get("strict_num_bridge_tokens", True)):
            if distributions.shape[1] != self.num_bridge_tokens:
                raise ValueError(
                    f"SoftVQ encoder returned {distributions.shape[1]} tokens, "
                    f"but latent_action.num_bridge_tokens={self.num_bridge_tokens}."
                )
        elif distributions.shape[1] < self.num_bridge_tokens:
            raise ValueError(
                f"SoftVQ encoder returned {distributions.shape[1]} tokens, "
                f"fewer than latent_action.num_bridge_tokens={self.num_bridge_tokens}."
            )
        if distributions.shape[-1] != self.codebook_size:
            raise ValueError(
                f"SoftVQ encoder returned codebook size {distributions.shape[-1]}, "
                f"but latent_action.codebook_size={self.codebook_size}."
            )

        distributions = distributions[:, : self.num_bridge_tokens]
        distributions = distributions.reshape(len(examples), pair_count * self.num_bridge_tokens, self.codebook_size)
        return distributions

    def _collect_la_padding_mask(self, examples: List[dict]) -> torch.Tensor:
        """Per-sample mask (1.0=keep, 0.0=padded) for the SoftVQ latent-action loss.

        Samples whose latent-action frame window ran past the trajectory end were
        clamped/repeated by the dataset, producing degenerate frame pairs the
        tokenizer never saw. They are excluded from the KL objective.
        """
        flags = [bool(example.get("la_padded", False)) for example in examples]
        return torch.tensor([0.0 if padded else 1.0 for padded in flags], dtype=torch.float32)

    def _softvq_kl_loss_from_cache(
        self,
        prefix_inputs: dict,
        past_key_values,
        target_distributions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        if self.softvq_input_proj is None or self.softvq_output_proj is None:
            raise RuntimeError("SoftVQ projection heads are not initialized.")

        target_distributions = target_distributions.to(prefix_inputs["input_ids"].device).float()
        eps = float(self.latent_action_cfg.get("softvq", {}).get("kl_eps", 1e-8))
        target_probs = target_distributions.clamp_min(eps)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        suffix_embeds = self.softvq_input_proj(target_probs).to(dtype=self.qwen_vl_interface.model.dtype)
        outputs = self._run_suffix_embeds_from_cache(prefix_inputs, past_key_values, suffix_embeds)
        hidden = outputs.hidden_states[-1]
        logits = self.softvq_output_proj(hidden)
        if logits.shape != target_probs.shape:
            raise ValueError(f"SoftVQ logits shape {tuple(logits.shape)} does not match target {tuple(target_probs.shape)}.")

        log_probs = F.log_softmax(logits.float(), dim=-1)
        kl_per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=1)
        entropy_per_sample = -(target_probs * target_probs.log()).sum(dim=-1).mean(dim=1)
        max_prob_per_sample = target_probs.max(dim=-1).values.mean(dim=1)

        if valid_mask is None:
            valid_mask = torch.ones_like(kl_per_sample)
        else:
            valid_mask = valid_mask.to(kl_per_sample.device, dtype=kl_per_sample.dtype)
        valid_count = int(valid_mask.sum().item())
        denom = valid_mask.sum().clamp_min(1.0)

        kl_loss = (kl_per_sample * valid_mask).sum() / denom
        entropy = (entropy_per_sample * valid_mask).sum() / denom
        max_prob = (max_prob_per_sample * valid_mask).sum() / denom
        kl_loss = self._scale_action_loss_for_global_mean(kl_loss, valid_count)
        return {
            "latent_action_loss": kl_loss,
            "latent_softvq_kl_loss": kl_loss,
            "latent_softvq_target_entropy": entropy,
            "latent_softvq_target_max_prob": max_prob,
        }

    def _forward_softvq(self, examples: List[dict]) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions, _ = self._resolve_state_inputs(instructions, state)

        prefix_inputs, past_key_values, prefix_outputs = self._run_prefix_cache(batch_images, instructions)
        self._print_training_sample_once(examples, instructions, None)

        continuous_action_loss = None
        action_train_batch_size = 0
        if self.train_continuous_action:
            if self.action_model is None or self.num_action_dit_layers <= 0:
                raise RuntimeError("Action model is not initialized. Set latent_action.train_action=true.")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vl_embs_list = self._project_vl_hidden_for_action(
                    list(prefix_outputs.hidden_states[-self.num_action_dit_layers :])
                )
            backbone_attention_mask = prefix_inputs.get("attention_mask", None)
            action_examples, action_vl_embs_list, action_attention_mask = self._select_action_training_batch(
                examples,
                vl_embs_list,
                backbone_attention_mask,
            )
            action_train_batch_size = len(action_examples)
            if action_examples:
                continuous_action_loss = self._continuous_action_loss(
                    action_examples,
                    action_vl_embs_list,
                    action_attention_mask,
                )
            else:
                dummy_attention_mask = (
                    backbone_attention_mask[:1] if backbone_attention_mask is not None else None
                )
                continuous_action_loss = self._continuous_action_loss(
                    examples[:1],
                    [hidden[:1] for hidden in vl_embs_list],
                    dummy_attention_mask,
                ) * 0.0
            continuous_action_loss = self._scale_action_loss_for_global_mean(
                continuous_action_loss,
                action_train_batch_size,
            )

        latent_outputs = {}
        latent_action_loss = None
        if self.train_latent_action:
            gt_distributions = self._make_softvq_targets(examples)
            la_valid_mask = self._collect_la_padding_mask(examples)
            latent_outputs = self._softvq_kl_loss_from_cache(
                prefix_inputs, past_key_values, gt_distributions, la_valid_mask
            )
            latent_action_loss = latent_outputs["latent_action_loss"]

        total_loss = None
        if continuous_action_loss is not None:
            total_loss = continuous_action_loss * float(self.latent_action_cfg.action_loss_weight)
        if latent_action_loss is not None:
            weighted_latent = latent_action_loss * float(self.latent_action_cfg.latent_loss_weight)
            total_loss = weighted_latent if total_loss is None else total_loss + weighted_latent
        if total_loss is None:
            raise RuntimeError("No loss was computed. Check latent_action.train_latent/train_action config.")

        out = {"total_loss": total_loss, "action_loss": total_loss}
        if continuous_action_loss is not None:
            out["action_dit_loss"] = continuous_action_loss
            out["action_dis_loss"] = continuous_action_loss
            out["action_train_batch_size"] = action_train_batch_size
        if latent_action_loss is not None:
            out["latent_action_loss"] = latent_action_loss
        out.update(latent_outputs)
        return out

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if self.latent_action_backend == "softvq":
            return self._forward_softvq(examples)

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        # In action-expert mode the Action DiT reads state directly from
        # `examples` inside `_continuous_action_loss`, so we only need the
        # (possibly state-augmented) instructions here.
        instructions, _ = self._resolve_state_inputs(instructions, state)

        latent_solutions = None
        latent_action_loss = None
        vl_embs_list = None
        backbone_attention_mask = None
        condition_on_latent = self._should_use_latent_for_action()
        action_train_batch_size = 0
        latent_target_token_ids = None

        if self.latent_action_enabled and self.train_latent_action:
            latent_indices, latent_counts, latent_ar_solutions = self._make_latent_action_targets(examples, instructions)
            if self.latent_action_mode == "query":
                latent_solutions = self._make_latent_action_query_solutions(latent_counts)
                latent_target_token_ids = self._latent_indices_to_token_ids(latent_indices, latent_counts)
            else:
                latent_solutions = latent_ar_solutions
        self._print_training_sample_once(examples, instructions, latent_solutions)

        if self.latent_action_enabled and self.train_latent_action:
            vl_embs_list, backbone_attention_mask, outputs = self._encode_vl_hidden_states_with_solutions(
                batch_images,
                instructions,
                solutions=latent_solutions,
                latent_target_token_ids=latent_target_token_ids,
                labels=True,
                project_for_action=self.train_continuous_action,
            )
            latent_action_loss = outputs.loss
            if latent_action_loss is None or torch.isnan(latent_action_loss):
                latent_action_loss = torch.tensor(0.0, device=vl_embs_list[-1].device)

        continuous_action_loss = None
        if self.train_continuous_action:
            if vl_embs_list is None:
                if latent_solutions is None and condition_on_latent and self.latent_action_mode == "query":
                    latent_solutions = self._make_latent_action_query_solutions(self._get_query_pair_counts(examples))
                solutions = latent_solutions if condition_on_latent else None
                vl_embs_list, backbone_attention_mask, _ = self._encode_vl_hidden_states_with_solutions(
                    batch_images,
                    instructions,
                    solutions=solutions,
                    labels=False,
                    project_for_action=True,
                )
            action_examples, action_vl_embs_list, action_attention_mask = self._select_action_training_batch(
                examples,
                vl_embs_list,
                backbone_attention_mask,
            )
            action_train_batch_size = len(action_examples)
            if action_examples:
                continuous_action_loss = self._continuous_action_loss(
                    action_examples,
                    action_vl_embs_list,
                    action_attention_mask,
                )
            else:
                # Keep the action head in every rank's graph even when this local
                # batch has no target robot_type.  Otherwise DDP/ZeRO can hang
                # when different ranks use different parameter subsets.
                dummy_attention_mask = (
                    backbone_attention_mask[:1]
                    if backbone_attention_mask is not None
                    else None
                )
                continuous_action_loss = self._continuous_action_loss(
                    examples[:1],
                    [hidden[:1] for hidden in vl_embs_list],
                    dummy_attention_mask,
                ) * 0.0
            continuous_action_loss = self._scale_action_loss_for_global_mean(
                continuous_action_loss,
                action_train_batch_size,
            )

        total_loss = None
        if continuous_action_loss is not None:
            total_loss = continuous_action_loss * float(self.latent_action_cfg.action_loss_weight)
        if latent_action_loss is not None:
            weighted_latent = latent_action_loss * float(self.latent_action_cfg.latent_loss_weight)
            total_loss = weighted_latent if total_loss is None else total_loss + weighted_latent
        if total_loss is None:
            raise RuntimeError("No loss was computed. Check latent_action.train_latent/train_action config.")

        out = {"total_loss": total_loss, "action_loss": total_loss}
        if continuous_action_loss is not None:
            out["action_dit_loss"] = continuous_action_loss
            out["action_dis_loss"] = continuous_action_loss
            out["action_train_batch_size"] = action_train_batch_size
        if latent_action_loss is not None:
            out["latent_action_loss"] = latent_action_loss
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> dict:
        if not self.train_continuous_action or self.action_model is None:
            raise RuntimeError("Cannot predict continuous actions when latent_action.train_action=false.")
        # SoftVQ keeps the action branch independent of the latent objective and
        # relies on future frames that are unavailable at inference, so action
        # prediction is exactly the base QwenPI_v3 prompt-only path.
        if self.latent_action_backend == "softvq":
            return Qwen_PI_v3.predict_action(self, examples=examples, **kwargs)
        if not self._should_use_latent_for_action():
            return super().predict_action(examples=examples, **kwargs)

        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions, action_state = self._resolve_state_inputs(instructions, state)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        if self.latent_action_mode == "query":
            query_solutions = self._make_latent_action_query_solutions(self._get_query_pair_counts(examples))
            vl_embs_list, backbone_attention_mask, outputs, qwen_inputs = self._encode_vl_hidden_states_with_solutions(
                batch_images,
                instructions,
                solutions=query_solutions,
                labels=False,
                project_for_action=True,
                return_qwen_inputs=True,
            )
            latent_solutions = self._extract_query_latent_action_solutions(outputs.logits, qwen_inputs["input_ids"])
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
            max_new_tokens = self.latent_action_cfg.get("latent_generation_max_extra_tokens", None)
            if max_new_tokens is None:
                max_new_tokens = int(self.latent_action_cfg.num_codes_per_pair) * 4
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=int(max_new_tokens),
            )
            latent_solutions = self._extract_latent_action_solutions(generated_ids)
            vl_embs_list, backbone_attention_mask, _ = self._encode_vl_hidden_states_with_solutions(
                batch_images,
                instructions,
                solutions=latent_solutions,
                labels=False,
                project_for_action=True,
            )
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        action_state = (
            torch.from_numpy(np.array(action_state)).to(vl_embs_list[-1].device, dtype=vl_embs_list[-1].dtype)
            if action_state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state=action_state,
                encoder_attention_mask=backbone_attention_mask,
            )
        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "latent_action_solutions": latent_solutions,
        }

    def _extract_latent_action_solutions(self, generated_ids: torch.LongTensor) -> list[str]:
        token_set = set(self.latent_action_token_ids)
        solutions = []
        for seq in generated_ids:
            tokens = [self.latent_action_id_to_token[int(idx)] for idx in seq.tolist() if int(idx) in token_set]
            solutions.append("".join(tokens))
        return solutions
