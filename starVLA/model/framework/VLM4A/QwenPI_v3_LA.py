"""QwenPI_v3 with pluggable latent-action generation."""

from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

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
        self.train_latent_action = bool(self.latent_action_cfg.get("train_latent", True))
        self.train_continuous_action = bool(self.latent_action_cfg.get("train_action", True))
        if not self.train_latent_action and not self.train_continuous_action:
            raise ValueError("At least one of latent_action.train_latent or latent_action.train_action must be true.")

        self.latent_action_token_ids: list[int] = []
        self.latent_action_id_to_token: dict[int, str] = {}
        if self.latent_action_enabled:
            self._expand_latent_action_tokens()

        self.latent_action_encoder = None
        if self.latent_action_enabled and self.train_latent_action:
            self.latent_action_encoder = build_latent_action_encoder(self.config)

        self._printed_training_sample = False

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
            "codebook_size": 16,
            "num_codes_per_pair": 4,
            "token_format": "<robot_action_{i}>",
            "latent_generation_max_extra_tokens": None,
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
        current = self.config.framework.get("latent_action", {})
        self.config.framework.latent_action = OmegaConf.merge(OmegaConf.create(defaults), current)

    def _expand_latent_action_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        codebook_size = int(self.latent_action_cfg.codebook_size)
        token_format = str(self.latent_action_cfg.token_format)
        tokens = [token_format.format(i=i) for i in range(codebook_size)]

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

        action_token_min = min(self.latent_action_token_ids)
        action_token_max = max(self.latent_action_token_ids)
        latent_token_mask = (input_ids >= action_token_min) & (input_ids <= action_token_max)
        if not latent_token_mask.any():
            return attention_mask

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

    def _make_latent_action_solutions(self, examples: List[dict], instructions: list[str]) -> list[str]:
        if self.latent_action_encoder is None:
            raise ValueError("Latent-action encoder is not initialized.")
        frame_pairs, counts = self._build_latent_frame_pairs(examples)
        pair_instructions = []
        for instruction, count in zip(instructions, counts):
            pair_instructions.extend([instruction] * count)
        indices = self.latent_action_encoder.encode(frame_pairs, pair_instructions)
        return self._latent_indices_to_solutions(indices, counts)

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
        labels: bool = False,
        project_for_action: bool = True,
    ):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=solutions,
        )
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
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

            return self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state=None,
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

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions

        latent_solutions = None
        latent_action_loss = None
        vl_embs_list = None
        backbone_attention_mask = None
        condition_on_latent = self._should_use_latent_for_action()
        action_train_batch_size = 0

        if self.latent_action_enabled and self.train_latent_action:
            latent_solutions = self._make_latent_action_solutions(examples, instructions)
        self._print_training_sample_once(examples, instructions, latent_solutions)

        if self.latent_action_enabled and self.train_latent_action:
            vl_embs_list, backbone_attention_mask, outputs = self._encode_vl_hidden_states_with_solutions(
                batch_images,
                instructions,
                solutions=latent_solutions,
                labels=True,
                project_for_action=self.train_continuous_action,
            )
            latent_action_loss = outputs.loss
            if latent_action_loss is None or torch.isnan(latent_action_loss):
                latent_action_loss = torch.tensor(0.0, device=vl_embs_list[-1].device)

        continuous_action_loss = None
        if self.train_continuous_action:
            if vl_embs_list is None:
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
        if not self._should_use_latent_for_action():
            return super().predict_action(examples=examples, **kwargs)

        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        latent_solutions = None
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
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state=None,
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
