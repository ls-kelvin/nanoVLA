"""QwenWM_LA: QwenPI_v4 with decoupled in-head latent-action flow matching.

Unlike :class:`Qwen_PI_v4_LA` (which supervises latent actions on appended VLM
bridge-token positions), this framework moves latent-action learning **into the
action head**.  A dedicated head
(:class:`LayerwiseMetaqueryWMFlowmatchingActionHead`) runs two independent
flow-matching streams that share the DiT backbone and VLM hidden states:

  - action stream: conditioned on VLM hidden states + state (state cannot attend
    to action tokens, pi0-style block-causal mask);
  - latent stream: conditioned on VLM hidden states only.

Latent-action targets are Sharla post-quantize / pre-``output_proj`` embeddings
(backend is fixed to ``sharla``).

Dual-dataloader usage (see trainer.dataloader_loss_modes):
  - main / action dataloader -> ``loss_mode='joint'`` (action + latent);
  - secondary / latent dataloader -> ``loss_mode='latent'`` (latent only).
"""

from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v4 import Qwen_PI_v4
from starVLA.model.framework.share_tools import load_state_dict_ignore_pretrained_latent_encoder
from starVLA.model.modules.action_model.LayerwiseMetaqueryWM_ActionHeader import (
    LayerwiseMetaqueryWMFlowmatchingActionHead,
)
from starVLA.model.modules.latent_action import build_latent_action_encoder
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWM_LA")
class Qwen_WM_LA(Qwen_PI_v4):
    """QwenVL + decoupled latent-action/action flow-matching head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        load_latent_action_encoder = bool(kwargs.pop("load_latent_action_encoder", True))
        super().__init__(config=config, **kwargs)
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        self.latent_action_enabled = bool(self.latent_action_cfg.get("enabled", True))
        self.latent_action_backend = str(self.latent_action_cfg.get("backend", "sharla")).lower()
        if self.latent_action_backend != "sharla":
            raise NotImplementedError(
                "QwenWM_LA only supports backend='sharla', "
                f"got {self.latent_action_backend!r}."
            )
        self.train_latent_action = bool(self.latent_action_cfg.get("train_latent", True))
        self.train_continuous_action = bool(self.latent_action_cfg.get("train_action", True))
        if not self.train_latent_action and not self.train_continuous_action:
            raise ValueError("At least one of latent_action.train_latent or latent_action.train_action must be true.")
        self.detach_vl_embs_for_action_head = bool(
            self.latent_action_cfg.get("detach_vl_embs_for_action_head", False)
        )

        self.latent_action_encoder = None
        if self.latent_action_enabled and self.train_latent_action and load_latent_action_encoder:
            self.latent_action_encoder = build_latent_action_encoder(self.config)

        if self.latent_action_encoder is not None:
            self.latent_query_num = int(self.latent_action_encoder.query_num)
            latent_action_dim = int(self.latent_action_encoder.latent_dim)
        else:
            self.latent_query_num = int(self.latent_action_cfg.get("query_num", 8))
            cfg_dim = self.latent_action_cfg.get("latent_dim", None)
            if cfg_dim is None:
                cfg_dim = getattr(self.config.framework.action_model, "latent_action_dim", None)
            if cfg_dim is None:
                raise ValueError(
                    "latent_action.latent_dim must be set when the latent-action encoder is not loaded "
                    "(e.g. deployment with load_latent_action_encoder=false). "
                    "Set framework.latent_action.latent_dim or ensure "
                    "framework.action_model.latent_action_dim is present in the checkpoint config."
                )
            latent_action_dim = int(cfg_dim)

        # Rebuild the action head as the WM variant with the latent dimension injected.
        self.config.framework.action_model.latent_action_dim = latent_action_dim
        self.action_model = LayerwiseMetaqueryWMFlowmatchingActionHead(global_config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        self._load_latent_norm_stats(latent_action_dim)
        self._printed_training_sample = False

    def _load_latent_norm_stats(self, latent_action_dim: int) -> None:
        """Load Sharla embedding mean/std used to standardise continuous targets."""
        sharla_cfg = self.latent_action_cfg.get("sharla", {}) or {}
        norm_stats_path = sharla_cfg.get("norm_stats_path", None)
        if norm_stats_path is None or str(norm_stats_path) in ("", "null", "None"):
            raise ValueError(
                "framework.latent_action.sharla.norm_stats_path is required for QwenWM_LA. "
                "Build it via scripts/cache_sharla_embeddings.py (writes latent_norm_stats.json)."
            )
        from pathlib import Path
        import json

        path = Path(str(norm_stats_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Latent embedding norm stats not found: {path}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError(f"norm stats mean/std must be 1-D, got {tuple(mean.shape)} / {tuple(std.shape)}")
        if mean.numel() != latent_action_dim or std.numel() != latent_action_dim:
            raise ValueError(
                f"norm stats dim {mean.numel()} does not match latent_action_dim={latent_action_dim}."
            )
        if torch.any(std <= 0):
            raise ValueError("latent embedding norm stats std must be strictly positive.")
        self.register_buffer("latent_embed_mean", mean, persistent=False)
        self.register_buffer("latent_embed_std", std, persistent=False)

    def _normalize_latent_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        mean = self.latent_embed_mean.to(device=embeddings.device, dtype=embeddings.dtype)
        std = self.latent_embed_std.to(device=embeddings.device, dtype=embeddings.dtype)
        return (embeddings - mean) / std

    # ------------------------------------------------------------------ #
    # Config / state_dict
    # ------------------------------------------------------------------ #
    def _ensure_latent_action_defaults(self) -> None:
        from omegaconf import OmegaConf

        defaults = {
            "enabled": True,
            "backend": "sharla",
            "train_latent": True,
            "train_action": True,
            "latent_loss_weight": 1.0,
            "action_loss_weight": 1.0,
            "detach_vl_embs_for_action_head": False,
            "action_train_robot_types": None,
            "image_size": [224, 224],
            "query_num": 8,
            "latent_dim": None,
            "sharla": {
                "config_path": None,
                "ckpt_path": None,
                "image_size": [224, 224],
                "strict_load": True,
                "norm_stats_path": None,
            },
        }
        current = self.config.framework.get("latent_action", {})
        self.config.framework.latent_action = OmegaConf.merge(
            OmegaConf.create(defaults),
            current,
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [key for key in state_dict if key.startswith("latent_action_encoder.")]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return load_state_dict_ignore_pretrained_latent_encoder(
            self, state_dict, strict=strict, assign=assign
        )

    # ------------------------------------------------------------------ #
    # Helpers (mirrored from Qwen_PI_v4_LA)
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

    def _build_latent_frame_pairs(self, examples: List[dict]):
        frame_pairs = []
        counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    "QwenWM_LA requires `la_frames` with at least two frames for latent-action targets. "
                    "Enable datasets.vla_data.latent_action for this dataloader."
                )
            counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
        if len(set(counts)) != 1:
            raise ValueError(f"Latent-action frame-pair counts must match within a batch, got {counts}.")
        return frame_pairs, counts

    def _scale_action_loss_for_global_mean(self, loss: torch.Tensor, local_count: int) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return loss
        count = torch.tensor(float(local_count), device=loss.device, dtype=loss.dtype)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0:
            return loss * 0.0
        return loss * (dist.get_world_size() * float(local_count) / count)

    # ------------------------------------------------------------------ #
    # Latent targets (continuous tokenizer embeddings)
    # ------------------------------------------------------------------ #
    def _load_cached_embeddings(self, examples: List[dict], device, dtype) -> torch.Tensor:
        cached = []
        for example in examples:
            value = example.get("la_cached_embedding", None)
            if value is None:
                raise ValueError("Batch mixes cached and uncached latent embeddings.")
            if torch.is_tensor(value):
                tensor = value.detach().to(device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(value, device=device, dtype=dtype)
            if tensor.ndim != 2:
                raise ValueError(
                    f"la_cached_embedding must be [num_tokens, latent_dim], got {tuple(tensor.shape)}."
                )
            if tensor.shape[1] != self.action_model.latent_action_dim:
                raise ValueError(
                    f"la_cached_embedding latent_dim={tensor.shape[1]} does not match "
                    f"action head latent_action_dim={self.action_model.latent_action_dim}."
                )
            cached.append(tensor)
        token_counts = {item.shape[0] for item in cached}
        if len(token_counts) != 1:
            raise ValueError(f"Cached latent token counts must match within a batch, got {sorted(token_counts)}.")
        return torch.stack(cached, dim=0)

    def _make_continuous_targets(self, examples: List[dict], device, dtype) -> torch.Tensor:
        if examples and "la_cached_embedding" in examples[0]:
            embeddings = self._load_cached_embeddings(examples, device, dtype)
            return self._normalize_latent_embeddings(embeddings)

        if self.latent_action_encoder is None:
            raise RuntimeError(
                "Latent-action encoder is not initialized and no la_cached_embedding was provided."
            )
        frame_pairs, counts = self._build_latent_frame_pairs(examples)
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
        if embeddings.shape[2] != self.action_model.latent_action_dim:
            raise ValueError(
                f"{self.latent_action_backend} encoder returned latent_dim={embeddings.shape[2]}, "
                f"but action head latent_action_dim={self.action_model.latent_action_dim}."
            )
        pair_count = counts[0]
        embeddings = embeddings.reshape(
            len(examples), pair_count * self.latent_query_num, embeddings.shape[2]
        ).to(device=device, dtype=dtype)
        return self._normalize_latent_embeddings(embeddings)

    # ------------------------------------------------------------------ #
    # Head invocation
    # ------------------------------------------------------------------ #
    def _run_head(
        self,
        vl_embs_list: list[torch.Tensor],
        attention_mask: torch.Tensor | None,
        examples: List[dict],
        compute_action: bool,
        compute_latent: bool,
        latent_targets: torch.Tensor | None,
    ) -> dict:
        base_hidden = vl_embs_list[-1]
        device = base_hidden.device
        dtype = base_hidden.dtype
        r = self.repeated_diffusion_steps

        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = None
            state_t = None
            latent_t = None
            if compute_action:
                if "action" not in examples[0]:
                    raise ValueError("QwenWM_LA action/joint loss requires examples with `action`.")
                actions = torch.tensor(
                    np.array([example["action"] for example in examples]), device=device, dtype=dtype
                )
                actions_t = actions[:, -self.action_horizon :, :].repeat(r, 1, 1)
                if "state" in examples[0]:
                    state = torch.tensor(
                        np.array([example["state"] for example in examples]), device=device, dtype=dtype
                    )
                    state_t = state.repeat(r, 1, 1)
            if compute_latent:
                latent_t = latent_targets.to(device=device, dtype=dtype).repeat(r, 1, 1)

            if self.detach_vl_embs_for_action_head:
                vl_r = [hidden.detach().repeat(r, 1, 1) for hidden in vl_embs_list]
            else:
                vl_r = [hidden.repeat(r, 1, 1) for hidden in vl_embs_list]
            am_r = attention_mask.repeat(r, 1) if attention_mask is not None else None

            return self.action_model(
                vl_r,
                actions_t,
                state_t,
                latent_targets=latent_t,
                encoder_attention_mask=am_r,
            )

    def _print_training_sample_once(self, examples: List[dict], instructions: list[str]) -> None:
        if self._printed_training_sample or not self.training or not logger.is_rank_zero():
            return
        example = examples[0]
        logger.info(
            "First QwenWM_LA training sample (episode_index=%s, step_index=%s):\n"
            "  instruction: %s\n"
            "  la_frame_offsets: %s\n"
            "  latent_query_num: %s | latent_action_dim: %s",
            example.get("episode_index", None),
            example.get("step_index", None),
            instructions[0],
            example.get("la_frame_offsets", None),
            self.latent_query_num,
            self.action_model.latent_action_dim,
        )
        self._printed_training_sample = True

    # ------------------------------------------------------------------ #
    # Train / inference
    # ------------------------------------------------------------------ #
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
        vl_embs_list, attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        self._print_training_sample_once(examples, instructions)

        action_loss = None
        latent_loss = None
        action_train_batch_size = 0

        if loss_mode == "latent":
            latent_targets = self._make_continuous_targets(
                examples, vl_embs_list[-1].device, vl_embs_list[-1].dtype
            )
            outputs = self._run_head(
                vl_embs_list, attention_mask, examples,
                compute_action=False, compute_latent=True, latent_targets=latent_targets,
            )
            latent_loss = outputs["latent_loss"]
        else:
            sel_examples, sel_vl, sel_am = self._select_action_training_batch(
                examples, vl_embs_list, attention_mask
            )
            action_train_batch_size = len(sel_examples)
            if sel_examples:
                latent_targets = None
                if compute_latent:
                    latent_targets = self._make_continuous_targets(
                        sel_examples, sel_vl[-1].device, sel_vl[-1].dtype
                    )
                outputs = self._run_head(
                    sel_vl, sel_am, sel_examples,
                    compute_action=compute_action, compute_latent=compute_latent,
                    latent_targets=latent_targets,
                )
                if compute_action:
                    action_loss = self._scale_action_loss_for_global_mean(
                        outputs["action_loss"], action_train_batch_size
                    )
                if compute_latent:
                    latent_loss = outputs["latent_loss"]
            else:
                # No action-robot rows in this batch: keep training latent on all
                # examples (joint) and emit a zero action loss to preserve the graph.
                if compute_latent:
                    latent_targets_all = self._make_continuous_targets(
                        examples, vl_embs_list[-1].device, vl_embs_list[-1].dtype
                    )
                    outputs_latent = self._run_head(
                        vl_embs_list, attention_mask, examples,
                        compute_action=False, compute_latent=True, latent_targets=latent_targets_all,
                    )
                    latent_loss = outputs_latent["latent_loss"]
                if compute_action:
                    dummy_am = attention_mask[:1] if attention_mask is not None else None
                    outputs_dummy = self._run_head(
                        [hidden[:1] for hidden in vl_embs_list], dummy_am, examples[:1],
                        compute_action=True, compute_latent=False, latent_targets=None,
                    )
                    action_loss = self._scale_action_loss_for_global_mean(
                        outputs_dummy["action_loss"] * 0.0, 0
                    )

        total_loss = None
        if action_loss is not None:
            total_loss = action_loss * float(self.latent_action_cfg.get("action_loss_weight", 1.0))
        if latent_loss is not None:
            latent_weight = float(self.latent_action_cfg.get("latent_loss_weight", 1.0))
            weighted_latent = latent_loss * latent_weight
            total_loss = weighted_latent if total_loss is None else total_loss + weighted_latent
        if total_loss is None:
            raise RuntimeError("No loss was computed. Check loss_mode and latent_action train flags.")

        out = {
            "total_loss": total_loss,
            "action_loss": total_loss,
            "action_train_batch_size": action_train_batch_size,
        }
        if action_loss is not None:
            out["action_dit_loss"] = action_loss
            out["action_dis_loss"] = action_loss
        if latent_loss is not None:
            out["latent_action_loss"] = latent_loss
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
