# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenWMv2_LA: QwenPI_v5's causal-prefix + prefix-KV expert, with QwenWM_LA's
decoupled latent-action flow-matching branch.

Architecture is v5 (see ``QwenPI_v5``/``QwenPI_v5_LA``): the VLM runs as a
stock causal Qwen3-VL forward, and per-layer prefix K/V are recomputed and fed
to a Qwen2 action expert (``DualStreamFlowMatching``/``QwenVLWithExpert``).

Latent-action *prediction method* is QwenWM_LA's (see
``LayerwiseMetaqueryWMFlowmatchingActionHead``): latent and action are two
**independent** flow-matching streams that share the VLM prefix and expert
weights but run separate forward passes with separate noise/velocity and
separate input/output projections (``DualStreamFlowMatchingWM``). Unlike
``QwenPI_v5_LA``'s bridge-token discrete/soft-KL supervision, latent targets
here are continuous Sharla post-quantize / pre-``output_proj`` embeddings
(standardised by cached mean/std, exactly as in ``QwenWM_LA``); the VLM input
is never touched (no bridge tokens, no tokenizer/vocab resize).

Both branches need per-layer prefix K/V (``build_prefix_kvs`` + ``run_expert``),
so -- unlike ``QwenWM_LA`` -- merging two dataloaders' VLM forward passes into
one (``supports_merged_dual_forward``/``_forward_dual``) is worthwhile here and
mirrors ``QwenPI_v5_LA``'s implementation.
"""

from typing import List, Optional

import torch
import torch.distributed as dist

from starVLA.model.framework.VLM4A.QwenPI_v5 import Qwen_PI_v5
from starVLA.model.framework.share_tools import load_state_dict_ignore_pretrained_latent_encoder
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingWM
from starVLA.model.modules.latent_action import build_latent_action_encoder
from starVLA.model.modules.vlm.QWen3 import merge_qwen3_vl_inputs
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenWMv2_LA")
class Qwen_WMv2_LA(Qwen_PI_v5):
    """v5 prefix-KV expert + decoupled WM-style latent-action flow-matching branch."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        load_latent_action_encoder = bool(kwargs.pop("load_latent_action_encoder", True))
        super().__init__(config=config, **kwargs)

        self.latent_action_enabled = bool(self.latent_action_cfg.get("enabled", True))
        self.latent_action_backend = str(self.latent_action_cfg.get("backend", "sharla")).lower()
        if self.latent_action_backend != "sharla":
            raise NotImplementedError(
                f"QwenWMv2_LA only supports backend='sharla', got {self.latent_action_backend!r}."
            )
        self.train_latent_action = bool(self.latent_action_cfg.get("train_latent", True))
        self.train_continuous_action = bool(self.latent_action_cfg.get("train_action", True))
        if not self.train_latent_action and not self.train_continuous_action:
            raise ValueError(
                "At least one of latent_action.train_latent or latent_action.train_action must be true."
            )
        self.detach_vl_embs_for_action_head = bool(
            self.latent_action_cfg.get("detach_vl_embs_for_action_head", False)
        )

        self.latent_action_encoder = None
        if self.latent_action_enabled and self.train_latent_action and load_latent_action_encoder:
            self.latent_action_encoder = build_latent_action_encoder(self.config)
            encoder_dim = int(self.latent_action_encoder.latent_dim)
            if encoder_dim != self.action_model.latent_action_dim:
                raise ValueError(
                    f"framework.latent_action.latent_dim={self.action_model.latent_action_dim} does not "
                    f"match the Sharla encoder's actual latent_dim={encoder_dim}."
                )
            self.latent_query_num = int(self.latent_action_encoder.query_num)
        else:
            self.latent_query_num = int(self.latent_action_cfg.get("query_num", 8))

        self._load_latent_norm_stats(self.action_model.latent_action_dim)
        self._printed_training_sample = False

    # ------------------------------------------------------------------ #
    # Config / action-model construction
    # ------------------------------------------------------------------ #
    def _build_action_model(self) -> DualStreamFlowMatchingWM:
        """Resolve ``latent_action.latent_dim`` and build the WM-variant action model.

        Runs mid-way through ``Qwen_PI_v5.__init__`` (after ``self.config`` is
        merged, before the VLM is constructed), so ``latent_action_dim`` must be
        known from config rather than inferred from the (heavy, not-yet-loaded)
        Sharla encoder -- otherwise the action model, which owns the VLM here,
        would need to be rebuilt and the VLM loaded twice.
        """
        self._ensure_latent_action_defaults()
        self.latent_action_cfg = self.config.framework.latent_action
        latent_dim = self.latent_action_cfg.get("latent_dim", None)
        if latent_dim is None:
            raise ValueError(
                "framework.latent_action.latent_dim is required for QwenWMv2_LA (e.g. the Sharla "
                "codebook_dim). It cannot be auto-inferred from the encoder here because that would "
                "require constructing the action model (which owns the VLM) twice."
            )
        self.config.framework.action_model.latent_action_dim = int(latent_dim)
        return DualStreamFlowMatchingWM(global_config=self.config)

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
        for key in [key for key in state_dict if key.startswith("latent_action_encoder.")]:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return load_state_dict_ignore_pretrained_latent_encoder(
            self, state_dict, strict=strict, assign=assign
        )

    def _load_latent_norm_stats(self, latent_action_dim: int) -> None:
        """Load Sharla embedding mean/std used to standardise continuous targets."""
        sharla_cfg = self.latent_action_cfg.get("sharla", {}) or {}
        norm_stats_path = sharla_cfg.get("norm_stats_path", None)
        if norm_stats_path is None or str(norm_stats_path) in ("", "null", "None"):
            raise ValueError(
                "framework.latent_action.sharla.norm_stats_path is required for QwenWMv2_LA. "
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
    # per-sample action selection (mirrors QwenPI_v5_LA)
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

    def _action_row_weights(self, examples: List[dict], device) -> tuple[torch.Tensor | None, int]:
        """Rows excluded by ``action_train_robot_types`` get a zero action mask.

        The action stream's suffix rides in the same shared VLM prefix as the
        latent stream, so we cannot slice the batch; selection is applied to
        the flow-matching mask instead (same approach as ``QwenPI_v5_LA``).
        """
        allowed = self._get_action_train_robot_types()
        if allowed is None:
            return None, len(examples)
        flags = [1.0 if str(example.get("robot_type", "")) in allowed else 0.0 for example in examples]
        weights = torch.tensor(flags, dtype=torch.float32, device=device)
        return weights, int(sum(flags))

    def _prepare_action_inputs(self, examples: List[dict], device):
        state = self._prepare_state(examples, device)
        actions, action_mask = self._prepare_actions(examples, device)
        row_weights, action_train_batch_size = self._action_row_weights(examples, device)
        if row_weights is not None:
            action_mask = action_mask * row_weights[:, None, None]
        if state is None:
            state = torch.zeros(actions.shape[0], self.state_dim, device=device, dtype=actions.dtype)
        return state, actions, action_mask, action_train_batch_size

    def _scale_loss_for_global_mean(self, loss: torch.Tensor, local_count: int) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return loss
        count = torch.tensor(float(local_count), device=loss.device, dtype=loss.dtype)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0:
            return loss * 0.0
        return loss * (dist.get_world_size() * float(local_count) / count)

    # ------------------------------------------------------------------ #
    # latent targets (continuous Sharla embeddings, mirrors QwenWM_LA)
    # ------------------------------------------------------------------ #
    def _build_latent_frame_pairs(self, examples: List[dict]):
        frame_pairs = []
        counts = []
        for example in examples:
            frames = example.get("la_frames", None)
            if frames is None or len(frames) < 2:
                raise ValueError(
                    "QwenWMv2_LA requires `la_frames` with at least two frames for latent-action targets. "
                    "Enable datasets.vla_data.latent_action for this dataloader."
                )
            counts.append(len(frames) - 1)
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
        if len(set(counts)) != 1:
            raise ValueError(f"Latent-action frame-pair counts must match within a batch, got {counts}.")
        return frame_pairs, counts

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

    def _print_training_sample_once(self, examples: List[dict], instructions: list[str]) -> None:
        if self._printed_training_sample or not self.training or not logger.is_rank_zero():
            return
        example = examples[0]
        logger.info(
            "First QwenWMv2_LA training sample (episode_index=%s, step_index=%s):\n"
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
    # forward
    # ------------------------------------------------------------------ #
    def _resolve_loss_flags(self, loss_mode: str) -> tuple[bool, bool]:
        if loss_mode not in {"joint", "latent", "action"}:
            raise ValueError(f"loss_mode must be 'joint', 'latent', or 'action', got {loss_mode!r}.")

        compute_action = loss_mode in {"joint", "action"} and self.train_continuous_action
        compute_latent = loss_mode in {"joint", "latent"} and self.train_latent_action
        if loss_mode == "action" and not compute_action:
            raise RuntimeError("Action loss requested, but latent_action.train_action=false.")
        if loss_mode == "latent" and not compute_latent:
            raise RuntimeError("Latent loss requested, but latent_action.train_latent=false.")
        if compute_latent and not self.latent_action_enabled:
            raise RuntimeError("Latent-action loss requested, but latent action is disabled.")
        return compute_action, compute_latent

    def _maybe_detach_prefix_kvs(self, prefix_kvs):
        if not self.detach_vl_embs_for_action_head:
            return prefix_kvs
        return [(key.detach(), value.detach()) for key, value in prefix_kvs]

    def _assemble_losses(
        self,
        action_dit_loss: torch.Tensor | None,
        latent_action_loss: torch.Tensor | None,
        action_train_batch_size: int,
    ) -> dict:
        total_loss = None
        if action_dit_loss is not None:
            total_loss = action_dit_loss * float(self.latent_action_cfg.action_loss_weight)
        if latent_action_loss is not None:
            latent_weight = float(self.latent_action_cfg.get("latent_loss_weight", 1.0))
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
        return out

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

        action_dit_loss = None
        latent_action_loss = None
        action_train_batch_size = 0

        if compute_action:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            action_dit_loss = self.action_model.flow_matching_loss(
                prefix, prefix_kvs, state, actions, action_mask, num_repeats=self.repeated_diffusion_steps
            )
        if compute_latent:
            latent_targets = self._make_continuous_targets(examples, device, torch.float32)
            latent_action_loss = self.action_model.flow_matching_loss_latent(
                prefix, prefix_kvs, latent_targets, num_repeats=self.repeated_diffusion_steps
            )

        return self._assemble_losses(action_dit_loss, latent_action_loss, action_train_batch_size)

    # ------------------------------------------------------------------ #
    # merged dual-dataloader forward
    # ------------------------------------------------------------------ #
    def supports_merged_dual_forward(self, dual_loss_modes: dict | None = None) -> bool:
        """Both streams need per-layer prefix K/V, so merging is always valid
        (unlike ``QwenPI_v5_LA``, where a pure ``action`` stream skips the
        bridge-token suffix and would misalign the merged prefix layout).
        """
        modes = dual_loss_modes or {}
        return all(
            str(modes.get(name, "")).lower() in {"joint", "latent", "action"}
            for name in ("latent", "action")
        )

    def _forward_dual(self, batches: dict, dual_loss_modes: dict | None) -> dict:
        """Run both dual-dataloader streams through a single merged VLM prefix pass.

        Both action and latent branches read per-layer prefix K/V, so -- unlike
        ``QwenWM_LA`` (whose latent branch reads ``vl_embs_list`` directly and
        gains little from merging) -- sharing one VLM forward here saves a full
        VLM pass per training step.
        """
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

            action_dit_loss = None
            latent_action_loss = None
            action_train_batch_size = 0
            if stream["compute_action"]:
                state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                    stream["examples"], device
                )
                action_dit_loss = self.action_model.flow_matching_loss(
                    stream_prefix, prefix_kvs, state, actions, action_mask,
                    num_repeats=self.repeated_diffusion_steps,
                )
            if stream["compute_latent"]:
                latent_targets = self._make_continuous_targets(stream["examples"], device, torch.float32)
                latent_action_loss = self.action_model.flow_matching_loss_latent(
                    stream_prefix, prefix_kvs, latent_targets, num_repeats=self.repeated_diffusion_steps
                )

            outputs[stream["name"]] = self._assemble_losses(
                action_dit_loss, latent_action_loss, action_train_batch_size
            )

        return {"streams": outputs}
