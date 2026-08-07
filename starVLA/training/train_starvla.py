# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader, build_vla_eval_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.experiment_tracker import build_experiment_tracker
from starVLA.training.trainer_utils.metrics_jsonl import metrics_jsonl_path
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    create_accelerator_from_config,
    enable_torch_load_legacy_pickle,
    normalize_dotlist_args,
)

enable_torch_load_legacy_pickle()

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def _is_qwen_metaquery_la_training(cfg) -> bool:
    return str(cfg.framework.name) == "QwenMetaQuery_LA"


def _supports_dual_vla_dataloaders(cfg) -> bool:
    return str(cfg.framework.name) in {
        "QwenMetaQuery_LA",
        "QwenPI_v3_LA",
        "QwenPI_v4_LA",
        "QwenPI_v5_LA",
        "QwenWM_LA",
        "QwenWMv2_LA"
    }


def _dataloader_loss_modes(cfg) -> dict:
    """Per-dataloader forward loss modes; defaults preserve legacy latent/action split."""
    modes = {"latent": "latent", "action": "action"}
    configured = cfg.trainer.get("dataloader_loss_modes", None)
    if configured:
        for key in ("latent", "action"):
            value = configured.get(key, None)
            if value is not None and str(value).strip() != "":
                modes[key] = str(value).strip().lower()
    return modes


def _use_dual_vla_dataloaders(cfg) -> bool:
    return bool(cfg.trainer.get("use_dual_vla_dataloaders", False))


def _make_action_dataloader_cfg(cfg):
    base_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
    action_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    latent_action_cfg = action_cfg.datasets.vla_data.get("latent_action", None)
    if latent_action_cfg is None:
        action_cfg.datasets.vla_data.latent_action = OmegaConf.create({})
        latent_action_cfg = action_cfg.datasets.vla_data.latent_action
    # When the action dataloader runs a joint/latent forward (e.g. QwenWM_LA), it
    # needs `la_frames`, so keep latent-action frame packing enabled. Otherwise the
    # action loader only forwards actions and latent packing is skipped for speed.
    action_loss_mode = _dataloader_loss_modes(cfg)["action"]
    latent_action_cfg.enabled = action_loss_mode in {"joint", "latent"}
    latent_action_cfg.load_action = True
    latent_action_cfg.load_state = True
    return action_cfg


def _make_eval_dataloader_cfg(cfg, action_cfg=None):
    """Build eval dataloader config: action/state for MSE, no la_frames packing."""
    base = action_cfg if action_cfg is not None else cfg
    base_cfg = base.unwrap() if isinstance(base, AccessTrackedConfig) else base
    eval_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    latent_action_cfg = eval_cfg.datasets.vla_data.get("latent_action", None)
    if latent_action_cfg is None:
        eval_cfg.datasets.vla_data.latent_action = OmegaConf.create({})
        latent_action_cfg = eval_cfg.datasets.vla_data.latent_action
    latent_action_cfg.enabled = False
    latent_action_cfg.load_action = True
    latent_action_cfg.load_state = True
    return eval_cfg


def _make_latent_dataloader_cfg(cfg):
    base_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
    latent_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    latent_data_mix = latent_cfg.datasets.vla_data.get("latent_data_mix", None)
    if latent_data_mix is None or str(latent_data_mix).strip() == "":
        raise ValueError(
            "trainer.use_dual_vla_dataloaders=true requires "
            "datasets.vla_data.latent_data_mix for the latent-action dataloader. "
            "datasets.vla_data.data_mix is reserved for action-generation training in dual mode."
        )
    latent_cfg.datasets.vla_data.data_mix = latent_data_mix
    latent_action_cfg = latent_cfg.datasets.vla_data.get("latent_action", None)
    if latent_action_cfg is None:
        latent_cfg.datasets.vla_data.latent_action = OmegaConf.create({})
        latent_action_cfg = latent_cfg.datasets.vla_data.latent_action
    latent_action_cfg.load_action = False
    latent_action_cfg.load_state = False
    latent_cfg.datasets.vla_data.include_state = False
    return latent_cfg


def _get_vla_dataloader_batch_size(cfg, dataloader_name: str) -> int | None:
    vla_data_cfg = cfg.datasets.vla_data
    override_key = f"{dataloader_name}_per_device_batch_size"
    override_batch_size = getattr(vla_data_cfg, override_key, None)
    if override_batch_size is None:
        return None
    return int(override_batch_size)


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Prepare VLA training data."""
    data_cfg = cfg
    is_metaquery_la = _is_qwen_metaquery_la_training(cfg)
    use_dual_dataloaders = _use_dual_vla_dataloaders(cfg)
    if use_dual_dataloaders and not _supports_dual_vla_dataloaders(cfg):
        raise ValueError(
            "trainer.use_dual_vla_dataloaders=true is only supported for "
            "framework.name in {'QwenMetaQuery_LA', 'QwenPI_v3_LA', 'QwenPI_v4_LA', 'QwenPI_v5_LA', "
            "'QwenWM_LA'}."
        )

    latent_batch_size = _get_vla_dataloader_batch_size(cfg, "latent") if use_dual_dataloaders else None
    if use_dual_dataloaders:
        data_cfg = _make_latent_dataloader_cfg(cfg)
        logger.info(
            "Creating VLA latent Dataset with Mixture `%s` (load_action=%s, load_state=%s)",
            data_cfg.datasets.vla_data.data_mix,
            data_cfg.datasets.vla_data.latent_action.load_action,
            data_cfg.datasets.vla_data.latent_action.load_state,
        )
    else:
        logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(
        cfg=data_cfg,
        dataset_py=data_cfg.datasets.vla_data.dataset_py,
        save_dataset_stats=not (is_metaquery_la or use_dual_dataloaders),
        batch_size=latent_batch_size,
    )
    vla_action_train_dataloader = None
    action_cfg = None
    if use_dual_dataloaders:
        action_cfg = _make_action_dataloader_cfg(cfg)
        action_batch_size = _get_vla_dataloader_batch_size(cfg, "action")
        logger.info(
            "Creating VLA action Dataset with Mixture `%s` (load_action=%s, load_state=%s)",
            action_cfg.datasets.vla_data.data_mix,
            action_cfg.datasets.vla_data.latent_action.load_action,
            action_cfg.datasets.vla_data.latent_action.load_state,
        )
        vla_action_train_dataloader = build_dataloader(
            cfg=action_cfg,
            dataset_py=action_cfg.datasets.vla_data.dataset_py,
            save_dataset_stats=True,
            batch_size=action_batch_size,
        )
    eval_cfg = _make_eval_dataloader_cfg(cfg, action_cfg=action_cfg)
    vla_eval_dataloader = build_vla_eval_dataloader(
        cfg=eval_cfg,
        num_samples=getattr(cfg.trainer, "eval_num_samples", None),
        batch_size=getattr(cfg.trainer, "eval_batch_size", None),
        seed=cfg.seed,
    )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, vla_eval_dataloader, vla_action_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        vla_eval_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_action_train_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_action_train_dataloader = vla_action_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.use_dual_vla_dataloaders = vla_action_train_dataloader is not None
        self.dataloader_loss_modes = _dataloader_loss_modes(cfg)
        # Frameworks whose two dual streams share the same VLM input layout can run
        # them as one batched forward, which keeps the VLM off tiny per-stream batches.
        supports_merge = getattr(model, "supports_merged_dual_forward", None)
        self.merged_dual_forward = self.use_dual_vla_dataloaders and bool(
            supports_merge is not None and supports_merge(self.dataloader_loss_modes)
        )

        self.completed_steps = 0
        self.tracker = None
        self.latent_per_device_batch_size = self._get_effective_batch_size("latent")
        self.total_batch_size = self._calculate_total_batch_size(self.latent_per_device_batch_size)
        self.action_per_device_batch_size = (
            self._get_effective_batch_size("action")
            if self.use_dual_vla_dataloaders
            else self.latent_per_device_batch_size
        )
        self.action_total_batch_size = (
            self._calculate_total_batch_size(self.action_per_device_batch_size)
            if self.use_dual_vla_dataloaders
            else None
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        self.dump_parameter_status(self.model, self.config.output_dir)
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(model=self.model, cfg=self.config)
        # lr_scheduler is intentionally never passed to accelerator.prepare() (see
        # setup_distributed_training calls below), so it must be registered explicitly
        # for accelerator.save_state()/load_state() to actually persist/restore it.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        self._adjust_lr_scheduler_for_resume()

        if self.use_dual_vla_dataloaders:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_action_train_dataloader = (
                self.setup_distributed_training(
                    self.accelerator,
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.vla_action_train_dataloader,
                )
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        self._resume_full_state_if_needed()
        self._init_tracker()

    def _get_effective_batch_size(self, dataloader_name: str) -> int:
        override_key = f"{dataloader_name}_per_device_batch_size"
        override_batch_size = getattr(self.config.datasets.vla_data, override_key, None)
        if override_batch_size is not None:
            return int(override_batch_size)
        return int(self.config.datasets.vla_data.per_device_batch_size)

    def _calculate_total_batch_size(self, per_device_batch_size: int):
        """Calculate global batch size."""
        return (
            per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_tracker(self):
        """Initialize experiment tracking."""
        if self.accelerator.is_main_process:
            self.tracker = build_experiment_tracker(
                self.config,
                log_dir=os.path.join(self.config.output_dir, "wandb"),
                group="vla-train",
                metrics_jsonl_path=metrics_jsonl_path(self.config.output_dir),
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._full_state_resume_path = None

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            full_state_path, full_state_step = self._find_latest_full_state()
            if full_state_path is not None:
                self._full_state_resume_path = full_state_path
                self.completed_steps = full_state_step
                logger.info(
                    f"Will resume from full training state: {full_state_path}, step={full_state_step} "
                    "(actual load happens after accelerator.prepare)"
                )
                return

            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

        extra_weight_checkpoint = getattr(self.config.trainer, "extra_weight_checkpoint", None)
        if extra_weight_checkpoint:
            self.model = self.load_extra_weights(self.model, extra_weight_checkpoint)
            logger.info(f"Loaded extra weight checkpoint: {extra_weight_checkpoint}")

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps.

        Skipped when resuming from a full training state: lr_scheduler is registered
        via accelerator.register_for_checkpointing() (see prepare_training), so
        accelerator.load_state() restores its step count directly. Manually stepping
        it here as well would double-advance it.
        """
        if self._full_state_resume_path is not None:
            logger.info("Skipping manual LR scheduler adjustment (full state resume will restore scheduler)")
            return
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _find_latest_full_state(self):
        """Find the latest full training state directory.

        Returns:
            (path, step) if found, (None, 0) otherwise.
        """
        states_dir = os.path.join(self.config.output_dir, "training_states")
        if not os.path.exists(states_dir):
            return None, 0

        state_dirs = []
        for d in os.listdir(states_dir):
            if not d.startswith("state_"):
                continue
            full_path = os.path.join(states_dir, d)
            if not os.path.isdir(full_path):
                continue
            try:
                step = int(d.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            state_dirs.append((full_path, step))

        if not state_dirs:
            return None, 0

        state_dirs.sort(key=lambda x: x[1])
        latest_path, latest_step = state_dirs[-1]

        trainer_state_file = os.path.join(latest_path, "trainer_state.json")
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file) as f:
                trainer_state = json.load(f)
            latest_step = trainer_state.get("completed_steps", latest_step)

        return latest_path, latest_step

    def _resume_full_state_if_needed(self):
        """Load full training state after accelerator.prepare() if flagged during init."""
        if self._full_state_resume_path is None:
            return
        self.accelerator.load_state(self._full_state_resume_path)
        logger.info(f"✅ Full training state loaded from {self._full_state_resume_path}, step={self.completed_steps}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _save_full_state(self):
        """Save full training state (model, optimizer, scheduler, RNG) via accelerator.

        Controlled by ``trainer.save_full_state`` (bool, default False).
        All ranks must participate because DeepSpeed/FSDP shards are per-rank.
        """
        if not getattr(self.config.trainer, "save_full_state", True):
            return

        states_dir = os.path.join(self.config.output_dir, "training_states")
        state_dir = os.path.join(states_dir, f"state_{self.completed_steps}")

        self.accelerator.save_state(state_dir)

        if self.accelerator.is_main_process:
            with open(os.path.join(state_dir, "trainer_state.json"), "w") as f:
                json.dump({"completed_steps": self.completed_steps}, f)
            logger.info(f"✅ Full training state saved at {state_dir}")
            self._cleanup_old_full_states()

        self.accelerator.wait_for_everyone()

    def _cleanup_old_full_states(self):
        """Remove oldest full-state directories when exceeding max_full_state_checkpoints."""
        max_ckpts = getattr(self.config.trainer, "max_full_state_checkpoints", 2)
        if max_ckpts is None or int(max_ckpts) <= 0:
            return
        max_ckpts = int(max_ckpts)

        states_dir = os.path.join(self.config.output_dir, "training_states")
        if not os.path.exists(states_dir):
            return

        state_dirs = []
        for d in os.listdir(states_dir):
            if not d.startswith("state_"):
                continue
            full_path = os.path.join(states_dir, d)
            if not os.path.isdir(full_path):
                continue
            try:
                step = int(d.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            state_dirs.append((full_path, step))

        state_dirs.sort(key=lambda x: x[1])

        while len(state_dirs) > max_ckpts:
            oldest_path, oldest_step = state_dirs.pop(0)
            shutil.rmtree(oldest_path, ignore_errors=True)
            logger.info(f"🗑️ Removed old training state: {oldest_path} (step {oldest_step})")

    def _reduce_mean_metric(self, value: torch.Tensor) -> torch.Tensor:
        """All-reduce a scalar metric across ranks; every rank must call this together."""
        return self.accelerator.reduce(value.detach(), reduction="mean")

    def _log_metrics(self, metrics):
        """Record training metrics.

        Tensor metrics stay on device until a logging step. ``action_dit_loss`` /
        ``action_loss`` are reduced to a global mean; ``latent_action_loss`` is
        already the sum of active dual-stream latent losses from ``_train_step``.
        """
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return

        prepared = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                tensor = value.detach()
                if key in {"action_dit_loss", "action_loss"}:
                    # All ranks must participate in the reduce.
                    tensor = self._reduce_mean_metric(tensor)
                if dist.get_rank() == 0:
                    prepared[key] = float(tensor)
            elif dist.get_rank() == 0:
                prepared[key] = value

        if dist.get_rank() != 0:
            return

        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", str(i))
            prepared[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        prepared["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
        self.tracker.log(prepared, step=self.completed_steps)
        logger.info(f"Step {self.completed_steps}, Loss: {prepared})")

    def _dataloader_ranges(self, dataloader_name: str):
        ranges_cfg = self.config.trainer.get("dataloader_active_ranges", None)
        if ranges_cfg is None:
            return [[0, None]]
        return ranges_cfg.get(dataloader_name, [[0, None]])

    def _is_dataloader_active(self, dataloader_name: str, step: int) -> bool:
        ranges = self._dataloader_ranges(dataloader_name)
        if ranges is None:
            return True
        for range_spec in ranges:
            if len(range_spec) != 2:
                raise ValueError(
                    f"trainer.dataloader_active_ranges.{dataloader_name} entries must be [start, end], "
                    f"got {range_spec}."
                )
            start, end = range_spec
            start = int(start)
            end = None if end is None else int(end)
            if step >= start and (end is None or step < end):
                return True
        return False

    def _active_dataloaders_for_step(self, step: int) -> dict[str, bool]:
        if not self.use_dual_vla_dataloaders:
            return {"latent": True, "action": False}

        active = {
            "latent": self._is_dataloader_active("latent", step),
            "action": self._is_dataloader_active("action", step),
        }
        if not active["latent"] and not active["action"]:
            raise ValueError(
                "No QwenMetaQuery_LA dataloader is active at step "
                f"{step}. Check trainer.dataloader_active_ranges."
            )
        return active

    def _count_consumed_batches(self, dataloader_name: str) -> int:
        if not self.use_dual_vla_dataloaders:
            return self.completed_steps * self.accelerator.gradient_accumulation_steps

        active_steps = sum(
            1
            for step in range(int(self.completed_steps))
            if self._is_dataloader_active(dataloader_name, step)
        )
        return active_steps * self.accelerator.gradient_accumulation_steps

    def _init_dataloader_iterator(self, dataloader, iter_attr: str, epoch_attr: str, consumed_batches: int, label: str):
        setattr(self, epoch_attr, 0)

        if consumed_batches <= 0:
            setattr(self, iter_attr, iter(dataloader))
            return

        try:
            dataloader_length = len(dataloader)
        except TypeError:
            dataloader_length = 0

        if dataloader_length <= 0:
            self.accelerator.print(
                f"Unable to determine {label} dataloader length; skipping consumed batches in the current iterator only."
            )
            setattr(
                self,
                iter_attr,
                iter(self.accelerator.skip_first_batches(dataloader, num_batches=consumed_batches)),
            )
            return

        epoch_count = consumed_batches // dataloader_length
        batches_to_skip = consumed_batches % dataloader_length
        setattr(self, epoch_attr, epoch_count)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_count)

        resumed_dataloader = dataloader
        if batches_to_skip > 0:
            resumed_dataloader = self.accelerator.skip_first_batches(
                dataloader,
                num_batches=batches_to_skip,
            )

        self.accelerator.print(f"Resumed {label} dataloader at epoch {epoch_count}, batch offset {batches_to_skip}")
        setattr(self, iter_attr, iter(resumed_dataloader))

    def _create_data_iterators(self):
        """Create data iterators."""
        self._init_dataloader_iterator(
            self.vla_train_dataloader,
            "vla_iter",
            "vla_epoch_count",
            self._count_consumed_batches("latent"),
            "VLA latent",
        )
        if self.use_dual_vla_dataloaders:
            self._init_dataloader_iterator(
                self.vla_action_train_dataloader,
                "vla_action_iter",
                "vla_action_epoch_count",
                self._count_consumed_batches("action"),
                "VLA action",
            )

    def _next_from_iterator(self, dataloader, iter_attr: str, epoch_attr: str):
        iterator = getattr(self, iter_attr)
        try:
            return next(iterator)
        except StopIteration:
            epoch_count = getattr(self, epoch_attr, 0)
            iterator, epoch_count = TrainerUtils._reset_dataloader(dataloader, epoch_count)
            setattr(self, iter_attr, iterator)
            setattr(self, epoch_attr, epoch_count)
            return next(iterator)

    def _get_next_batch(self, active_dataloaders: dict[str, bool] | None = None):
        """Get next batch (automatically handle data loop)."""
        if not self.use_dual_vla_dataloaders:
            return self._next_from_iterator(self.vla_train_dataloader, "vla_iter", "vla_epoch_count")

        active_dataloaders = active_dataloaders or {"latent": True, "action": True}
        batches = {}
        if active_dataloaders.get("latent", False):
            batches["latent"] = self._next_from_iterator(self.vla_train_dataloader, "vla_iter", "vla_epoch_count")
        if active_dataloaders.get("action", False):
            batches["action"] = self._next_from_iterator(
                self.vla_action_train_dataloader,
                "vla_action_iter",
                "vla_action_epoch_count",
            )
        return batches

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self.update_dynamic_sampling_weights(self.vla_train_dataloader, self.completed_steps)
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            active_dataloaders = self._active_dataloaders_for_step(self.completed_steps)
            sampling_metrics = self.update_dynamic_sampling_weights(
                self.vla_train_dataloader, self.completed_steps
            )
            if self.use_dual_vla_dataloaders:
                action_sampling_metrics = self.update_dynamic_sampling_weights(
                    self.vla_action_train_dataloader, self.completed_steps
                )
                sampling_metrics.update(action_sampling_metrics)
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch(active_dataloaders)
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, active_dataloaders=active_dataloaders)
            step_metrics.update(sampling_metrics)
            t_end_model = time.perf_counter()

            did_optimizer_step = self.accelerator.sync_gradients
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if not did_optimizer_step:
                continue

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                self._save_full_state()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate action prediction on a fixed validation subset."""
        step_metrics = step_metrics or {}
        model = self.accelerator.unwrap_model(self.model)
        if not getattr(model, "train_continuous_action", True) or getattr(model, "action_model", True) is None:
            if self.accelerator.is_main_process:
                step_metrics["validation/skipped_action_mse"] = 1
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        was_training = self.model.training
        self.model.eval()

        total_squared_error = 0.0
        total_elements = 0
        total_samples = 0
        total_batches = 0
        action_eval_robot_types = (
            model._get_action_train_robot_types()
            if hasattr(model, "_get_action_train_robot_types")
            else None
        )

        eval_error = None
        try:
            with torch.inference_mode():
                for examples in self.vla_eval_dataloader:
                    if action_eval_robot_types is not None:
                        examples = [
                            example
                            for example in examples
                            if str(example.get("robot_type", "")) in action_eval_robot_types
                        ]
                        if not examples:
                            continue

                    actions = np.asarray([example["action"] for example in examples])
                    output_dict = model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)
                    normalized_actions = np.asarray(output_dict["normalized_actions"])
                    if actions.shape != normalized_actions.shape:
                        if (
                            actions.ndim == 3
                            and normalized_actions.ndim == 3
                            and actions.shape[0] == normalized_actions.shape[0]
                            and actions.shape[2] == normalized_actions.shape[2]
                            and actions.shape[1] >= normalized_actions.shape[1]
                        ):
                            actions = actions[:, -normalized_actions.shape[1] :, :]
                        else:
                            raise ValueError(
                                f"Validation action shape mismatch: pred={normalized_actions.shape}, "
                                f"target={actions.shape}"
                            )

                    diff = normalized_actions - actions
                    total_squared_error += float(np.sum(diff**2))
                    total_elements += int(diff.size)
                    total_samples += int(actions.shape[0])
                    total_batches += 1
        except Exception as exc:
            eval_error = exc
            logger.exception("Action validation failed; skipping this validation pass.")
            total_squared_error = 0.0
            total_elements = 0
            total_samples = 0
            total_batches = 0
        finally:
            if was_training:
                self.model.train()

        eval_totals = torch.tensor(
            [total_squared_error, total_elements, total_samples, total_batches],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        eval_error_flag = torch.tensor(
            1 if eval_error is not None else 0,
            dtype=torch.int64,
            device=self.accelerator.device,
        )
        if dist.is_initialized():
            dist.all_reduce(eval_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_error_flag, op=dist.ReduceOp.MAX)

        if self.accelerator.is_main_process:
            global_squared_error, global_elements, global_samples, global_batches = eval_totals.tolist()
            if int(eval_error_flag.item()) > 0:
                step_metrics["validation/skipped_action_mse"] = 1
            elif global_elements > 0:
                step_metrics["mse_score"] = global_squared_error / global_elements
            else:
                step_metrics["validation/skipped_action_mse"] = 1
            step_metrics["validation/num_samples"] = int(global_samples)
            step_metrics["validation/num_batches"] = int(global_batches)

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size (latent) = {self.latent_per_device_batch_size}")
            if self.use_dual_vla_dataloaders:
                logger.info(f"  Per device batch size (action) = {self.action_per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size (latent) = {self.total_batch_size}")
            if self.use_dual_vla_dataloaders:
                logger.info(f"  Total batch size (action) = {self.action_total_batch_size}")

    @staticmethod
    def _extract_action_dit_loss(output_dict: dict) -> torch.Tensor | None:
        if "action_dit_loss" in output_dict:
            return output_dict["action_dit_loss"]
        if "continuous_action_loss" in output_dict:
            return output_dict["continuous_action_loss"]
        if "action_loss" in output_dict:
            return output_dict["action_loss"]
        return None

    def _train_step(self, batch_vla, batch_vlm=None, active_dataloaders: dict[str, bool] | None = None):
        """Execute single training step."""
        profile_la_timing = os.getenv("PROFILE_LA_TIMING", "0").lower() in {"1", "true", "yes", "on"}
        backward_optim_start = None
        action_dit_loss = None
        latent_action_loss = None
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.use_dual_vla_dataloaders:
                    active_dataloaders = active_dataloaders or {"latent": True, "action": True}
                    output_dict = {}
                    total_loss = None

                    latent_active = active_dataloaders.get("latent", False)
                    action_active = active_dataloaders.get("action", False)
                    latent_output = None
                    action_output = None

                    if latent_active and action_active and self.merged_dual_forward:
                        # One VLM forward over both streams; rows never attend across
                        # the batch axis, so the losses match the two-forward path.
                        dual_output = self.model.forward(
                            batch_vla,
                            loss_mode="dual",
                            dual_loss_modes=self.dataloader_loss_modes,
                        )
                        latent_output = dual_output["streams"]["latent"]
                        action_output = dual_output["streams"]["action"]
                        output_dict.update(
                            {k: v for k, v in dual_output.items() if k.startswith("timing/")}
                        )
                    else:
                        if latent_active:
                            latent_output = self.model.forward(
                                batch_vla["latent"], loss_mode=self.dataloader_loss_modes["latent"]
                            )
                        if action_active:
                            action_output = self.model.forward(
                                batch_vla["action"], loss_mode=self.dataloader_loss_modes["action"]
                            )

                    if latent_output is not None:
                        total_loss = latent_output["total_loss"]
                        output_dict.update(latent_output)
                        if "latent_action_loss" in latent_output:
                            latent_action_loss = latent_output["latent_action_loss"]

                    if action_output is not None:
                        total_loss = (
                            action_output["total_loss"]
                            if total_loss is None
                            else total_loss + action_output["total_loss"]
                        )
                        output_dict.update(action_output)
                        if "latent_action_loss" in action_output:
                            action_latent = action_output["latent_action_loss"]
                            latent_action_loss = (
                                action_latent
                                if latent_action_loss is None
                                else latent_action_loss + action_latent
                            )
                        action_dit_loss = self._extract_action_dit_loss(action_output)

                    if total_loss is None:
                        raise RuntimeError("No active QwenMetaQuery_LA dataloader produced a loss.")
                    output_dict["total_loss"] = total_loss
                    if latent_action_loss is not None:
                        # Keep the summed dual-stream latent loss; do not keep the
                        # overwritten single-stream value from ``output_dict.update``.
                        output_dict["latent_action_loss"] = latent_action_loss
                else:
                    output_dict = self.model.forward(batch_vla)
                    total_loss = output_dict["total_loss"] if "total_loss" in output_dict else output_dict["action_loss"]
                    action_dit_loss = self._extract_action_dit_loss(output_dict)
                    if "latent_action_loss" in output_dict:
                        latent_action_loss = output_dict["latent_action_loss"]

            if profile_la_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_optim_start = time.perf_counter()

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if profile_la_timing and backward_optim_start is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                output_dict["timing/backward_optim"] = time.perf_counter() - backward_optim_start

        # Keep tensors until ``_log_metrics``; avoid per-step ``.item()`` sync.
        log_dict = {"total_loss": total_loss.detach()}
        if action_dit_loss is not None:
            log_dict["action_dit_loss"] = action_dit_loss.detach()
        if latent_action_loss is not None:
            log_dict["latent_action_loss"] = latent_action_loss.detach()
        if "action_train_batch_size" in output_dict:
            log_dict["action_train_batch_size"] = output_dict["action_train_batch_size"]
        for key, value in output_dict.items():
            if key.startswith("timing/"):
                log_dict[key] = float(value)
        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and self.tracker is not None:
            self.tracker.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    if os.getenv("ALLOW_TF32", "0").lower() in {"1", "true", "yes", "on"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for fp32 matmul / cudnn")

    accelerator = create_accelerator_from_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_eval_dataloader, vla_action_train_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_eval_dataloader=vla_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_action_train_dataloader=vla_action_train_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
