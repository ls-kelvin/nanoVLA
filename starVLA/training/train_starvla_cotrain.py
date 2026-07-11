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
import time
from pathlib import Path
from typing import Tuple

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
    create_accelerator_from_config,
    normalize_dotlist_args,
    setup_optimizer_and_scheduler,
)

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


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare co-training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)
    vla_eval_dataloader = build_vla_eval_dataloader(
        cfg=cfg,
        num_samples=getattr(cfg.trainer, "eval_num_samples", None),
        batch_size=getattr(cfg.trainer, "eval_batch_size", None),
        seed=cfg.seed,
    )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, vlm_train_dataloader, vla_eval_dataloader


class VLAMTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        vlm_train_dataloader,
        vla_eval_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.tracker = None
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so a later setup-step crash still
        # leaves a from_pretrained-able run dir behind.
        self._save_initial_configs()

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        extra_weight_checkpoint = getattr(self.config.trainer, "extra_weight_checkpoint", None)
        if extra_weight_checkpoint:
            self.model = self.load_extra_weights(self.model, extra_weight_checkpoint)
            logger.info(f"Loaded extra weight checkpoint: {extra_weight_checkpoint}")

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        self.dump_parameter_status(self.model, self.config.output_dir)
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(model=self.model, cfg=self.config)

        self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vlm_train_dataloader,
            )
        )

        self._init_tracker()
        self._init_checkpointing()

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
        logger.info(f"\U0001f4dd Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"\U0001f4ca Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
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

    def _init_checkpointing(self):
        """Initialize checkpoint directory."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

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

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            self.tracker.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        try:
            batch_vlm = next(self.vlm_iter)
        except StopIteration:
            if not hasattr(self, "vlm_epoch_count"):
                self.vlm_epoch_count = 0
            self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.vlm_train_dataloader, self.vlm_epoch_count)
            batch_vlm = next(self.vlm_iter)

        return batch_vla, batch_vlm

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
            sampling_metrics = self.update_dynamic_sampling_weights(
                self.vla_train_dataloader, self.completed_steps
            )
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
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
                dist.barrier()

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

                diff = normalized_actions - actions
                total_squared_error += float(np.sum(diff**2))
                total_elements += int(diff.size)
                total_samples += int(actions.shape[0])
                total_batches += 1

        if was_training:
            self.model.train()

        eval_totals = torch.tensor(
            [total_squared_error, total_elements, total_samples, total_batches],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        if dist.is_initialized():
            dist.all_reduce(eval_totals, op=dist.ReduceOp.SUM)

        if self.accelerator.is_main_process:
            global_squared_error, global_elements, global_samples, global_batches = eval_totals.tolist()
            if global_elements > 0:
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
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm):
        """Execute single training step."""
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                vla_total_loss = output_dict["total_loss"] if "total_loss" in output_dict else output_dict["action_loss"]
                if "action_dis_loss" in output_dict:
                    action_dis_loss = output_dict["action_dis_loss"]
                elif "continuous_action_loss" in output_dict:
                    action_dis_loss = output_dict["continuous_action_loss"]
                else:
                    action_dis_loss = output_dict["action_loss"]
            self.accelerator.backward(vla_total_loss)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unwrapped = self.accelerator.unwrap_model(self.model)
                vlm_output = unwrapped.qwen_vl_interface(**batch_vlm)
                vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
            self.accelerator.backward(vlm_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced.
            # See train_starvla.py for full explanation.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            log_dict.update(
                {
                    "total_loss": (vla_total_loss.detach() + vlm_loss.detach()).item(),
                    "action_dis_loss": action_dis_loss.item(),
                    "vlm_loss": vlm_loss.item(),
                }
            )
            if "latent_action_loss" in output_dict:
                log_dict["latent_action_loss"] = output_dict["latent_action_loss"].item()
            if "action_train_batch_size" in output_dict:
                log_dict["action_train_batch_size"] = output_dict["action_train_batch_size"]

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
    accelerator = create_accelerator_from_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vlm_train_dataloader, vla_eval_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        vla_eval_dataloader=vla_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
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
