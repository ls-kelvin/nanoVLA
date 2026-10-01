#!/usr/bin/env python3
"""Offline evaluation of training checkpoints on an extra valset (e.g. eval_data_mixes),
logging results back into the *same* SwanLab run as the original training.

This reuses the trainer's eval path without modifying it:
- eval dataloader via ``_make_eval_dataloader_cfg`` + ``build_vla_eval_dataloader``
  (same deterministic subset: seed + num_samples),
- MSE accumulation identical to ``VLATrainer._accumulate_action_mse``,
- metric keys identical to ``VLATrainer.eval_action_model``
  (``validation/<tag>/mse_score`` etc.),
- SwanLab resume pattern identical to ``sync_swanlab_from_metrics_jsonl.py``.

Example:
  source .venv/bin/activate
  python scripts/eval_ckpts_extra_valset.py \
    --output_dir results/Checkpoints2/0809_hdf5_aloha_clean_eef_action_hdf5_arx_clean_random_eef_latent_qwenwmv3_4b_la_sharla_a2a_foresight_soft_kl \
    --eval_tag aloha_random --data_mix hdf5_aloha_random_eef \
    --hdf5_root /mnt/netdata/Team/Robot/Data/train/zzt_robotwin/dataset

Minimal smoke test (no SwanLab / metrics.jsonl writes):
  python scripts/eval_ckpts_extra_valset.py --output_dir ... \
    --eval_tag aloha_random --data_mix hdf5_aloha_random_eef \
    --hdf5_root ... --ckpt steps_90000 --num_samples 8 --batch_size 4 --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each checkpoint on an extra valset and log to the original SwanLab run."
    )
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Training run dir containing checkpoints/, config.full.yaml, wandb/.")
    parser.add_argument("--eval_tag", type=str, required=True,
                        help="Metric tag, e.g. aloha_random -> validation/aloha_random/mse_score.")
    parser.add_argument("--data_mix", type=str, required=True,
                        help="Registered data mix name, e.g. hdf5_aloha_random_eef.")
    parser.add_argument("--hdf5_root", type=Path, default=None,
                        help="Override ROBOTWIN2_HDF5_ROOT (must be set before starVLA imports).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Only eval this checkpoint (e.g. steps_90000). Default: all steps_*_pytorch_model.pt.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Eval samples (default: trainer.eval_num_samples from config).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Eval batch size (default: trainer.eval_batch_size from config).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--swanlab_mode", type=str, default=os.environ.get("SWANLAB_MODE"),
                        help="SwanLab mode (default: SWANLAB_MODE env or SDK default).")
    parser.add_argument("--swanlab_run_id", type=str, default=None,
                        help="SwanLab experiment id to resume (default: latest run-* dir by mtime). "
                             "Recommended to set explicitly when multiple run dirs exist.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute metrics only; do not touch SwanLab or metrics.jsonl.")
    return parser.parse_args()


# Parse args and set environment BEFORE importing starVLA: the data registry reads
# ROBOTWIN2_HDF5_ROOT at import time.
_ARGS = _parse_args()
if _ARGS.hdf5_root is not None:
    os.environ["ROBOTWIN2_HDF5_ROOT"] = str(_ARGS.hdf5_root)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from starVLA.dataloader import build_vla_eval_dataloader  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402
from starVLA.training.train_starvla import _make_eval_dataloader_cfg  # noqa: E402
from starVLA.training.trainer_utils.metrics_jsonl import (  # noqa: E402
    append_metrics_jsonl,
    load_metrics_jsonl,
    metrics_jsonl_path,
)
from starVLA.training.trainer_utils.trainer_tools import enable_torch_load_legacy_pickle  # noqa: E402

enable_torch_load_legacy_pickle()

OLD_HOME_PREFIX = "/mnt/netdata/Team/Personal/zzt/"
NEW_HOME_PREFIX = "/mnt/netdata/Team/Personal/zetong.zhou/"


def _patch_config_paths(node: Any) -> Any:
    """Recursively rewrite relocated absolute paths inside a config container."""
    if isinstance(node, str):
        if OLD_HOME_PREFIX in node:
            patched = node.replace(OLD_HOME_PREFIX, NEW_HOME_PREFIX)
            # Only rewrite when the original path is gone and the relocated one exists;
            # otherwise keep the original (files may live under either prefix).
            if not os.path.exists(node) and os.path.exists(patched):
                return patched
        return node
    if isinstance(node, dict):
        return {key: _patch_config_paths(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_patch_config_paths(value) for value in node]
    return node


def _load_run_config(output_dir: Path, hdf5_root: Optional[Path]):
    for name in ("config.full.yaml", "config.yaml"):
        config_path = output_dir / name
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            container = _patch_config_paths(OmegaConf.to_container(cfg, resolve=True))
            cfg = OmegaConf.create(container)
            break
    else:
        raise FileNotFoundError(f"No config.full.yaml/config.yaml under {output_dir}")

    if hdf5_root is not None:
        cfg.datasets.vla_data.data_root_dir = str(hdf5_root)
    return apply_config_compat(cfg)


def _list_checkpoints(output_dir: Path, only: Optional[str]) -> list[tuple[int, Path]]:
    ckpt_dir = output_dir / "checkpoints"
    pattern = re.compile(r"^steps_(\d+)_pytorch_model\.pt$")
    found = []
    for path in ckpt_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if only is not None:
        wanted = f"{only}_pytorch_model.pt" if not only.endswith(".pt") else only
        found = [(step, path) for step, path in found if path.name == wanted]
        if not found:
            raise FileNotFoundError(f"Checkpoint {only} not found under {ckpt_dir}")
    return found


def _accumulate_action_mse(model, eval_dataloader, action_eval_robot_types):
    """Mirror of VLATrainer._accumulate_action_mse (single process)."""
    total_squared_error = 0.0
    total_elements = 0
    total_samples = 0
    total_batches = 0
    with torch.inference_mode():
        for examples in eval_dataloader:
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
    return total_squared_error, total_elements, total_samples, total_batches


def _find_latest_swanlab_run_dir(wandb_dir: Path) -> Path:
    candidates = [path for path in wandb_dir.glob("run-*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No SwanLab run directory found under {wandb_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _merged_metrics_record(metrics_path: Path, step: int, new_metrics: dict[str, Any]) -> dict[str, Any]:
    """Merge with any existing record at this step so a future sync (which dedupes
    by step, keeping the last record) does not drop previously logged metrics."""
    merged: dict[str, Any] = {}
    if metrics_path.exists():
        for record in load_metrics_jsonl(metrics_path):
            if int(record["step"]) == step:
                merged = record  # keep the latest existing record for this step
    merged.update(new_metrics)
    return merged


def main() -> None:
    args = _ARGS
    output_dir = args.output_dir.resolve()
    device = torch.device(args.device)

    # accelerate's get_logger requires an initialized state even for single-process eval.
    from accelerate import PartialState

    PartialState()

    cfg = _load_run_config(output_dir, args.hdf5_root)
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = int(getattr(cfg.trainer, "eval_num_samples", 512))
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(getattr(cfg.trainer, "eval_batch_size", 8))

    checkpoints = _list_checkpoints(output_dir, args.ckpt)
    print(f"Run dir: {output_dir}")
    print(f"Checkpoints to eval: {[step for step, _ in checkpoints]}")
    print(f"Eval mix: {args.data_mix} (tag={args.eval_tag}), "
          f"num_samples={num_samples}, batch_size={batch_size}, seed={cfg.seed}")

    print("Building eval dataloader ...")
    eval_cfg = _make_eval_dataloader_cfg(cfg, data_mix=args.data_mix)
    eval_dataloader = build_vla_eval_dataloader(
        cfg=eval_cfg,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=cfg.seed,
    )

    print("Building model framework ...")
    model = build_framework(cfg)
    model.to(device)
    model.eval()

    action_eval_robot_types = (
        model._get_action_train_robot_types()
        if hasattr(model, "_get_action_train_robot_types")
        else None
    )

    swanlab_run = None
    metrics_path = metrics_jsonl_path(output_dir)
    if not args.dry_run:
        import swanlab

        wandb_dir = output_dir / "wandb"
        if args.swanlab_run_id:
            run_id = args.swanlab_run_id
            run_dir = f"<explicit id {run_id}>"
        else:
            resolved = _find_latest_swanlab_run_dir(wandb_dir)
            run_id = resolved.name.rsplit("-", 1)[-1]
            run_dir = str(resolved)
        init_kwargs: dict[str, Any] = {
            "project": cfg.get("wandb_project"),
            "experiment_name": cfg.get("run_id"),
            "logdir": str(wandb_dir),
            "workspace": cfg.get("wandb_entity"),
            "resume": "allow",
            "id": run_id,
        }
        if args.swanlab_mode is not None:
            init_kwargs["mode"] = args.swanlab_mode
        print(f"Resuming SwanLab run: {run_dir} (id={run_id})")
        swanlab_run = swanlab.init(**init_kwargs)

    try:
        for step, ckpt_path in checkpoints:
            print(f"\n=== Evaluating {ckpt_path.name} (step {step}) ===")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            incompatible = model.load_state_dict(state_dict, strict=False)
            del state_dict
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            print(f"Checkpoint loaded (strict=False): {len(missing)} missing, {len(unexpected)} unexpected keys")
            if missing:
                print(f"  missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"  unexpected (first 5): {unexpected[:5]}")

            total_sq, total_el, total_samples, total_batches = _accumulate_action_mse(
                model, eval_dataloader, action_eval_robot_types
            )
            if total_el <= 0:
                print(f"⚠️ step {step}: no eval elements accumulated, skipping upload")
                continue
            mse = total_sq / total_el
            metrics = {
                f"validation/{args.eval_tag}/mse_score": mse,
                f"validation/{args.eval_tag}/num_samples": int(total_samples),
                f"validation/{args.eval_tag}/num_batches": int(total_batches),
            }
            print(f"step {step}: mse={mse:.6f} samples={total_samples} batches={total_batches}")

            if args.dry_run:
                continue
            swanlab_run.log(metrics, step=step)
            record = _merged_metrics_record(metrics_path, step, metrics)
            record.pop("step", None)
            append_metrics_jsonl(metrics_path, step, record)
            print(f"step {step}: logged to SwanLab and metrics.jsonl")
    finally:
        if swanlab_run is not None:
            import swanlab

            swanlab.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
