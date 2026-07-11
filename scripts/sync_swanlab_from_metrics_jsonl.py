#!/usr/bin/env python3
"""
Backfill SwanLab metrics from a local metrics.jsonl backup.

Use this when training is still running (or has finished) but SwanLab stopped
recording while metrics.jsonl kept growing.

Example:
  source .venv/bin/activate
  python scripts/sync_swanlab_from_metrics_jsonl.py \\
    --output_dir results/Checkpoints2/my_run \\
    --sync-cloud
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.training.trainer_utils.metrics_jsonl import (  # noqa: E402
    METRICS_JSONL_FILENAME,
    dedupe_metrics_records,
    load_metrics_jsonl,
    metrics_jsonl_path,
)


def _load_run_config(output_dir: Path) -> dict[str, Any]:
    for name in ("config.full.yaml", "config.yaml"):
        config_path = output_dir / name
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    raise FileNotFoundError(
        f"Could not find config.full.yaml or config.yaml under {output_dir}"
    )


def _find_latest_swanlab_run_dir(wandb_dir: Path) -> Path:
    candidates = [path for path in wandb_dir.glob("run-*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No SwanLab run directory found under {wandb_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_swanlab_run_id(run_dir: Path) -> str:
    if not run_dir.name.startswith("run-"):
        raise ValueError(f"Unexpected SwanLab run directory name: {run_dir.name}")
    return run_dir.name.rsplit("-", 1)[-1]


def _prepare_records(
    metrics_path: Path,
    from_step: int,
    to_step: Optional[int],
) -> list[dict[str, Any]]:
    records = dedupe_metrics_records(load_metrics_jsonl(metrics_path))
    filtered: list[dict[str, Any]] = []
    for record in records:
        step = int(record["step"])
        if step < from_step:
            continue
        if to_step is not None and step > to_step:
            continue
        filtered.append(record)
    return filtered


def _metrics_for_log(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "step"}


def sync_swanlab_from_metrics_jsonl(
    output_dir: Path,
    *,
    metrics_jsonl: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    swanlab_run_id: Optional[str] = None,
    from_step: int = 0,
    to_step: Optional[int] = None,
    dry_run: bool = False,
    sync_cloud: bool = False,
    swanlab_mode: Optional[str] = None,
) -> int:
    output_dir = output_dir.resolve()
    metrics_path = (metrics_jsonl or metrics_jsonl_path(output_dir)).resolve()
    records = _prepare_records(metrics_path, from_step=from_step, to_step=to_step)
    if not records:
        print(f"No metrics records to sync from {metrics_path}")
        return 0

    run_config = _load_run_config(output_dir)
    project = run_config.get("wandb_project")
    workspace = run_config.get("wandb_entity")
    experiment_name = run_config.get("run_id")

    wandb_dir = output_dir / "wandb"
    resolved_run_dir = run_dir.resolve() if run_dir is not None else _find_latest_swanlab_run_dir(wandb_dir)
    resolved_run_id = swanlab_run_id or _extract_swanlab_run_id(resolved_run_dir)

    print(f"metrics.jsonl: {metrics_path}")
    print(f"SwanLab run dir: {resolved_run_dir}")
    print(f"SwanLab run id: {resolved_run_id}")
    print(f"Records to replay: {len(records)} (step {records[0]['step']} .. {records[-1]['step']})")

    if dry_run:
        preview = records[:3]
        if len(records) > 6:
            preview = preview + [{"step": "..."}] + records[-3:]
        print("Dry run preview:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    import swanlab

    init_kwargs: dict[str, Any] = {
        "project": project,
        "experiment_name": experiment_name,
        "logdir": str(wandb_dir),
        "workspace": workspace,
        "resume": "allow",
        "id": resolved_run_id,
    }
    if swanlab_mode is not None:
        init_kwargs["mode"] = swanlab_mode

    swanlab.init(**init_kwargs)
    try:
        for record in records:
            step = int(record["step"])
            swanlab.log(_metrics_for_log(record), step=step)
    finally:
        swanlab.finish()

    print(f"Replayed {len(records)} metric records into SwanLab run `{resolved_run_id}`.")

    if sync_cloud:
        cmd = ["swanlab", "sync", str(resolved_run_dir), "--id", resolved_run_id]
        if project:
            cmd.extend(["-p", str(project)])
        if workspace:
            cmd.extend(["-w", str(workspace)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay metrics.jsonl into an existing SwanLab run (resume mode)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Training output directory that contains metrics.jsonl and wandb/.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help=f"Path to metrics jsonl (default: <output_dir>/{METRICS_JSONL_FILENAME}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="SwanLab run directory under output_dir/wandb/ (default: latest run-*).",
    )
    parser.add_argument(
        "--swanlab-run-id",
        type=str,
        default=None,
        help="SwanLab experiment id (default: parsed from run directory name).",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=0,
        help="Only replay metrics with step >= this value.",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=None,
        help="Only replay metrics with step <= this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be replayed without touching SwanLab.",
    )
    parser.add_argument(
        "--sync-cloud",
        action="store_true",
        help="Run `swanlab sync` after replaying local metrics.",
    )
    parser.add_argument(
        "--swanlab-mode",
        type=str,
        default=os.environ.get("SWANLAB_MODE"),
        help="SwanLab mode for replay (default: SWANLAB_MODE env or SDK default).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sync_swanlab_from_metrics_jsonl(
        args.output_dir,
        metrics_jsonl=args.metrics_jsonl,
        run_dir=args.run_dir,
        swanlab_run_id=args.swanlab_run_id,
        from_step=args.from_step,
        to_step=args.to_step,
        dry_run=args.dry_run,
        sync_cloud=args.sync_cloud,
        swanlab_mode=args.swanlab_mode,
    )


if __name__ == "__main__":
    main()
