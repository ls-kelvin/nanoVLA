from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from starVLA.training.trainer_utils.metrics_jsonl import append_metrics_jsonl

logger = logging.getLogger(__name__)


def _unwrap_config(cfg: Any) -> Any:
    if hasattr(cfg, "unwrap"):
        cfg = cfg.unwrap()
    return cfg


def _to_plain_config(cfg: Any) -> Any:
    cfg = _unwrap_config(cfg)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _select_tracker_backend(cfg: Any) -> str:
    explicit_backend = getattr(cfg, "tracker_backend", None)
    if explicit_backend:
        return str(explicit_backend).strip().lower()

    trackers = getattr(cfg, "trackers", None)
    if trackers:
        for candidate in trackers:
            candidate_name = str(candidate).strip().lower()
            if candidate_name in {"swanlab", "wandb"}:
                return candidate_name

    return "swanlab"


@dataclass
class ExperimentTracker:
    backend: str
    module: Any
    metrics_jsonl_path: Optional[Path] = None

    def log(self, metrics, step=None):
        if self.metrics_jsonl_path is not None and step is not None:
            append_metrics_jsonl(self.metrics_jsonl_path, step, metrics)

        try:
            if step is None:
                self.module.log(metrics)
            else:
                self.module.log(metrics, step=step)
        except Exception as exc:
            logger.warning("Experiment tracker log failed at step %s: %s", step, exc)

    def finish(self):
        finish = getattr(self.module, "finish", None)
        if callable(finish):
            finish()


def build_experiment_tracker(
    cfg: Any,
    log_dir: str,
    group: str = "vla-train",
    metrics_jsonl_path: str | Path | None = None,
) -> ExperimentTracker:
    backend = _select_tracker_backend(cfg)
    run_name = getattr(cfg, "run_id", None)
    project_name = getattr(cfg, "wandb_project", None)
    workspace_name = getattr(cfg, "wandb_entity", None)
    jsonl_path = Path(metrics_jsonl_path) if metrics_jsonl_path is not None else None

    if backend == "wandb":
        module = importlib.import_module("wandb")
        module.init(
            name=run_name,
            dir=log_dir,
            project=project_name,
            entity=workspace_name,
            group=group,
        )
        return ExperimentTracker(backend=backend, module=module, metrics_jsonl_path=jsonl_path)

    if backend == "swanlab":
        module = importlib.import_module("swanlab")
        module.init(
            project=project_name,
            experiment_name=run_name,
            logdir=log_dir,
            group=group,
            workspace=workspace_name,
            config=_to_plain_config(cfg),
        )
        return ExperimentTracker(backend=backend, module=module, metrics_jsonl_path=jsonl_path)

    raise ValueError(
        f"Unsupported tracker_backend `{backend}`. "
        "Expected one of: wandb, swanlab."
    )
