#!/usr/bin/env python3
"""Compute pairwise Wasserstein-2 distances between embodiment action-space groups."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.action_space_metrics import run_wasserstein_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity. Default: INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    cfg = OmegaConf.load(args.config)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"Config must resolve to a dict, got {type(cfg_dict)}")

    matrix_path, heatmap_path = run_wasserstein_from_config(cfg_dict)
    print(f"Wrote distance matrix to {matrix_path}")
    print(f"Wrote heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
