#!/usr/bin/env python3
"""Pre-compute Sharla soft_kl distribution caches (per-episode ``.pt``).

Reads build options from ``soft_kl_cache`` in the training YAML.

    python scripts/cache_sharla_soft_kl_episodes.py \\
        --config-yaml examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_sharla.yaml

Multi-GPU: launch via the thin shell wrapper (or torchrun).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config-yaml",
        required=True,
        type=Path,
        help="Training YAML with a soft_kl_cache section.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode existing episode files (overrides soft_kl_cache.overwrite).",
    )
    return parser.parse_args()


def _maybe_init_distributed() -> tuple[bool, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True, local_rank


def main() -> None:
    args = parse_args()
    distributed, local_rank = _maybe_init_distributed()

    from omegaconf import OmegaConf
    from starVLA.dataloader.episode_soft_kl_cache import build_episode_soft_kl_cache

    config = OmegaConf.load(args.config_yaml)
    cache_cfg = config.get("soft_kl_cache", None)
    if cache_cfg is None:
        raise ValueError(f"Missing soft_kl_cache section in {args.config_yaml}")

    output_dir = cache_cfg.get("output_dir", None)
    if output_dir is None or str(output_dir) in ("", "null", "None"):
        raise ValueError("soft_kl_cache.output_dir is required")

    data_root_dir = cache_cfg.get("data_root_dir", None)
    if data_root_dir is not None and str(data_root_dir) not in ("", "null", "None"):
        config.datasets.vla_data.data_root_dir = str(data_root_dir)

    data_mixes = cache_cfg.get("data_mixes", None)
    if data_mixes is not None and str(data_mixes) not in ("", "null", "None"):
        data_mixes = [str(x) for x in data_mixes]
    else:
        data_mixes = None

    device_cfg = cache_cfg.get("device", None)
    if device_cfg is None or str(device_cfg) in ("", "null", "None"):
        if torch.cuda.is_available():
            device = f"cuda:{local_rank}" if distributed else "cuda"
        else:
            device = "cpu"
    else:
        device = str(device_cfg)

    overwrite = bool(args.overwrite) or bool(cache_cfg.get("overwrite", False))

    torch.set_float32_matmul_precision("high")
    try:
        build_episode_soft_kl_cache(
            config,
            cache_dir=str(output_dir),
            device=device,
            batch_size=int(cache_cfg.get("batch_size", 256)),
            num_workers=int(cache_cfg.get("num_workers", 8)),
            use_bf16=bool(cache_cfg.get("use_bf16", True)),
            overwrite=overwrite,
            data_mixes=data_mixes,
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
