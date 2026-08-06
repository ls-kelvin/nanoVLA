#!/usr/bin/env python3
"""Pre-compute Sharla continuous embedding caches (per-episode ``.pt``).

Caches post-quantize / pre-``output_proj`` embeddings and writes
``latent_norm_stats.json`` under the same root.

Reads build options from ``embedding_cache`` in the training YAML.

    python scripts/cache_sharla_embeddings.py \\
        --config-yaml examples/Robotwin/new_train/starvla_qwenwm_hdf5_aloha_clean_random_la_sharla.yaml

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
        help="Training YAML with an embedding_cache section.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode existing episode files (overrides embedding_cache.overwrite).",
    )
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="Skip writing latent_norm_stats.json after cache build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override embedding_cache.output_dir.",
    )
    parser.add_argument(
        "--data-root-dir",
        type=Path,
        default=None,
        help="Override datasets.vla_data.data_root_dir for cache building.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Unused by the Python entry (shell wrapper reads YAML); accepted for CLI symmetry.",
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
    from starVLA.dataloader.episode_embedding_cache import build_episode_embedding_cache

    config = OmegaConf.load(args.config_yaml)
    cache_cfg = config.get("embedding_cache", None)
    if cache_cfg is None:
        raise ValueError(f"Missing embedding_cache section in {args.config_yaml}")

    output_dir = cache_cfg.get("output_dir", None)
    if args.output_dir is not None:
        output_dir = args.output_dir
    if output_dir is None or str(output_dir) in ("", "null", "None"):
        raise ValueError("embedding_cache.output_dir is required (or pass --output-dir)")

    data_root_dir = cache_cfg.get("data_root_dir", None)
    if args.data_root_dir is not None:
        data_root_dir = args.data_root_dir
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
    compute_norm_stats = (not bool(args.skip_norm_stats)) and bool(
        cache_cfg.get("compute_norm_stats", True)
    )

    torch.set_float32_matmul_precision("high")
    try:
        build_episode_embedding_cache(
            config,
            cache_dir=str(output_dir),
            device=device,
            batch_size=int(cache_cfg.get("batch_size", 256)),
            num_workers=int(cache_cfg.get("num_workers", 8)),
            use_bf16=bool(cache_cfg.get("use_bf16", True)),
            overwrite=overwrite,
            data_mixes=data_mixes,
            compute_norm_stats=compute_norm_stats,
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
