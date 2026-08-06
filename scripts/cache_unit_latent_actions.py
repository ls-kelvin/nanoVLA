#!/usr/bin/env python3
"""Pre-compute UniT latent-action token indices and save to disk.

This script iterates over all trajectories in an HDF5 dataset, encodes
frame pairs using the UniT tokenizer, and saves the resulting VQ indices
in chunk files.  During VLA training, setting
``datasets.vla_data.latent_action.cached_indices_path`` to the output
directory skips the on-the-fly encoder forward pass entirely.

Output layout:
    <output_dir>/
        manifest.json          # {dataset_name/episode_index -> chunk filename}
        metadata.json          # config info for validation
        chunk_0000.npz         # contains multiple episodes keyed by traj_id
        chunk_0001.npz
        ...

Usage (single GPU):
    python scripts/cache_unit_latent_actions.py \
        --config-yaml scripts/hdf5_aloha_clean_random_la_unit.yaml \
        --output-dir /path/to/cache_dir \
        --batch-size 128 \
        --num-workers 8

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 scripts/cache_unit_latent_actions.py \
        --config-yaml scripts/hdf5_aloha_clean_random_la_unit.yaml \
        --output-dir /path/to/cache_dir \
        --batch-size 128 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-yaml", required=True, type=Path, help="Training YAML config path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output cache directory.")
    parser.add_argument("--batch-size", default=128, type=int, help="Encoding batch size per GPU.")
    parser.add_argument("--num-workers", default=8, type=int, help="DataLoader workers for parallel preprocessing.")
    parser.add_argument("--chunk-size", default=500, type=int, help="Number of episodes per chunk file.")
    parser.add_argument("--device", default="auto", help="Torch device (default: auto).")
    parser.add_argument(
        "--data-mix", default=None,
        help="Override data mix to process (default: use latent_data_mix from config).",
    )
    parser.add_argument(
        "--data-root-dir", default=None, type=Path,
        help="Override datasets.vla_data.data_root_dir.",
    )
    return parser.parse_args()


def resolve_device(device: str, local_rank: int) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    return torch.device(device)


def load_config(yaml_path: Path) -> "OmegaConf":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(yaml_path)
    return cfg


def build_dataset(data_cfg, data_mix: str):
    """Build the HDF5 latent-action dataset for a given mixture."""
    from omegaconf import OmegaConf

    cfg_copy = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    cfg_copy.data_mix = data_mix
    from starVLA.dataloader.hdf5_la_dataset import get_vla_dataset

    return get_vla_dataset(
        data_cfg=cfg_copy,
        mode="train",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=42,
    )


def build_encoder(config):
    from starVLA.model.modules.latent_action import build_latent_action_encoder

    encoder = build_latent_action_encoder(config)
    encoder.eval()
    return encoder


def get_la_config(full_cfg):
    """Extract latent action config from dataset section."""
    return full_cfg.datasets.vla_data.get("latent_action", {})


def resolve_video_keys(la_cfg, dataset) -> list[str]:
    """Resolve video keys list from config, with fallback."""
    from starVLA.dataloader.hdf5_dataset import _cfg_get

    video_keys_cfg = _cfg_get(la_cfg, "video_keys", None)
    if video_keys_cfg is not None and len(video_keys_cfg) > 0:
        return [f"video.{k}" if not str(k).startswith("video.") else str(k) for k in video_keys_cfg]
    single_key = _cfg_get(la_cfg, "video_key", None)
    if single_key:
        if not str(single_key).startswith("video."):
            single_key = f"video.{single_key}"
        return [single_key]
    return [dataset.datasets[0].modality_keys["video"][0]]


def compute_offsets(la_cfg, action_horizon: int, robot_type: str | None) -> list[int]:
    from starVLA.dataloader.hdf5_dataset import _cfg_get
    from starVLA.dataloader.lerobot_la_datasets import _resolve_latent_action_horizon, _resolve_latent_action_stride

    stride = _resolve_latent_action_stride(la_cfg, robot_type)
    la_horizon = _resolve_latent_action_horizon(la_cfg, robot_type, action_horizon)
    offsets = list(range(0, la_horizon + 1, stride))
    include_terminal = _cfg_get(la_cfg, "include_terminal_frame", True)
    if include_terminal and offsets[-1] != la_horizon:
        offsets.append(la_horizon)
    return offsets


def save_chunk(chunk_data: dict[str, np.ndarray], chunk_path: Path) -> None:
    """Save a chunk of episodes as a single compressed npz file."""
    np.savez_compressed(chunk_path, **chunk_data)


def store_episode(
    traj_key: str,
    indices: np.ndarray,
    chunk_data: dict[str, np.ndarray],
    manifest: dict[str, str],
    output_dir: Path,
    local_rank: int,
    chunk_idx: int,
    chunk_size: int,
    episodes_in_chunk: int,
    total_encoded: int,
) -> tuple[dict[str, np.ndarray], int, int, int]:
    chunk_data[traj_key] = indices
    episodes_in_chunk += 1
    total_encoded += 1

    if episodes_in_chunk >= chunk_size:
        chunk_name = f"chunk_{local_rank:02d}_{chunk_idx:04d}.npz"
        save_chunk(chunk_data, output_dir / chunk_name)
        for key in chunk_data:
            manifest[key] = chunk_name
        return {}, 0, chunk_idx + 1, total_encoded
    return chunk_data, episodes_in_chunk, chunk_idx, total_encoded


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()

    device = resolve_device(args.device, local_rank)
    torch.set_float32_matmul_precision("high")
    config = load_config(args.config_yaml)
    data_cfg = config.datasets.vla_data
    la_cfg = get_la_config(config)

    if args.data_root_dir is not None:
        data_cfg.data_root_dir = str(args.data_root_dir)

    data_mixes = []
    if args.data_mix:
        data_mixes.append(args.data_mix)
    else:
        latent_mix = data_cfg.get("latent_data_mix", None)
        if latent_mix:
            data_mixes.append(str(latent_mix))
        else:
            data_mixes.append(str(data_cfg.data_mix))

    from omegaconf import OmegaConf
    from starVLA.dataloader.hdf5_dataset import _cfg_get
    from starVLA.dataloader.unit_la_cache_dataset import (
        EpisodeAssembler,
        UnitLatentActionCacheDataset,
        build_pair_entries,
        make_pair_dataloader,
    )

    framework_cfg = OmegaConf.create({
        "framework": OmegaConf.to_container(config.framework, resolve=True)
    })
    encoder = build_encoder(framework_cfg)
    encoder.to(device)

    image_size = tuple(_cfg_get(la_cfg, "image_size", [224, 224]))
    action_horizon = int(_cfg_get(la_cfg, "horizon", 32))
    chunk_size = args.chunk_size

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    chunk_data: dict[str, np.ndarray] = {}
    chunk_idx = 0
    episodes_in_chunk = 0
    total_encoded = 0

    all_single_datasets = []
    all_entries = []
    video_keys: list[str] | None = None
    total_episodes = 0

    for mix_name in data_mixes:
        dataset = build_dataset(data_cfg, mix_name)
        mix_video_keys = resolve_video_keys(la_cfg, dataset)
        if video_keys is None:
            video_keys = mix_video_keys
        elif video_keys != mix_video_keys:
            raise ValueError(f"video_keys mismatch across mixes: {video_keys} vs {mix_video_keys}")

        for ds_idx, single_ds in enumerate(dataset.datasets):
            global_ds_idx = len(all_single_datasets)
            all_single_datasets.append(single_ds)

            robot_type = single_ds.lerobot_info_meta.get("robot_type", None)
            robot_type = str(robot_type) if robot_type is not None else None
            offsets = compute_offsets(la_cfg, action_horizon, robot_type)

            traj_ids = list(single_ds.trajectory_ids)
            my_traj_ids = traj_ids[local_rank::world_size] if is_distributed else traj_ids
            total_episodes += len(my_traj_ids)
            all_entries.extend(build_pair_entries(single_ds, global_ds_idx, my_traj_ids, offsets))

    assert video_keys is not None
    multiview = len(video_keys) > 1
    pair_dataset = UnitLatentActionCacheDataset(
        single_datasets=all_single_datasets,
        entries=all_entries,
        video_keys=video_keys,
        image_size=image_size,
        multiview=multiview,
    )
    pair_loader = make_pair_dataloader(
        dataset=pair_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if local_rank == 0 and all_entries:
        traj_meta: dict[str, tuple[int, int]] = {}
        for e in all_entries:
            if e.traj_key not in traj_meta:
                traj_meta[e.traj_key] = (e.ep_num_steps, e.num_pairs)
        naive_total = sum(ep * n_pairs for ep, n_pairs in traj_meta.values())
        unique_total = len(all_entries)
        print(f"Unique encoding pairs: {unique_total} | Naive (step×num_pairs): {naive_total} | "
              f"Saved: {naive_total - unique_total} ({100*(1-unique_total/max(naive_total,1)):.1f}%)")

    assembler = EpisodeAssembler()
    progress = tqdm(
        total=total_episodes,
        desc="Caching latent actions",
        unit="ep",
        disable=local_rank != 0,
    )

    for obs, goal, meta in pair_loader:
        obs = obs.to(device=device, dtype=encoder.dtype, non_blocking=device.type == "cuda")
        goal = goal.to(device=device, dtype=encoder.dtype, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            indices = encoder.encode_tensors(obs, goal).detach().cpu().numpy()

        for row_idx, item in enumerate(meta):
            finalized = assembler.add(
                traj_key=item["traj_key"],
                obs_abs=item["obs_abs"],
                goal_abs=item["goal_abs"],
                num_pairs=item["num_pairs"],
                ep_num_steps=item["ep_num_steps"],
                total_unique=item["total_unique"],
                offsets=item["offsets"],
                pair_indices=indices[row_idx],
            )
            if finalized is None:
                continue

            traj_key, episode_indices = finalized
            chunk_data, episodes_in_chunk, chunk_idx, total_encoded = store_episode(
                traj_key=traj_key,
                indices=episode_indices,
                chunk_data=chunk_data,
                manifest=manifest,
                output_dir=output_dir,
                local_rank=local_rank,
                chunk_idx=chunk_idx,
                chunk_size=chunk_size,
                episodes_in_chunk=episodes_in_chunk,
                total_encoded=total_encoded,
            )
            progress.update(1)

    if chunk_data:
        chunk_name = f"chunk_{local_rank:02d}_{chunk_idx:04d}.npz"
        save_chunk(chunk_data, output_dir / chunk_name)
        for key in chunk_data:
            manifest[key] = chunk_name

    progress.close()

    manifest_path = output_dir / f"manifest_rank{local_rank:02d}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    if is_distributed:
        dist.barrier()

    if local_rank == 0:
        merged_manifest: dict[str, str] = {}
        for rank in range(world_size):
            rank_manifest_path = output_dir / f"manifest_rank{rank:02d}.json"
            if rank_manifest_path.exists():
                with open(rank_manifest_path, "r", encoding="utf-8") as f:
                    merged_manifest.update(json.load(f))
                rank_manifest_path.unlink()

        merged_manifest_path = output_dir / "manifest.json"
        with open(merged_manifest_path, "w", encoding="utf-8") as f:
            json.dump(merged_manifest, f, ensure_ascii=False)

        video_keys_cfg = _cfg_get(la_cfg, "video_keys", None)
        if video_keys_cfg is not None and len(video_keys_cfg) > 0:
            meta_video_keys = [str(k) for k in video_keys_cfg]
        else:
            meta_video_keys = [str(_cfg_get(la_cfg, "video_key", "cam_high"))]
        metadata = {
            "config_yaml": str(args.config_yaml),
            "video_keys": meta_video_keys,
            "image_size": list(image_size),
            "action_horizon": action_horizon,
            "stride": int(_cfg_get(la_cfg, "stride", 16)),
            "include_terminal_frame": bool(_cfg_get(la_cfg, "include_terminal_frame", True)),
            "tokenizer_path": str(_cfg_get(config.framework.latent_action, "groot_tokenizer_path", "")),
            "data_mixes": data_mixes,
            "chunk_size": chunk_size,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
