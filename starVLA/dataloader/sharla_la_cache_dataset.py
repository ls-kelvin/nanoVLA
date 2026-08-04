"""Pair-level DataLoader helpers for Sharla soft_kl episode-cache building.

Each sample is one stride window for a single episode step: endpoints plus the
dense mid frames. Workers decode HDF5 frames and emit ``[0, 1]`` CHW float
tensors so the GPU encode loop stays busy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_H5_FILE_CACHE: dict[tuple[int, int], h5py.File] = {}


def make_la_cache_traj_key(dataset_name: str, trajectory_id: int) -> str:
    """Globally unique episode key: ``{dataset_name}/{episode_index}``."""
    return f"{dataset_name}/{int(trajectory_id)}"


def sharla_la_cache_worker_init(_worker_id: int) -> None:
    """Reset per-worker HDF5 handle cache."""
    global _H5_FILE_CACHE
    _H5_FILE_CACHE = {}


@dataclass(frozen=True)
class SharlaPairEntry:
    traj_key: str
    dataset_name: str
    traj_id: int
    ds_idx: int
    step: int
    obs_abs: int
    goal_abs: int
    mid_abs: tuple[int, ...]
    ep_num_steps: int
    stride: int


def build_sharla_pair_entries(
    single_dataset,
    ds_idx: int,
    trajectory_ids: list[int],
    *,
    stride: int,
) -> list[SharlaPairEntry]:
    """Emit one full-stride window per episode step.

    Endpoints are ``(t, min(t + stride, T - 1))``; mid frames are every index
    between them. Past the episode end, indices clamp to the last frame so the
    window always has ``stride - 1`` mid frames.
    """
    entries: list[SharlaPairEntry] = []
    stride = int(stride)
    for traj_id in trajectory_ids:
        trajectory_index = single_dataset.get_trajectory_index(int(traj_id))
        ep_len = int(single_dataset.trajectory_lengths[trajectory_index])
        traj_key = make_la_cache_traj_key(single_dataset.dataset_name, traj_id)
        for step in range(ep_len):
            obs_abs = step
            goal_abs = min(step + stride, ep_len - 1)
            mid_abs = tuple(min(step + offset, ep_len - 1) for offset in range(1, stride))
            entries.append(
                SharlaPairEntry(
                    traj_key=traj_key,
                    dataset_name=str(single_dataset.dataset_name),
                    traj_id=int(traj_id),
                    ds_idx=int(ds_idx),
                    step=int(step),
                    obs_abs=int(obs_abs),
                    goal_abs=int(goal_abs),
                    mid_abs=mid_abs,
                    ep_num_steps=ep_len,
                    stride=stride,
                )
            )
    return entries


def _get_h5_file(single_dataset, ds_idx: int, traj_id: int) -> h5py.File:
    cache_key = (ds_idx, traj_id)
    if cache_key not in _H5_FILE_CACHE:
        path = single_dataset.get_episode_data_path(traj_id)
        _H5_FILE_CACHE[cache_key] = h5py.File(path, "r")
    return _H5_FILE_CACHE[cache_key]


def _read_frame_tensor(
    single_dataset,
    ds_idx: int,
    traj_id: int,
    video_key: str,
    frame_idx: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Decode one frame to CHW float tensor in ``[0, 1]`` (Sharla preprocess)."""
    h5_file = _get_h5_file(single_dataset, ds_idx, traj_id)
    subkey = video_key.replace("video.", "")
    original_key = single_dataset.lerobot_modality_meta.video[subkey].original_key or subkey
    raw = h5_file[original_key][frame_idx]
    image = single_dataset._decode_image(raw)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image from HDF5 decode, got {type(image)!r}")
    image = image.convert("RGB").resize(image_size, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class SharlaLatentActionCacheDataset(Dataset):
    """One sample = one full-stride frame window with CPU-side Sharla preprocessing."""

    def __init__(
        self,
        single_datasets: list[Any],
        entries: list[SharlaPairEntry],
        video_key: str,
        image_size: tuple[int, int],
    ) -> None:
        self.single_datasets = single_datasets
        self.entries = entries
        self.video_key = video_key
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        single_ds = self.single_datasets[entry.ds_idx]
        obs = _read_frame_tensor(
            single_ds,
            entry.ds_idx,
            entry.traj_id,
            self.video_key,
            entry.obs_abs,
            self.image_size,
        )
        goal = _read_frame_tensor(
            single_ds,
            entry.ds_idx,
            entry.traj_id,
            self.video_key,
            entry.goal_abs,
            self.image_size,
        )
        mid_frames = [
            _read_frame_tensor(
                single_ds,
                entry.ds_idx,
                entry.traj_id,
                self.video_key,
                mid_abs,
                self.image_size,
            )
            for mid_abs in entry.mid_abs
        ]
        fmid = (
            torch.stack(mid_frames, dim=0)
            if mid_frames
            else torch.empty((0, 3, *self.image_size), dtype=torch.float32)
        )
        return {
            "f0": obs,
            "f1": goal,
            "fmid": fmid,
            "traj_key": entry.traj_key,
            "dataset_name": entry.dataset_name,
            "traj_id": entry.traj_id,
            "step": entry.step,
            "ep_num_steps": entry.ep_num_steps,
            "stride": entry.stride,
        }


def collate_sharla_la_pairs(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    f0 = torch.stack([item["f0"] for item in batch], dim=0)
    f1 = torch.stack([item["f1"] for item in batch], dim=0)
    fmid = torch.stack([item["fmid"] for item in batch], dim=0)
    meta = [
        {
            "traj_key": item["traj_key"],
            "dataset_name": item["dataset_name"],
            "traj_id": item["traj_id"],
            "step": item["step"],
            "ep_num_steps": item["ep_num_steps"],
            "stride": item["stride"],
        }
        for item in batch
    ]
    return f0, f1, fmid, meta


def make_sharla_pair_dataloader(
    dataset: SharlaLatentActionCacheDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_sharla_la_pairs,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = sharla_la_cache_worker_init
    return DataLoader(dataset, **loader_kwargs)


class SoftDistributionEpisodeAssembler:
    """Collect per-step soft codebook weights and emit full episode tensors."""

    def __init__(self) -> None:
        self._steps: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        self._meta: dict[str, dict[str, Any]] = {}

    def pending_traj_keys(self) -> list[str]:
        return sorted(self._steps.keys())

    def add(
        self,
        traj_key: str,
        step: int,
        ep_num_steps: int,
        distribution: torch.Tensor,
        *,
        dataset_name: str,
        traj_id: int,
        stride: int,
    ) -> tuple[str, torch.Tensor, dict[str, Any]] | None:
        if traj_key not in self._meta:
            self._meta[traj_key] = {
                "dataset_name": dataset_name,
                "traj_id": int(traj_id),
                "stride": int(stride),
                "ep_num_steps": int(ep_num_steps),
            }
        self._steps[traj_key][int(step)] = distribution.detach().cpu().float()
        if len(self._steps[traj_key]) >= int(ep_num_steps):
            return self._finalize(traj_key)
        return None

    def _finalize(self, traj_key: str) -> tuple[str, torch.Tensor, dict[str, Any]]:
        meta = self._meta.pop(traj_key)
        step_map = self._steps.pop(traj_key)
        ep_num_steps = int(meta["ep_num_steps"])
        missing = [step for step in range(ep_num_steps) if step not in step_map]
        if missing:
            raise RuntimeError(
                f"Incomplete episode assembly for {traj_key}: missing steps {missing[:8]}..."
            )
        values = torch.stack([step_map[step] for step in range(ep_num_steps)], dim=0)
        return traj_key, values, meta
