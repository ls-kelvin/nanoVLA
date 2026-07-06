"""Dataset for parallel UniT latent-action index caching.

Each sample is one (obs, goal) frame pair with metadata for cross-episode batching.
Workers decode HDF5 frames and convert them to ImageNet-normalized tensors in parallel.
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

from starVLA.model.modules.latent_action.unit_encoder import UniTVisualLatentActionEncoder

_H5_FILE_CACHE: dict[tuple[int, int], h5py.File] = {}


def make_la_cache_traj_key(dataset_name: str, trajectory_id: int) -> str:
    """Globally unique manifest key: ``{dataset_name}/{episode_index}``."""
    return f"{dataset_name}/{int(trajectory_id)}"


def unit_la_cache_worker_init(_worker_id: int) -> None:
    """Reset per-worker HDF5 handle cache."""
    global _H5_FILE_CACHE
    _H5_FILE_CACHE = {}


@dataclass(frozen=True)
class PairEntry:
    traj_key: str
    traj_id: int
    ds_idx: int
    ep_num_steps: int
    num_pairs: int
    obs_abs: int
    goal_abs: int
    total_unique: int
    offsets: tuple[int, ...]


def build_pair_entries(
    single_dataset,
    ds_idx: int,
    trajectory_ids: list[int],
    offsets: list[int],
) -> list[PairEntry]:
    """Build unique (obs_abs, goal_abs) frame pairs across trajectories.

    With overlapping offsets (e.g. stride=16, horizon=32 → offsets [0,16,32]),
    many pairs are shared across different (step, pair_idx) slots.  Only unique
    pairs are emitted; the EpisodeAssembler reconstructs per-step indices from
    them using the stored offsets.
    """
    num_pairs = len(offsets) - 1
    offsets_tuple = tuple(offsets)
    entries: list[PairEntry] = []
    for traj_id in trajectory_ids:
        trajectory_index = single_dataset.get_trajectory_index(int(traj_id))
        ep_len = int(single_dataset.trajectory_lengths[trajectory_index])
        traj_key = make_la_cache_traj_key(single_dataset.dataset_name, traj_id)
        seen: set[tuple[int, int]] = set()
        traj_pairs: list[tuple[int, int]] = []
        for step_idx in range(ep_len):
            for pair_idx in range(num_pairs):
                obs_abs = min(max(step_idx + offsets[pair_idx], 0), ep_len - 1)
                goal_abs = min(max(step_idx + offsets[pair_idx + 1], 0), ep_len - 1)
                key = (obs_abs, goal_abs)
                if key not in seen:
                    seen.add(key)
                    traj_pairs.append(key)
        total_unique = len(traj_pairs)
        for obs_abs, goal_abs in traj_pairs:
            entries.append(
                PairEntry(
                    traj_key=traj_key,
                    traj_id=int(traj_id),
                    ds_idx=ds_idx,
                    ep_num_steps=ep_len,
                    num_pairs=num_pairs,
                    obs_abs=obs_abs,
                    goal_abs=goal_abs,
                    total_unique=total_unique,
                    offsets=offsets_tuple,
                )
            )
    return entries


def _get_h5_file(single_dataset, ds_idx: int, traj_id: int) -> h5py.File:
    cache_key = (ds_idx, traj_id)
    if cache_key not in _H5_FILE_CACHE:
        path = single_dataset.get_episode_data_path(traj_id)
        _H5_FILE_CACHE[cache_key] = h5py.File(path, "r")
    return _H5_FILE_CACHE[cache_key]


def _read_frame(
    single_dataset,
    ds_idx: int,
    traj_id: int,
    video_key: str,
    frame_idx: int,
    image_size: tuple[int, int],
) -> Image.Image:
    h5_file = _get_h5_file(single_dataset, ds_idx, traj_id)
    subkey = video_key.replace("video.", "")
    original_key = single_dataset.lerobot_modality_meta.video[subkey].original_key or subkey
    raw = h5_file[original_key][frame_idx]
    image = single_dataset._decode_image(raw)
    return image.resize(image_size, Image.BILINEAR)


class UnitLatentActionCacheDataset(Dataset):
    """One sample = one frame pair with CPU-side ImageNet tensor preprocessing."""

    def __init__(
        self,
        single_datasets: list[Any],
        entries: list[PairEntry],
        video_keys: list[str],
        image_size: tuple[int, int],
        multiview: bool,
    ) -> None:
        self.single_datasets = single_datasets
        self.entries = entries
        self.video_keys = video_keys
        self.image_size = image_size
        self.multiview = multiview

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        single_ds = self.single_datasets[entry.ds_idx]

        if self.multiview:
            obs_views = [
                _read_frame(single_ds, entry.ds_idx, entry.traj_id, vk, entry.obs_abs, self.image_size)
                for vk in self.video_keys
            ]
            goal_views = [
                _read_frame(single_ds, entry.ds_idx, entry.traj_id, vk, entry.goal_abs, self.image_size)
                for vk in self.video_keys
            ]
            obs = torch.stack(
                [UniTVisualLatentActionEncoder.imagenet_tensor_from_pil(img, self.image_size) for img in obs_views],
                dim=0,
            )
            goal = torch.stack(
                [UniTVisualLatentActionEncoder.imagenet_tensor_from_pil(img, self.image_size) for img in goal_views],
                dim=0,
            )
        else:
            obs_img = _read_frame(
                single_ds, entry.ds_idx, entry.traj_id, self.video_keys[0], entry.obs_abs, self.image_size
            )
            goal_img = _read_frame(
                single_ds, entry.ds_idx, entry.traj_id, self.video_keys[0], entry.goal_abs, self.image_size
            )
            obs = UniTVisualLatentActionEncoder.imagenet_tensor_from_pil(obs_img, self.image_size)
            goal = UniTVisualLatentActionEncoder.imagenet_tensor_from_pil(goal_img, self.image_size)

        return {
            "obs": obs,
            "goal": goal,
            "traj_key": entry.traj_key,
            "obs_abs": entry.obs_abs,
            "goal_abs": entry.goal_abs,
            "num_pairs": entry.num_pairs,
            "ep_num_steps": entry.ep_num_steps,
            "total_unique": entry.total_unique,
            "offsets": entry.offsets,
        }


def collate_unit_la_pairs(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    obs = torch.stack([item["obs"] for item in batch], dim=0)
    goal = torch.stack([item["goal"] for item in batch], dim=0)
    if obs.ndim == 4 and obs.shape[1] == 3:
        obs = obs.unsqueeze(1)
        goal = goal.unsqueeze(1)
    meta = [
        {
            "traj_key": item["traj_key"],
            "obs_abs": item["obs_abs"],
            "goal_abs": item["goal_abs"],
            "num_pairs": item["num_pairs"],
            "ep_num_steps": item["ep_num_steps"],
            "total_unique": item["total_unique"],
            "offsets": item["offsets"],
        }
        for item in batch
    ]
    return obs, goal, meta


def make_pair_dataloader(
    dataset: UnitLatentActionCacheDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_unit_la_pairs,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["worker_init_fn"] = unit_la_cache_worker_init
    return DataLoader(dataset, **loader_kwargs)


class EpisodeAssembler:
    """Reassemble unique pair indices back into per-episode arrays.

    Each unique (obs_abs, goal_abs) pair is encoded once.  When all unique
    pairs for a trajectory have been collected, the assembler reconstructs
    per-step indices by looking up the correct pair for every (step, pair_idx)
    using the stored offsets.
    """

    def __init__(self) -> None:
        self._pairs: dict[str, dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
        self._meta: dict[str, tuple[int, int, int, tuple[int, ...]]] = {}

    def add(
        self,
        traj_key: str,
        obs_abs: int,
        goal_abs: int,
        num_pairs: int,
        ep_num_steps: int,
        total_unique: int,
        offsets: tuple[int, ...],
        pair_indices: np.ndarray,
    ) -> tuple[str, np.ndarray] | None:
        if traj_key not in self._meta:
            self._meta[traj_key] = (ep_num_steps, num_pairs, total_unique, offsets)
        self._pairs[traj_key][(obs_abs, goal_abs)] = pair_indices

        if len(self._pairs[traj_key]) >= self._meta[traj_key][2]:
            return self._finalize(traj_key)
        return None

    def _finalize(self, traj_key: str) -> tuple[str, np.ndarray]:
        ep_num_steps, num_pairs, _, offsets = self._meta.pop(traj_key)
        pair_map = self._pairs.pop(traj_key)
        all_indices = []
        for step_idx in range(ep_num_steps):
            step_pairs = []
            for pair_idx in range(num_pairs):
                oa = min(max(step_idx + offsets[pair_idx], 0), ep_num_steps - 1)
                ga = min(max(step_idx + offsets[pair_idx + 1], 0), ep_num_steps - 1)
                step_pairs.append(pair_map[(oa, ga)])
            step_indices = np.stack(step_pairs, axis=0)
            step_flat = step_indices.reshape(num_pairs * step_indices.shape[1], step_indices.shape[2])
            all_indices.append(step_flat)
        return traj_key, np.stack(all_indices, axis=0)
