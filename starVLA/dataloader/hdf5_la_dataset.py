"""HDF5 VLA dataset variant that adds latent-action frame samples."""

from pathlib import Path

import numpy as np

import starVLA.dataloader.hdf5_dataset as hdf5_base
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP, EmbodimentTag
from starVLA.dataloader.hdf5_dataset import (
    HDF5SingleDataset,
    _cfg_get,
    _check_hdf5_action_type_matches_config,
    _mixture_entry_matches_include,
    _resolve_hdf5_dataset_path,
    collate_fn,
)
from starVLA.dataloader.lerobot_la_datasets import (
    _filter_latent_modalities,
    _resolve_latent_action_horizon,
    _resolve_latent_action_stride,
)
from starVLA.dataloader.episode_embedding_cache import EpisodeEmbeddingCache
from starVLA.dataloader.episode_soft_kl_cache import EpisodeSoftKLCache
from starVLA.dataloader.unit_la_cache_dataset import make_la_cache_traj_key


class LatentActionHDF5SingleDataset(HDF5SingleDataset):
    """Raw-HDF5 dataset with optional UniT/latent-action frame-window packing."""

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        data["_la_trajectory_id"] = trajectory_id
        data["_la_base_index"] = base_index
        return data

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)

        hist_cfg = _cfg_get(self.data_cfg, "history_frame", {})
        if _cfg_get(hist_cfg, "enabled", False):
            traj_id = data.get("_la_trajectory_id", None)
            bi = data.get("_la_base_index", None)
            if traj_id is not None and bi is not None:
                offset = int(_cfg_get(hist_cfg, "offset", 8))
                # Only prepend history when base_index can reach exactly -offset;
                # otherwise leave sample["image"] unchanged (no clamp/pad).
                if int(bi) >= offset:
                    video_keys_cfg = _cfg_get(hist_cfg, "video_keys", [])
                    obs_image_size = tuple(_cfg_get(self.data_cfg, "obs_image_size", (224, 224)))
                    history_images = []
                    for key in video_keys_cfg:
                        vk = f"video.{key}" if not str(key).startswith("video.") else str(key)
                        frames = self.get_video_frames_by_offsets(
                            int(traj_id), vk, int(bi), [-offset]
                        )
                        history_images.append(frames[0].resize(obs_image_size))
                    sample["image"] = history_images + sample["image"]

        la_cfg = _cfg_get(self.data_cfg, "latent_action", {})
        if not _cfg_get(la_cfg, "enabled", False):
            return sample

        trajectory_id = data.get("_la_trajectory_id", None)
        base_index = data.get("_la_base_index", None)
        if trajectory_id is None or base_index is None:
            raise ValueError("Latent-action dataset context is missing from sample data.")

        robot_type = sample.get("robot_type", None)
        robot_type = str(robot_type) if robot_type is not None else None
        action_horizon = _cfg_get(la_cfg, "horizon", None)
        if "action" in sample:
            action_horizon = int(sample["action"].shape[0])
        elif action_horizon is None:
            raise ValueError(
                "Latent-action samples without `action` require "
                "datasets.vla_data.latent_action.horizon to be set."
            )
        else:
            action_horizon = int(action_horizon)

        stride = _resolve_latent_action_stride(la_cfg, robot_type)
        la_horizon = _resolve_latent_action_horizon(la_cfg, robot_type, action_horizon)
        offsets = list(range(0, la_horizon + 1, stride))
        if _cfg_get(la_cfg, "include_terminal_frame", True) and offsets[-1] != la_horizon:
            offsets.append(la_horizon)

        trajectory_index = self.get_trajectory_index(int(trajectory_id))
        max_length = int(self.trajectory_lengths[trajectory_index])
        sample["la_padded"] = bool(int(base_index) + int(offsets[-1]) > max_length - 1)

        episode_cache_dir = _cfg_get(la_cfg, "episode_cache_dir", None)
        if episode_cache_dir is not None and str(episode_cache_dir) not in ("", "null", "None"):
            cached_distribution = self._load_episode_soft_kl_cache(
                str(episode_cache_dir), int(trajectory_id), int(base_index), offsets
            )
            if cached_distribution is not None:
                sample["la_cached_distribution"] = cached_distribution
                sample["la_frame_offsets"] = offsets
                window_count = len(offsets) - 1
                sample["la_frames"] = [None] * (window_count + 1)
                return sample

        embedding_cache_dir = _cfg_get(la_cfg, "embedding_cache_dir", None)
        if embedding_cache_dir is not None and str(embedding_cache_dir) not in ("", "null", "None"):
            cached_embedding = self._load_episode_embedding_cache(
                str(embedding_cache_dir), int(trajectory_id), int(base_index), offsets
            )
            if cached_embedding is not None:
                sample["la_cached_embedding"] = cached_embedding
                sample["la_frame_offsets"] = offsets
                window_count = len(offsets) - 1
                sample["la_frames"] = [None] * (window_count + 1)
                return sample

        cached_indices_path = _cfg_get(la_cfg, "cached_indices_path", None)
        if cached_indices_path is not None and str(cached_indices_path) not in ("", "null", "None"):
            cached_indices = self._load_cached_indices(
                str(cached_indices_path), int(trajectory_id), int(base_index), offsets
            )
            if cached_indices is not None:
                sample["la_cached_indices"] = cached_indices
                sample["la_frame_offsets"] = offsets
                window_count = len(offsets) - 1
                sample["la_frames"] = [None] * (window_count + 1)
                return sample

        video_keys_cfg = _cfg_get(la_cfg, "video_keys", None)
        if video_keys_cfg is not None and len(video_keys_cfg) > 0:
            video_keys = [f"video.{k}" if not str(k).startswith("video.") else str(k) for k in video_keys_cfg]
        else:
            single_key = _cfg_get(la_cfg, "video_key", None) or self.modality_keys["video"][0]
            if not str(single_key).startswith("video."):
                single_key = f"video.{single_key}"
            video_keys = [single_key]
        image_size = tuple(_cfg_get(la_cfg, "image_size", [224, 224]))
        multiview = len(video_keys) > 1

        full_window = bool(_cfg_get(la_cfg, "full_window", False))
        if full_window:
            sample["la_frames"] = self._pack_full_window_frames(
                int(trajectory_id), str(video_keys[0]), int(base_index), offsets, image_size
            )
        elif multiview:
            per_view_frames = [
                self.get_video_frames_by_offsets(int(trajectory_id), vk, int(base_index), offsets)
                for vk in video_keys
            ]
            sample["la_frames"] = [
                [per_view_frames[v][t].resize(image_size) for v in range(len(video_keys))]
                for t in range(len(offsets))
            ]
        else:
            sample["la_frames"] = [
                frame.resize(image_size)
                for frame in self.get_video_frames_by_offsets(
                    int(trajectory_id), str(video_keys[0]), int(base_index), offsets
                )
            ]
        sample["la_frame_offsets"] = offsets
        return sample

    def _load_episode_soft_kl_cache(
        self,
        cache_dir: str,
        trajectory_id: int,
        base_index: int,
        offsets: list[int],
    ):
        """Load soft_kl teacher weights ``[num_pairs * Q, C]`` from episode cache."""
        if not hasattr(self, "_episode_soft_kl_cache"):
            la_cfg = _cfg_get(self.data_cfg, "latent_action", {})
            fingerprint = _cfg_get(la_cfg, "cache_fingerprint", None)
            self._episode_soft_kl_cache = EpisodeSoftKLCache(
                cache_dir,
                expected_fingerprint=(
                    str(fingerprint) if fingerprint not in (None, "", "null") else None
                ),
            )
        num_pairs = max(len(offsets) - 1, 1)
        stride = int(offsets[1] - offsets[0]) if len(offsets) >= 2 else 1
        try:
            return self._episode_soft_kl_cache.read_window(
                self.dataset_name,
                int(trajectory_id),
                int(base_index),
                num_pairs=num_pairs,
                stride=stride,
            )
        except FileNotFoundError:
            return None

    def _load_episode_embedding_cache(
        self,
        cache_dir: str,
        trajectory_id: int,
        base_index: int,
        offsets: list[int],
    ):
        """Load continuous embeddings ``[num_pairs * Q, D]`` from episode cache."""
        if not hasattr(self, "_episode_embedding_cache"):
            la_cfg = _cfg_get(self.data_cfg, "latent_action", {})
            fingerprint = _cfg_get(la_cfg, "embedding_cache_fingerprint", None)
            self._episode_embedding_cache = EpisodeEmbeddingCache(
                cache_dir,
                expected_fingerprint=(
                    str(fingerprint) if fingerprint not in (None, "", "null") else None
                ),
            )
        num_pairs = max(len(offsets) - 1, 1)
        stride = int(offsets[1] - offsets[0]) if len(offsets) >= 2 else 1
        try:
            return self._episode_embedding_cache.read_window(
                self.dataset_name,
                int(trajectory_id),
                int(base_index),
                num_pairs=num_pairs,
                stride=stride,
            )
        except FileNotFoundError:
            return None

    def _load_cached_indices(
        self,
        cache_dir: str,
        trajectory_id: int,
        base_index: int,
        offsets: list[int],
    ) -> "np.ndarray | None":
        """Load pre-computed latent-action indices from cache directory (chunk format)."""
        cache_path = Path(cache_dir)
        if not hasattr(self, "_cache_manifest"):
            manifest_path = cache_path / "manifest.json"
            if not manifest_path.exists():
                return None
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._cache_manifest = json.load(f)
            self._cache_chunk_name: str | None = None
            self._cache_chunk_data: dict | None = None

        traj_key = make_la_cache_traj_key(self.dataset_name, trajectory_id)
        if traj_key not in self._cache_manifest:
            legacy_key = str(int(trajectory_id))
            if legacy_key in self._cache_manifest:
                traj_key = legacy_key
            else:
                return None

        chunk_name = self._cache_manifest[traj_key]
        if self._cache_chunk_name != chunk_name:
            chunk_path = cache_path / chunk_name
            if not chunk_path.exists():
                return None
            self._cache_chunk_data = np.load(chunk_path)
            self._cache_chunk_name = chunk_name

        if traj_key not in self._cache_chunk_data:
            return None
        all_indices = self._cache_chunk_data[traj_key]
        if base_index >= all_indices.shape[0]:
            return None
        return all_indices[base_index]

    def _pack_full_window_frames(
        self,
        trajectory_id: int,
        video_key: str,
        base_index: int,
        offsets: list[int],
        image_size: tuple[int, int],
    ) -> list[list[hdf5_base.Image.Image]]:
        clips = []
        for i in range(len(offsets) - 1):
            window_offsets = list(range(offsets[i], offsets[i + 1] + 1))
            frames = self.get_video_frames_by_offsets(trajectory_id, video_key, base_index, window_offsets)
            clips.append([frame.resize(image_size) for frame in frames])
        return clips

    def get_video_frames_by_offsets(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
        offsets: list[int],
    ) -> list[hdf5_base.Image.Image]:
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = int(self.trajectory_lengths[trajectory_index])
        step_indices = hdf5_base.np.asarray(offsets, dtype=hdf5_base.np.int64) + int(base_index)
        step_indices = hdf5_base.np.maximum(step_indices, 0)
        step_indices = hdf5_base.np.minimum(step_indices, max_length - 1)

        if not key.startswith("video."):
            raise ValueError(f"Video key must start with 'video.', got {key!r}.")
        subkey = key.replace("video.", "")
        original_key = self.lerobot_modality_meta.video[subkey].original_key or subkey
        path = self.get_episode_data_path(trajectory_id)

        frames = []
        with hdf5_base.h5py.File(path, "r") as f:
            image_ds = f[original_key]
            for idx in step_indices:
                frames.append(self._decode_image(image_ds[int(idx)]))
        return frames


def make_HDF5LatentActionSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    data_cfg=None,
) -> LatentActionHDF5SingleDataset:
    hdf5_action_type = str(_cfg_get(data_cfg, "hdf5_action_type", "qpos")).lower()
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    _check_hdf5_action_type_matches_config(hdf5_action_type, robot_type, data_config)

    modality_config = dict(data_config.modality_config())
    transforms = data_config.transform()
    modality_config, transforms = _filter_latent_modalities(modality_config, transforms, data_cfg)

    embodiment_tag = getattr(data_config, "embodiment_tag", EmbodimentTag.NEW_EMBODIMENT)
    dataset_path = _resolve_hdf5_dataset_path(Path(data_root_dir), data_name)
    return LatentActionHDF5SingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        data_cfg=data_cfg,
        dataset_name=data_name,
        robot_type=robot_type,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    include_robot_types: list[str] | set[str] | tuple[str, ...] | None = None,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    include_robot_types = {str(robot_type) for robot_type in include_robot_types} if include_robot_types else None

    dataset_mixture = []
    included_datasets = set()
    for d_name, d_weight, robot_type in mixture_spec:
        if not _mixture_entry_matches_include(d_name, robot_type, include_robot_types):
            continue
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        included_datasets.add(dataset_key)
        dataset_mixture.append(
            (
                make_HDF5LatentActionSingleDataset(data_root_dir, d_name, robot_type, data_cfg=data_cfg),
                d_weight,
            )
        )

    if include_robot_types is not None and not dataset_mixture:
        raise ValueError(
            f"No datasets in data_mix={data_mix!r} match include_robot_types={sorted(include_robot_types)}."
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
