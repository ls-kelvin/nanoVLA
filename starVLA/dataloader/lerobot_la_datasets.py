"""LeRobot VLA dataset variant that adds latent-action frame samples.

This module intentionally leaves ``lerobot_datasets.py`` unchanged.  It reuses
the same registry, transforms, mixture sampler, and statistics code, while
packing an additional ``la_frames`` field for frameworks such as
``QwenPI_v3_LA``.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


def collate_fn(batch):
    return batch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


class LatentActionLeRobotSingleDataset(LeRobotSingleDataset):
    """Single LeRobot dataset with optional LA frame packing."""

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        data["_la_trajectory_id"] = trajectory_id
        data["_la_base_index"] = base_index
        return data

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        la_cfg = _cfg_get(self.data_cfg, "latent_action", {})
        if not _cfg_get(la_cfg, "enabled", False):
            return sample

        trajectory_id = data.get("_la_trajectory_id", None)
        base_index = data.get("_la_base_index", None)
        if trajectory_id is None or base_index is None:
            raise ValueError("Latent-action dataset context is missing from sample data.")

        action_horizon = int(sample["action"].shape[0])
        stride = int(_cfg_get(la_cfg, "stride", 4))
        if stride <= 0:
            raise ValueError(f"latent_action.stride must be positive, got {stride}.")

        offsets = list(range(0, action_horizon, stride))
        if _cfg_get(la_cfg, "include_terminal_frame", True) and offsets[-1] != action_horizon - 1:
            offsets.append(action_horizon - 1)

        video_key = _cfg_get(la_cfg, "video_key", None) or self.modality_keys["video"][0]
        if not str(video_key).startswith("video."):
            video_key = f"video.{video_key}"
        image_size = tuple(_cfg_get(la_cfg, "image_size", [224, 224]))
        sample["la_frames"] = [
            frame.resize(image_size)
            for frame in self.get_video_frames_by_offsets(int(trajectory_id), str(video_key), int(base_index), offsets)
        ]
        sample["la_frame_offsets"] = offsets
        return sample

    def get_video_frames_by_offsets(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
        offsets: list[int],
    ) -> list[Image.Image]:
        """Read arbitrary video frames relative to ``base_index``."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = int(self.trajectory_lengths[trajectory_index])
        step_indices = np.asarray(offsets, dtype=np.int64) + int(base_index)
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)

        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        subkey = key.replace("video.", "")
        original_key = self.lerobot_modality_meta.video[subkey].original_key
        if original_key is None:
            original_key = subkey

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        if original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()
            return [_as_pil(self._decode_image_entry(image_entries[int(idx)])) for idx in step_indices]

        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        video_timestamp = self.curr_traj_data["timestamp"].to_numpy()[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            video_timestamp = video_timestamp + float(from_timestamps.get(original_key, 0.0))

        frames = get_frames_by_timestamps(
            self.get_video_path(trajectory_id, subkey).as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        return [_as_pil(frame) for frame in frames]

    def _decode_image_entry(self, entry):
        if isinstance(entry, np.ndarray):
            return entry
        if isinstance(entry, Image.Image):
            return np.array(entry.convert("RGB"))
        if isinstance(entry, dict):
            img_bytes = entry.get("bytes", None)
            img_path = entry.get("path", None)
            if img_bytes is not None:
                return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            if img_path is not None:
                path_obj = Path(img_path)
                if not path_obj.is_absolute():
                    path_obj = self.dataset_path / path_obj
                return np.array(Image.open(path_obj).convert("RGB"))
        raise TypeError(f"Unsupported image entry type: {type(entry)}")


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LatentActionLeRobotSingleDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = Path(data_root_dir) / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    return LatentActionLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = [
        (
            make_LeRobotSingleDataset(
                Path(data_root_dir),
                d_name,
                robot_type,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            ),
            d_weight,
        )
        for d_name, d_weight, robot_type in filtered_mixture_spec
    ]

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
