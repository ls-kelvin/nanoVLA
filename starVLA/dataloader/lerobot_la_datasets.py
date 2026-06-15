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
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


def collate_fn(batch):
    return batch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _mixture_entry_matches_include(d_name: str, robot_type: str, include_patterns: set[str] | None) -> bool:
    if include_patterns is None:
        return True
    match_text = "\n".join([str(d_name), str(robot_type)])
    return any(pattern in match_text for pattern in include_patterns)


def _modality_enabled(la_cfg, modality: str) -> bool:
    return bool(_cfg_get(la_cfg, f"load_{modality}", True))


def _key_is_disabled(key: str, disabled_modalities: set[str]) -> bool:
    return any(str(key).startswith(f"{modality}.") for modality in disabled_modalities)


def _filter_keyed_dict(value, disabled_modalities: set[str]):
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if not _key_is_disabled(key, disabled_modalities)}


def _filter_transform_modalities(transform, disabled_modalities: set[str]):
    if isinstance(transform, ComposedModalityTransform):
        filtered_transforms = []
        for child in transform.transforms:
            child = _filter_transform_modalities(child, disabled_modalities)
            if child is not None:
                filtered_transforms.append(child)
        transform.transforms = filtered_transforms
        return transform

    for modality in disabled_modalities:
        concat_attr = f"{modality}_concat_order"
        if hasattr(transform, concat_attr):
            setattr(transform, concat_attr, None)
        dims_attr = f"{modality}_dims"
        if hasattr(transform, dims_attr):
            setattr(transform, dims_attr, _filter_keyed_dict(getattr(transform, dims_attr), disabled_modalities))

    for attr in (
        "normalization_modes",
        "normalization_statistics",
        "target_rotations",
        "modality_metadata",
        "input_dtypes",
        "output_dtypes",
    ):
        if hasattr(transform, attr):
            setattr(transform, attr, _filter_keyed_dict(getattr(transform, attr), disabled_modalities))

    apply_to = list(getattr(transform, "apply_to", []) or [])
    if not apply_to:
        return transform

    remaining_apply_to = [key for key in apply_to if not _key_is_disabled(key, disabled_modalities)]
    transform.apply_to = remaining_apply_to
    if not remaining_apply_to and len(remaining_apply_to) != len(apply_to):
        return None
    return transform


def _filter_latent_modalities(modality_config: dict, transforms, data_cfg):
    la_cfg = _cfg_get(data_cfg, "latent_action", {}) or {}
    disabled_modalities = {
        modality
        for modality in ("action", "state")
        if not _modality_enabled(la_cfg, modality)
    }
    if not disabled_modalities:
        return modality_config, transforms

    modality_config = dict(modality_config)
    for modality in disabled_modalities:
        modality_config.pop(modality, None)
    transforms = _filter_transform_modalities(transforms, disabled_modalities)
    return modality_config, transforms


def _resolve_latent_action_stride(la_cfg, robot_type: str | None) -> int:
    stride = int(_cfg_get(la_cfg, "stride", 4))
    if robot_type:
        stride_overrides = _cfg_get(la_cfg, "stride_overrides", None)
        robot_stride = _cfg_get(stride_overrides, robot_type, None)
        if robot_stride is not None:
            stride = int(robot_stride)

        horizon_overrides = _cfg_get(la_cfg, "horizon_overrides", None)
        robot_horizon = _cfg_get(horizon_overrides, robot_type, None)
        robot_frame_stride = _cfg_get(robot_horizon, "frame_stride", None)
        if robot_frame_stride is not None:
            stride = int(robot_frame_stride)

    if stride <= 0:
        raise ValueError(f"latent_action.stride must be positive, got {stride}.")
    return stride


def _resolve_latent_action_horizon(la_cfg, robot_type: str | None, fallback: int) -> int:
    """Resolve the latent-action frame window length per embodiment.

    Defaults to ``latent_action.horizon`` (falling back to the action chunk
    length when unset) and may be overridden per robot via
    ``horizon_overrides.<robot>.horizon``.
    """
    horizon = _cfg_get(la_cfg, "horizon", None)
    horizon = int(horizon) if horizon is not None else int(fallback)
    if robot_type:
        horizon_overrides = _cfg_get(la_cfg, "horizon_overrides", None)
        robot_horizon = _cfg_get(horizon_overrides, robot_type, None)
        robot_horizon_value = _cfg_get(robot_horizon, "horizon", None)
        if robot_horizon_value is not None:
            horizon = int(robot_horizon_value)

    if horizon <= 0:
        raise ValueError(f"latent_action.horizon must be positive, got {horizon}.")
    return horizon


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

        video_key = _cfg_get(la_cfg, "video_key", None) or self.modality_keys["video"][0]
        if not str(video_key).startswith("video."):
            video_key = f"video.{video_key}"
        image_size = tuple(_cfg_get(la_cfg, "image_size", [224, 224]))

        full_window = bool(_cfg_get(la_cfg, "full_window", False))
        if full_window:
            sample["la_frames"] = self._pack_full_window_frames(
                int(trajectory_id), str(video_key), int(base_index), offsets, image_size
            )
        else:
            sample["la_frames"] = [
                frame.resize(image_size)
                for frame in self.get_video_frames_by_offsets(
                    int(trajectory_id), str(video_key), int(base_index), offsets
                )
            ]
        sample["la_frame_offsets"] = offsets
        return sample

    def _pack_full_window_frames(
        self,
        trajectory_id: int,
        video_key: str,
        base_index: int,
        offsets: list[int],
        image_size: tuple[int, int],
    ) -> list[list[Image.Image]]:
        """Return all frames within each stride window as a list of clips.

        For offsets [0, 16, 32], returns:
          [[frame_0, frame_1, ..., frame_16], [frame_16, frame_17, ..., frame_32]]
        """
        clips = []
        for i in range(len(offsets) - 1):
            window_offsets = list(range(offsets[i], offsets[i + 1] + 1))
            frames = self.get_video_frames_by_offsets(
                trajectory_id, video_key, base_index, window_offsets
            )
            clips.append([frame.resize(image_size) for frame in frames])
        return clips

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
    modality_config = dict(data_config.modality_config())
    transforms = data_config.transform()
    modality_config, transforms = _filter_latent_modalities(modality_config, transforms, data_cfg)
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
    include_robot_types: list[str] | set[str] | tuple[str, ...] | None = None,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    include_robot_types = {str(robot_type) for robot_type in include_robot_types} if include_robot_types else None

    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        if not _mixture_entry_matches_include(d_name, robot_type, include_robot_types):
            continue
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    if include_robot_types is not None and not filtered_mixture_spec:
        raise ValueError(
            f"No datasets in data_mix={data_mix!r} match include_robot_types={sorted(include_robot_types)}."
        )

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
