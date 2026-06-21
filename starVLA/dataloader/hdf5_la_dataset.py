"""HDF5 VLA dataset variant that adds latent-action frame samples."""

from pathlib import Path

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


class LatentActionHDF5SingleDataset(HDF5SingleDataset):
    """Raw-HDF5 dataset with optional UniT/latent-action frame-window packing."""

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
