import hashlib
import io
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    LeRobotModalityMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


def collate_fn(batch):
    return batch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _natural_episode_id(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode"):
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(stem.replace("episode", ""))


def _normalize_action_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"absolute", "abs"}:
        return "abs"
    if mode in {"delta", "relative_delta"}:
        return "delta"
    if mode in {"relative", "rel"}:
        return "rel"
    return mode


def _normalize_action_mode_apply_keys(action_mode_apply_keys):
    if action_mode_apply_keys is None:
        return None
    if isinstance(action_mode_apply_keys, str):
        return [action_mode_apply_keys]
    return [str(key) for key in action_mode_apply_keys]


def _normalize_action_mode_state_map(action_mode_state_map) -> dict[str, str]:
    return {str(action_key): str(state_key) for action_key, state_key in (action_mode_state_map or {}).items()}


def _normalize_image_channel_order(channel_order: str) -> str:
    channel_order = str(channel_order).lower()
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError(f"image channel order must be 'rgb' or 'bgr', got {channel_order!r}")
    return channel_order


def _resolve_hdf5_dataset_path(data_root_dir: Path, data_name: str) -> Path:
    direct = data_root_dir / data_name
    if direct.is_dir():
        return direct

    parts = Path(data_name).parts
    if len(parts) == 3:
        task, robot, domain = parts
        candidates = sorted((data_root_dir / task).glob(f"{robot}_{domain}_*"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"HDF5 dataset path not found for {data_name!r} under {data_root_dir}")


def _episode_files(dataset_path: Path) -> list[Path]:
    files = sorted((dataset_path / "data").glob("episode*.hdf5"), key=_natural_episode_id)
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {dataset_path / 'data'}")
    return files


def _read_instruction_candidates(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates: list[str] = []
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, str))
    elif isinstance(payload, list):
        candidates.extend(item for item in payload if isinstance(item, str))

    candidates = [item for item in candidates if item.strip()]
    if not candidates:
        raise ValueError(f"No usable instruction strings found in {path}")
    return candidates


def _stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def _data_config_uses_endpose(data_config) -> bool:
    action_keys = getattr(data_config, "action_keys", [])
    state_keys = getattr(data_config, "state_keys", [])
    keys = list(action_keys) + list(state_keys)
    return any("endpose" in str(key) for key in keys)


def _check_hdf5_action_type_matches_config(hdf5_action_type: str, robot_type: str, data_config) -> None:
    uses_endpose = _data_config_uses_endpose(data_config)
    if hdf5_action_type == "eef" and not uses_endpose:
        raise ValueError(
            "hdf5_action_type='eef' requires an EEF robot_type/DataConfig "
            f"(for example 'robotwin32_eef'), got robot_type={robot_type!r}."
        )
    if hdf5_action_type == "qpos" and uses_endpose:
        raise ValueError(
            "hdf5_action_type='qpos' requires a qpos/joint robot_type/DataConfig "
            f"(for example 'robotwin32'), got robot_type={robot_type!r}."
        )


class HDF5SingleDataset(Dataset):
    """RoboTwin2.0 raw-HDF5 dataset with the same sample surface as LeRobot."""

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        transforms: ComposedModalityTransform,
        embodiment_tag: str | EmbodimentTag,
        data_cfg=None,
        dataset_name: str | None = None,
        robot_type: str | None = None,
    ):
        self.data_cfg = data_cfg
        self.modality_configs = modality_configs
        self.transforms = transforms
        self._dataset_path = Path(dataset_path)
        self._dataset_name = dataset_name or self._dataset_path.name
        self.robot_type = robot_type
        self.hdf5_action_type = str(_cfg_get(data_cfg, "hdf5_action_type", "qpos")).lower()
        if self.hdf5_action_type not in {"qpos", "eef"}:
            raise ValueError(f"hdf5_action_type must be 'qpos' or 'eef', got {self.hdf5_action_type!r}")
        self.hdf5_image_channel_order = _normalize_image_channel_order(
            _cfg_get(data_cfg, "hdf5_image_channel_order", "bgr")
        )

        self.tag = embodiment_tag.value if isinstance(embodiment_tag, EmbodimentTag) else str(embodiment_tag)
        self._episode_files = _episode_files(self._dataset_path)
        self._cache_dir = self._dataset_path / "meta" / "hdf5_dataset" / self.hdf5_action_type
        self.curr_traj_id = None
        self.curr_traj_data = None

        self._init_action_mode()
        self._ensure_cache()

        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._tasks = self._get_tasks()
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)
        self._check_integrity()

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Initialized HDF5 dataset {self.dataset_name} with {self.tag}")

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        return self._delta_indices

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        return self._lerobot_info_meta

    @property
    def tasks(self) -> pd.DataFrame:
        return self._tasks

    def _cache_config(self) -> dict:
        files = [
            (path.name, path.stat().st_size, int(path.stat().st_mtime))
            for path in self._episode_files
        ]
        return {
            "format": "starvla_hdf5_dataset",
            "version": 1,
            "hdf5_action_type": self.hdf5_action_type,
            "dataset_name": self.dataset_name,
            "instruction_seed": int(_cfg_get(self.data_cfg, "hdf5_instruction_seed", 42)),
            "files_hash": hashlib.md5(repr(files).encode("utf-8")).hexdigest(),
        }

    def _ensure_cache(self) -> None:
        config_path = self._cache_dir / "cache_config.json"
        required = [
            self._cache_dir / "episodes.jsonl",
            self._cache_dir / "tasks.jsonl",
            self._cache_dir / "info.json",
            self._cache_dir / "modality.json",
            self._cache_dir / "stats_gr00t.json",
        ]
        config = self._cache_config()

        cache_valid = False
        if config_path.exists() and all(path.exists() for path in required):
            with open(config_path, "r", encoding="utf-8") as f:
                cache_valid = json.load(f) == config

        if cache_valid:
            return

        if _is_main_process():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_cache_files(config)

        if dist.is_initialized():
            dist.barrier()

    def _write_cache_files(self, config: dict) -> None:
        first_path = self._episode_files[0]
        with h5py.File(first_path, "r") as f:
            length = int(f["joint_action/vector"].shape[0])
            action_dim = int(self._read_action_array_from_file(f).shape[1])
            image_size = self._decode_image(f["observation/head_camera/rgb"][0]).size

        episodes = []
        tasks = []
        state_chunks = []
        action_chunks = []
        rng = random.Random(config["instruction_seed"])

        for episode_file in self._episode_files:
            episode_id = _natural_episode_id(episode_file)
            instruction_path = self._dataset_path / "instructions" / f"episode{episode_id}.json"
            instruction = rng.choice(_read_instruction_candidates(instruction_path))
            task_index = len(tasks)
            tasks.append({"task_index": task_index, "task": instruction})

            with h5py.File(episode_file, "r") as f:
                action_array = self._read_action_array_from_file(f).astype(np.float32)
                state_array = self._read_state_array_from_file(f).astype(np.float32)
                episodes.append(
                    {
                        "episode_index": episode_id,
                        "length": int(action_array.shape[0]),
                        "path": f"data/{episode_file.name}",
                        "task_index": task_index,
                    }
                )
                state_chunks.append(state_array)
                action_chunks.append(action_array)

        state_values = np.concatenate(state_chunks, axis=0)
        action_values = np.concatenate(action_chunks, axis=0)
        if state_values.shape[1] != action_dim:
            raise ValueError(f"State/action dim mismatch in {self.dataset_path}: {state_values.shape} vs {action_dim}")

        width, height = image_size
        info = self._build_info(action_dim=action_dim, width=width, height=height, total_frames=int(state_values.shape[0]))
        modality = self._build_modality(action_dim=action_dim)
        stats = {
            "observation.state": _stats(state_values),
            "action": _stats(action_values),
        }

        self._write_jsonl(self._cache_dir / "episodes.jsonl", episodes)
        self._write_jsonl(self._cache_dir / "tasks.jsonl", tasks)
        self._write_json(self._cache_dir / "info.json", info)
        self._write_json(self._cache_dir / "modality.json", modality)
        self._write_json(self._cache_dir / "stats_gr00t.json", stats)
        self._write_json(self._cache_dir / "cache_config.json", config)

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def _build_info(self, action_dim: int, width: int, height: int, total_frames: int) -> dict:
        fps = int(_cfg_get(self.data_cfg, "hdf5_fps", 30))
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [[f"dim_{idx}" for idx in range(action_dim)]],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [[f"dim_{idx}" for idx in range(action_dim)]],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for video_key in ["observation/head_camera/rgb", "observation/left_camera/rgb", "observation/right_camera/rgb"]:
            features[video_key] = {
                "dtype": "image",
                "shape": [3, height, width],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.fps": fps,
                    "video.channels": 3,
                },
            }

        return {
            "codebase_version": "hdf5",
            "robot_type": self._infer_robot_type_from_path(),
            "total_episodes": len(self._episode_files),
            "total_frames": total_frames,
            "total_tasks": len(self._episode_files),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {"train": f"0:{len(self._episode_files)}"},
            "data_path": "data/episode{episode_index}.hdf5",
            "video_path": "",
            "features": features,
        }

    def _build_modality(self, action_dim: int) -> dict:
        if self.hdf5_action_type == "eef":
            state = {
                "left_endpose": {"start": 0, "end": 7, "original_key": "observation.state"},
                "left_gripper": {"start": 7, "end": 8, "original_key": "observation.state"},
                "right_endpose": {"start": 8, "end": 15, "original_key": "observation.state"},
                "right_gripper": {"start": 15, "end": 16, "original_key": "observation.state"},
            }
            action = {
                "left_endpose": {"start": 0, "end": 7, "original_key": "action"},
                "left_gripper": {"start": 7, "end": 8, "original_key": "action"},
                "right_endpose": {"start": 8, "end": 15, "original_key": "action"},
                "right_gripper": {"start": 15, "end": 16, "original_key": "action"},
            }
        else:
            arm_dim = (action_dim - 2) // 2
            if action_dim != arm_dim * 2 + 2:
                raise ValueError(f"Unsupported qpos action dim {action_dim}; expected 2 arms plus 2 grippers.")
            state = {
                "left_joints": {"start": 0, "end": arm_dim, "original_key": "observation.state"},
                "left_gripper": {"start": arm_dim, "end": arm_dim + 1, "original_key": "observation.state"},
                "right_joints": {"start": arm_dim + 1, "end": arm_dim * 2 + 1, "original_key": "observation.state"},
                "right_gripper": {"start": arm_dim * 2 + 1, "end": action_dim, "original_key": "observation.state"},
            }
            action = {
                "left_joints": {"start": 0, "end": arm_dim, "original_key": "action"},
                "left_gripper": {"start": arm_dim, "end": arm_dim + 1, "original_key": "action"},
                "right_joints": {"start": arm_dim + 1, "end": arm_dim * 2 + 1, "original_key": "action"},
                "right_gripper": {"start": arm_dim * 2 + 1, "end": action_dim, "original_key": "action"},
            }

        return {
            "action": action,
            "state": state,
            "video": {
                "cam_high": {"original_key": "observation/head_camera/rgb"},
                "cam_left_wrist": {"original_key": "observation/left_camera/rgb"},
                "cam_right_wrist": {"original_key": "observation/right_camera/rgb"},
            },
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
            },
        }

    def _infer_robot_type_from_path(self) -> str:
        name = self._dataset_path.name
        for suffix in ("_clean_", "_randomized_"):
            if suffix in name:
                return name.split(suffix, 1)[0]
        return self.robot_type or name

    def _read_state_array_from_file(self, h5_file) -> np.ndarray:
        return self._read_action_array_from_file(h5_file)

    def _read_action_array_from_file(self, h5_file) -> np.ndarray:
        if self.hdf5_action_type == "qpos":
            return np.asarray(h5_file["joint_action/vector"], dtype=np.float32)
        left_endpose = np.asarray(h5_file["endpose/left_endpose"], dtype=np.float32)
        right_endpose = np.asarray(h5_file["endpose/right_endpose"], dtype=np.float32)
        left_gripper = np.asarray(h5_file["endpose/left_gripper"], dtype=np.float32).reshape(-1, 1)
        right_gripper = np.asarray(h5_file["endpose/right_gripper"], dtype=np.float32).reshape(-1, 1)
        return np.concatenate([left_endpose, left_gripper, right_endpose, right_gripper], axis=1)

    def _decode_image(self, raw) -> Image.Image:
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        else:
            raw = bytes(raw)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if self.hdf5_image_channel_order == "bgr":
            image = Image.fromarray(np.asarray(image)[:, :, ::-1])
        return image

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        with open(self._cache_dir / "modality.json", "r", encoding="utf-8") as f:
            return LeRobotModalityMetadata.model_validate(json.load(f))

    def _get_lerobot_info_meta(self) -> dict:
        with open(self._cache_dir / "info.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_tasks(self) -> pd.DataFrame:
        with open(self._cache_dir / "tasks.jsonl", "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
        return pd.DataFrame(tasks).set_index("task_index")

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        self.trajectory_ids_to_metadata = {}
        trajectory_ids = []
        trajectory_lengths = []
        with open(self._cache_dir / "episodes.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                episode = json.loads(line)
                episode_id = int(episode["episode_index"])
                trajectory_ids.append(episode_id)
                trajectory_lengths.append(int(episode["length"]))
                self.trajectory_ids_to_metadata[episode_id] = episode
        return np.asarray(trajectory_ids, dtype=np.int64), np.asarray(trajectory_lengths, dtype=np.int64)

    def _get_modality_keys(self) -> dict:
        return {modality: config.modality_keys for modality, config in self.modality_configs.items()}

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        delta_indices = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.asarray(config.delta_indices, dtype=np.int64)
        return delta_indices

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        with open(self._cache_dir / "stats_gr00t.json", "r", encoding="utf-8") as f:
            raw_stats = json.load(f)

        simplified_modality_meta = {"state": {}, "action": {}, "video": {}}
        dataset_statistics = {"state": {}, "action": {}}

        for modality in ["state", "action"]:
            modality_config = self.modality_configs.get(modality)
            if modality_config is None:
                continue
            le_state_action_meta = getattr(self.lerobot_modality_meta, modality)
            requested_subkeys = [key.split(".", 1)[1] for key in modality_config.modality_keys]
            for subkey in requested_subkeys:
                if subkey not in le_state_action_meta:
                    raise ValueError(
                        f"{modality} key {subkey!r} is missing in {self._cache_dir / 'modality.json'}"
                    )
                key_meta = le_state_action_meta[subkey]
                simplified_modality_meta[modality][subkey] = {
                    "absolute": key_meta.absolute,
                    "rotation_type": key_meta.rotation_type,
                    "shape": [key_meta.end - key_meta.start],
                    "continuous": True,
                }
                original_key = key_meta.original_key or subkey
                indices = np.arange(key_meta.start, key_meta.end)
                dataset_statistics[modality][subkey] = {
                    stat_name: np.asarray(raw_stats[original_key][stat_name])[indices].tolist()
                    for stat_name in ["mean", "std", "min", "max", "q01", "q99"]
                }

        for video_key, video_meta in self.lerobot_modality_meta.video.items():
            original_key = video_meta.original_key or video_key
            info = self.lerobot_info_meta["features"][original_key]
            height = info["shape"][info["names"].index("height")]
            width = info["shape"][info["names"].index("width")]
            simplified_modality_meta["video"][video_key] = {
                "resolution": [width, height],
                "channels": 3,
                "fps": self.lerobot_info_meta.get("fps", 30),
            }

        return DatasetMetadata(
            statistics=dataset_statistics,
            modalities=simplified_modality_meta,
            embodiment_tag=embodiment_tag,
        )

    def _get_all_steps(self) -> list[tuple[int, int]]:
        steps_path = self._cache_dir / "steps.pkl"
        if steps_path.exists():
            with open(steps_path, "rb") as f:
                import pickle

                return pickle.load(f)["steps"]

        steps = [
            (int(trajectory_id), int(base_index))
            for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths)
            for base_index in range(int(trajectory_length))
        ]
        if _is_main_process():
            import pickle

            tmp_path = steps_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump({"steps": steps}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)
        if dist.is_initialized():
            dist.barrier()
        return steps

    def _check_integrity(self) -> None:
        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                self.lerobot_modality_meta.get_key_meta(key)

    def _init_action_mode(self) -> None:
        action_mode = _cfg_get(self.data_cfg, "action_mode", "abs")
        action_mode = "abs" if action_mode is None else _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode
        self._action_mode_apply_keys = _normalize_action_mode_apply_keys(
            _cfg_get(self.data_cfg, "action_mode_apply_keys", None)
        )
        self._action_mode_state_map = _normalize_action_mode_state_map(
            _cfg_get(self.data_cfg, "action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]
        if not action_key.startswith("action."):
            return None
        state_key = action_key.replace("action.", "state.", 1)
        return state_key if state_key in self.modality_keys.get("state", []) else None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                continue
            state_key = self._infer_state_key_for_action(action_key)
            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values
            data[action_key] = out
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.all_steps)

    def __str__(self) -> str:
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)
        sample["episode_index"] = int(trajectory_id)
        sample["step_index"] = int(base_index)
        sample["episode_path"] = self.get_episode_data_path(trajectory_id).as_posix()
        return sample

    def get_episode_data_path(self, trajectory_id: int) -> Path:
        return self.dataset_path / self.trajectory_ids_to_metadata[int(trajectory_id)]["path"]

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        return self.get_episode_data_path(trajectory_id)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        indices = np.where(self.trajectory_ids == int(trajectory_id))[0]
        if len(indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {indices=}")
        return int(indices[0])

    def get_trajectory_data(self, trajectory_id: int) -> dict:
        trajectory_id = int(trajectory_id)
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data

        path = self.get_episode_data_path(trajectory_id)
        with h5py.File(path, "r") as f:
            action = self._read_action_array_from_file(f).astype(np.float32)
            state = self._read_state_array_from_file(f).astype(np.float32)
        task_index = int(self.trajectory_ids_to_metadata[trajectory_id]["task_index"])
        length = int(action.shape[0])
        fps = float(self.lerobot_info_meta.get("fps", 30))
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = {
            "observation.state": state,
            "action": action,
            "task_index": np.full(length, task_index, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / fps,
        }
        return self.curr_traj_data

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        raw_data = array[step_indices[~padding_positions]]
        expected_shape = (len(step_indices),) if raw_data.ndim == 1 else (len(step_indices), *array.shape[1:])
        output = np.zeros(expected_shape, dtype=array.dtype)
        output[~padding_positions] = raw_data
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding_indices] = array[0]
                output[end_padding_indices] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_state_or_action(self, trajectory_id: int, modality: str, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + int(base_index)
        max_length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        subkey = key.replace(modality + ".", "")
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        key_meta = le_state_or_action_cfg[subkey]
        original_key = key_meta.original_key or subkey
        data = self.get_trajectory_data(trajectory_id)
        data_array = np.asarray(data[original_key], dtype=np.float32)
        le_indices = np.arange(key_meta.start, key_meta.end)
        data_array = data_array[:, le_indices]
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + int(base_index)
        max_length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        subkey = key.replace("video.", "")
        original_key = self.lerobot_modality_meta.video[subkey].original_key or subkey
        path = self.get_episode_data_path(trajectory_id)
        frames = []
        with h5py.File(path, "r") as f:
            image_ds = f[original_key]
            for idx in step_indices:
                frames.append(np.asarray(self._decode_image(image_ds[int(idx)]), dtype=np.uint8))
        return np.stack(frames)

    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        step_indices = self.delta_indices[key] + int(base_index)
        max_length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        data = self.get_trajectory_data(trajectory_id)
        task_indices = data["task_index"][step_indices]
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(self, trajectory_id: int, modality: str, key: str, base_index: int):
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        if modality in {"state", "action"}:
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        if modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        raise ValueError(f"Invalid modality: {modality}")

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return self._apply_action_mode(data)

    def _pack_sample(self, data: dict) -> dict:
        obs_image_size = tuple(_cfg_get(self.data_cfg, "obs_image_size", (224, 224)))
        step_images = []
        for video_key in self.modality_keys["video"]:
            image = Image.fromarray(data[video_key][0]).resize(obs_image_size)
            step_images.append(image)

        sample = {
            "image": step_images,
            "lang": data[self.modality_keys["language"][0]][0],
            "robot_tag": self.tag,
            "robot_type": self.lerobot_info_meta.get("robot_type", None),
        }

        action_keys = self.modality_keys.get("action", [])
        if action_keys:
            sample["action"] = np.concatenate([data[action_key] for action_key in action_keys], axis=1).astype(np.float16)

        if _cfg_get(self.data_cfg, "include_state", False) not in ["False", False]:
            state_keys = self.modality_keys.get("state", [])
            if state_keys:
                sample["state"] = np.concatenate([data[state_key] for state_key in state_keys], axis=1).astype(np.float16)
        return sample


def make_HDF5SingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    data_cfg=None,
) -> HDF5SingleDataset:
    hdf5_action_type = str(_cfg_get(data_cfg, "hdf5_action_type", "qpos")).lower()
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    _check_hdf5_action_type_matches_config(hdf5_action_type, robot_type, data_config)
    embodiment_tag = getattr(data_config, "embodiment_tag", EmbodimentTag.NEW_EMBODIMENT)
    dataset_path = _resolve_hdf5_dataset_path(Path(data_root_dir), data_name)
    return HDF5SingleDataset(
        dataset_path=dataset_path,
        modality_configs=data_config.modality_config(),
        transforms=data_config.transform(),
        embodiment_tag=embodiment_tag,
        data_cfg=data_cfg,
        dataset_name=data_name,
        robot_type=robot_type,
    )


def _mixture_entry_matches_include(d_name: str, robot_type: str, include_patterns: set[str] | None) -> bool:
    if include_patterns is None:
        return True
    match_text = "\n".join([str(d_name), str(robot_type)])
    return any(pattern in match_text for pattern in include_patterns)


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
        dataset_mixture.append((make_HDF5SingleDataset(data_root_dir, d_name, robot_type, data_cfg=data_cfg), d_weight))

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
