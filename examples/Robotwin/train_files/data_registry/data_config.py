"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

import os
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    state_input_keys = ["state.left_joints", "state.left_gripper", "state.right_joints", "state.right_gripper"]
    # Per-key dims for PolicyNormProcessor (Agilex 6-DOF arms + binary gripper = 14-D total)
    action_key_dims = {"action.left_joints": 6, "action.right_joints": 6, "action.left_gripper": 1, "action.right_gripper": 1}
    state_key_dims  = {"state.left_joints": 6, "state.right_joints": 6, "state.left_gripper": 1, "state.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 (action_indices=50)
# ---------------------------------------------------------------------------
class AgilexData50Config(AgilexDataConfig):
    action_indices = list(range(50))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])
        
class AgilexData32Config(AgilexDataConfig):
    action_indices = list(range(32))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "mean_std", "state.right_joints": "mean_std",
                    "state.left_gripper": "min_max", "state.right_gripper": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "mean_std", "action.right_joints": "mean_std",
                    "action.left_gripper": "min_max", "action.right_gripper": "min_max",
                },
            ),
        ])
        
class AgilexData48Config(AgilexDataConfig):
    action_indices = list(range(48))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "mean_std", "state.right_joints": "mean_std",
                    "state.left_gripper": "min_max", "state.right_gripper": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "mean_std", "action.right_joints": "mean_std",
                    "action.left_gripper": "min_max", "action.right_gripper": "min_max",
                },
            ),
        ])


class Robotwin32EefDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_endpose", "state.right_endpose", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_endpose", "action.right_endpose", "action.left_gripper", "action.right_gripper"]
    state_input_keys = ["state.left_endpose", "state.left_gripper", "state.right_endpose", "state.right_gripper"]
    action_key_dims = {"action.left_endpose": 7, "action.right_endpose": 7, "action.left_gripper": 1, "action.right_gripper": 1}
    state_key_dims = {"state.left_endpose": 7, "state.right_endpose": 7, "state.left_gripper": 1, "state.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(32))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_endpose": "q99",
                    "state.right_endpose": "q99",
                    "state.left_gripper": "min_max",
                    "state.right_gripper": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_endpose": "q99",
                    "action.right_endpose": "q99",
                    "action.left_gripper": "min_max",
                    "action.right_gripper": "min_max",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — ARX X5
# ---------------------------------------------------------------------------
class ArxX5DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "q99", "state.right_joints": "q99",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "q99", "action.right_joints": "q99",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


_ROBOTWIN2_LEROBOT_ROOT = Path(
    os.environ.get(
        "ROBOTWIN2_LEROBOT_ROOT",
        "/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/lerobot",
    )
)

_ROBOTWIN2_HDF5_ROOT = Path(
    os.environ.get(
        "ROBOTWIN2_HDF5_ROOT",
        "/mnt/netdata/Team/Personal/jjc/data/RoboTwin2.0/dataset",
    )
)


def _as_filter_set(values) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def _normalize_domains(domains) -> set[str] | None:
    domains = _as_filter_set(domains)
    if domains is None:
        return None
    aliases = {
        "random": "randomized",
        "rand": "randomized",
        "randomized": "randomized",
        "clean": "clean",
    }
    normalized = set()
    for domain in domains:
        key = domain.lower()
        if key not in aliases:
            raise ValueError(f"Unknown RoboTwin2 domain {domain!r}; expected clean or randomized.")
        normalized.add(aliases[key])
    return normalized


def _matches_robotwin2_selection(
    task: str,
    embodiment: str,
    domain: str,
    tasks: set[str] | None,
    embodiments: set[str] | None,
    domains: set[str] | None,
) -> bool:
    if tasks is not None and task not in tasks:
        return False
    if embodiments is not None and embodiment not in embodiments:
        return False
    if domains is not None and domain not in domains:
        return False
    return True


def _parse_lerobot_dataset_name(dataset_name: str) -> tuple[str, str, str] | None:
    parts = Path(dataset_name).parts
    if len(parts) >= 3:
        task, embodiment, domain = parts[-3], parts[-2], parts[-1]
        if domain in {"clean", "randomized"}:
            return task, embodiment, domain
    if len(parts) == 2:
        task, split_name = parts
        for domain in ("clean", "randomized"):
            suffix = f"_{domain}"
            if split_name.endswith(suffix):
                return task, split_name[: -len(suffix)], domain
    return None


def _parse_hdf5_split_dir(split_dir: Path) -> tuple[str, str, str] | None:
    task = split_dir.parent.name
    split_name = split_dir.name
    for domain in ("clean", "randomized"):
        marker = f"_{domain}_"
        if marker in split_name:
            embodiment = split_name.split(marker, 1)[0]
            return task, embodiment, domain
    return None


def _discover(
    robot_type: str = "robotwin",
    include_dataset_substrings: list[str] | None = None,
    tasks: list[str] | set[str] | tuple[str, ...] | str | None = None,
    embodiments: list[str] | set[str] | tuple[str, ...] | str | None = None,
    domains: list[str] | set[str] | tuple[str, ...] | str | None = None,
    layout: str = "lerobot",
    root: Path | str | None = None,
) -> list[tuple[str, float, str]]:
    """Discover RoboTwin2 datasets with structured task/embodiment/domain filters.

    Args:
        robot_type: StarVLA robot_type/DataConfig key placed in mixture entries.
        include_dataset_substrings: Backward-compatible substring filter.
        tasks: Task names such as "hanging_mug" or ["place_phone_stand", ...].
        embodiments: Embodiments such as "aloha-agilex", "arx-x5", "piper", "ur5", "franka".
        domains: "clean", "randomized", or "random" alias.
        layout: "lerobot" for <task>/<embodiment>/<domain> with meta/modality.json,
            or "hdf5" for raw <task>/<embodiment>_<domain>_<count> directories.
        root: Optional override root. Defaults to ROBOTWIN2_LEROBOT_ROOT or ROBOTWIN2_HDF5_ROOT.

    Returns:
        Mixture entries as (dataset_name, weight, robot_type). For hdf5 layout,
        dataset_name is normalized to <task>/<embodiment>/<domain>.
    """
    tasks = _as_filter_set(tasks)
    embodiments = _as_filter_set(embodiments)
    domains = _normalize_domains(domains)
    layout = str(layout).lower()
    if layout not in {"lerobot", "hdf5"}:
        raise ValueError(f"Unknown RoboTwin2 layout {layout!r}; expected lerobot or hdf5.")

    root = Path(root) if root is not None else (_ROBOTWIN2_HDF5_ROOT if layout == "hdf5" else _ROBOTWIN2_LEROBOT_ROOT)
    if not root.is_dir():
        return []

    mixture: list[tuple[str, float, str]] = []
    if layout == "lerobot":
        dataset_dirs = sorted(modality_file.parent.parent for modality_file in root.glob("**/meta/modality.json"))
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.relative_to(root).as_posix()
            parsed = _parse_lerobot_dataset_name(dataset_name)
            if parsed is None:
                continue
            task, embodiment, domain = parsed
            if not _matches_robotwin2_selection(task, embodiment, domain, tasks, embodiments, domains):
                continue
            if include_dataset_substrings and not any(substr in dataset_name for substr in include_dataset_substrings):
                continue
            mixture.append((dataset_name, 1.0, robot_type))
        return mixture

    for split_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        parsed = _parse_hdf5_split_dir(split_dir)
        if parsed is None:
            continue
        task, embodiment, domain = parsed
        dataset_name = f"{task}/{embodiment}/{domain}"
        if not _matches_robotwin2_selection(task, embodiment, domain, tasks, embodiments, domains):
            continue
        if include_dataset_substrings and not any(substr in dataset_name for substr in include_dataset_substrings):
            continue
        if not (split_dir / "data").is_dir():
            continue
        mixture.append((dataset_name, 1.0, robot_type))
    return mixture


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin50": AgilexData50Config(),
    "robotwin32": AgilexData32Config(),
    "arx_x5": ArxX5DataConfig(),
    "robotwin48": AgilexData48Config(),
    "robotwin32_eef": Robotwin32EefDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "hdf5": [
        ("blocks_ranking_size/aloha-agilex/clean", 1.0, "robotwin32"),
    ],
    "hdf5_aloha_clean": _discover(
        "robotwin32",
        embodiments=["aloha-agilex"],
        domains=["clean"],
        layout="hdf5",
    ),
    "hdf5_aloha_clean_eef": _discover(
        "robotwin32_eef",
        embodiments=["aloha-agilex"],
        domains=["clean"],
        layout="hdf5",
    ),
    "hdf5_aloha_random_eef": _discover(
        "robotwin32_eef",
        embodiments=["aloha-agilex"],
        domains=["randomized"],
        layout="hdf5",
    ),
    "hdf5_arx_clean_random_eef": _discover(
        "robotwin32_eef",
        embodiments=["arx-x5"],
        domains=["clean", "randomized"],
        layout="hdf5",
    ),
    "hdf5_arx_clean_random_eef+hdf5_aloha_clean_eef": _discover(
        "robotwin32_eef",
        embodiments=["arx-x5"],
        domains=["clean", "randomized"],
        layout="hdf5",)+_discover(
        "robotwin32_eef",
        embodiments=["aloha-agilex"],
        domains=["clean"],
        layout="hdf5",
    ),
    "hdf5_aloha_random_eef": _discover(
        "robotwin32_eef",
        embodiments=["aloha-agilex"],
        domains=["randomized"],
        layout="hdf5",
    ),
    "hdf5_aloha_clean_random_eef": _discover(
        "robotwin32_eef",
        embodiments=["aloha-agilex"],
        domains=["clean", "randomized"],
        layout="hdf5",
    ),
    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"), ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"), ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"), ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"), ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"), ("Randomized/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"), ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"), ("Randomized/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"), ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"), ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"), ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"), ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"), ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"), ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"), ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"), ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"), ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"), ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"), ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"), ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"), ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"), ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"), ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"), ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"), ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"), ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"), ("Randomized/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"), ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"), ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"), ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"), ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"), ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"), ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"), ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"), ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"), ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"), ("Randomized/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"), ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"), ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"), ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"), ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"), ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"), ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"), ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"), ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"), ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"), ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"), ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"), ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"), ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"), ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"), ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"), ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"), ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"), ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"), ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"), ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"), ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"), ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"), ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"), ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"), ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"), ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"), ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"), ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"), ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"), ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"), ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"), ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"), ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"), ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"), ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"), ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"), ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"), ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"), ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"), ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin")],
    "robotwin_task2": [("place_a2b_left", 1.0, "robotwin"), ("place_a2b_right", 1.0, "robotwin")],
    "arx_x5": [("arx_x5", 1.0, "arx_x5")],
    "robotwin2_non_franka_all": _discover(
        "robotwin50",
        include_dataset_substrings=["aloha-agilex", "arx-x5", "piper", "ur5"],
    ),
    "robotwin32_non_franka_all_aloha_3": [
        (dataset_name, 3.0 if "aloha" in dataset_name else weight, robot_type)
        for dataset_name, weight, robot_type in _discover(
            "robotwin32",
            include_dataset_substrings=["aloha-agilex", "arx-x5", "piper", "ur5"],
        )
    ],
    "robotwin_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/aloha-agilex/randomized", 1.0, "robotwin50")
    ],
    "robotwin_cross_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 3.0, "robotwin50"), 
        ("place_phone_stand/aloha-agilex/randomized", 3.0, "robotwin50"),
        ("place_phone_stand/arx-x5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/arx-x5/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/piper/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/piper/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/ur5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/ur5/randomized", 1.0, "robotwin50"),
    ],
    "robotwin_no_piper_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 3.0, "robotwin50"), 
        ("place_phone_stand/aloha-agilex/randomized", 3.0, "robotwin50"),
        ("place_phone_stand/arx-x5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/arx-x5/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/franka/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/franka/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/ur5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/ur5/randomized", 1.0, "robotwin50"),
    ],
    "robotwin32_cross_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 3.0, "robotwin32"), 
        ("place_phone_stand/aloha-agilex/randomized", 3.0, "robotwin32"),
        ("place_phone_stand/arx-x5/clean", 1.0, "robotwin32"), 
        ("place_phone_stand/arx-x5/randomized", 1.0, "robotwin32"),
        ("place_phone_stand/piper/clean", 1.0, "robotwin32"), 
        ("place_phone_stand/piper/randomized", 1.0, "robotwin32"),
        ("place_phone_stand/ur5/clean", 1.0, "robotwin32"), 
        ("place_phone_stand/ur5/randomized", 1.0, "robotwin32"),
        # ("place_phone_stand/franka/clean", 1.0, "robotwin32"), 
        # ("place_phone_stand/franka/randomized", 1.0, "robotwin32"),
    ],
    "robotwin_cross_place_phone_stand_10": [
        ("place_phone_stand/aloha-agilex/clean", 10.0, "robotwin50"), 
        ("place_phone_stand/aloha-agilex/randomized", 10.0, "robotwin50"),
        ("place_phone_stand/arx-x5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/arx-x5/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/piper/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/piper/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/ur5/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/ur5/randomized", 1.0, "robotwin50"),
    ],
    "robotwin_aloha_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 1.0, "robotwin50"), 
        ("place_phone_stand/aloha-agilex/randomized", 1.0, "robotwin50"),
        ("place_phone_stand/arx-x5/clean", 0.0, "robotwin50"), 
        ("place_phone_stand/arx-x5/randomized", 0.0, "robotwin50"),
        ("place_phone_stand/piper/clean", 0.0, "robotwin50"), 
        ("place_phone_stand/piper/randomized", 0.0, "robotwin50"),
        ("place_phone_stand/ur5/clean", 0.0, "robotwin50"), 
        ("place_phone_stand/ur5/randomized", 0.0, "robotwin50"),
    ],
    "robotwin32_aloha_place_phone_stand": [
        ("place_phone_stand/aloha-agilex/clean", 1.0, "robotwin32"), 
        ("place_phone_stand/aloha-agilex/randomized", 1.0, "robotwin32"),
    ],
    "robotwin32_aloha_hanging_mug": [
        ("hanging_mug/aloha-agilex/clean", 1.0, "robotwin32"), 
        ("hanging_mug/aloha-agilex/randomized", 1.0, "robotwin32"),
    ],
    "robotwin48_aloha_hanging_mug_clean": [
        ("hanging_mug/aloha-agilex/clean", 1.0, "robotwin48"),
    ],
    "robotwin48_aloha_hanging_mug_randomized": [
        ("hanging_mug/aloha-agilex/randomized", 1.0, "robotwin48"),
    ],
    "robotwin32_all": _discover(
        "robotwin32",
        include_dataset_substrings=["aloha-agilex"],
    ),
    "robotwin32_5task": [
        ("hanging_mug/aloha-agilex/clean", 1.0, "robotwin32"),
        ("adjust_bottle/aloha-agilex/clean", 1.0, "robotwin32"),
        ("beat_block_hammer/aloha-agilex/clean", 1.0, "robotwin32"),
        ("click_bell/aloha-agilex/clean", 1.0, "robotwin32"),
        ("lift_pot/aloha-agilex/clean", 1.0, "robotwin32"),
        ("hanging_mug/aloha-agilex/randomized", 1.0, "robotwin32"),
        ("adjust_bottle/aloha-agilex/randomized", 1.0, "robotwin32"),
        ("beat_block_hammer/aloha-agilex/randomized", 1.0, "robotwin32"),
        ("click_bell/aloha-agilex/randomized", 1.0, "robotwin32"),
        ("lift_pot/aloha-agilex/randomized", 1.0, "robotwin32"),
    ],
}
