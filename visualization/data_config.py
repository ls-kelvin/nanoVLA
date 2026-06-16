"""Visualization-local robot_type DataConfig registry.

One config entry per robot embodiment. Kept separate from training registries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


@dataclass(frozen=True)
class RobotVizDataConfig:
    """Per-robot visualization DataConfig."""

    robot_type: str
    action_indices: list[int]
    left_joints_target_dim: int = 7
    right_joints_target_dim: int = 7
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    video_keys: tuple[str, ...] = (
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    )
    state_keys: tuple[str, ...] = (
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    )
    action_keys: tuple[str, ...] = (
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    )
    language_keys: tuple[str, ...] = ("annotation.human.action.task_description",)
    observation_indices: tuple[int, ...] = (0,)
    state_norm_modes: dict[str, str] | None = None
    action_norm_modes: dict[str, str] | None = None
    binary_threshold: float | None = None

    @property
    def horizon(self) -> int:
        return len(self.action_indices)

    def with_horizon(self, horizon: int) -> RobotVizDataConfig:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        return replace(self, action_indices=list(range(horizon)))

    @property
    def padded_action_dim(self) -> int:
        """Per-timestep action dim after left/right joint padding."""
        return (
            self.left_joints_target_dim
            + 1
            + self.right_joints_target_dim
            + 1
        )

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.video_keys),
            ),
            "state": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.state_keys),
            ),
            "action": ModalityConfig(
                delta_indices=list(self.action_indices),
                modality_keys=list(self.action_keys),
            ),
            "language": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.language_keys),
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        state_norm = self.state_norm_modes or {
            "state.left_joints": "q99",
            "state.right_joints": "q99",
            "state.left_gripper": "binary",
            "state.right_gripper": "binary",
        }
        action_norm = self.action_norm_modes or {
            "action.left_joints": "q99",
            "action.right_joints": "q99",
            "action.left_gripper": "binary",
            "action.right_gripper": "binary",
        }

        state_kwargs: dict = {"apply_to": list(self.state_keys), "normalization_modes": state_norm}
        action_kwargs: dict = {"apply_to": list(self.action_keys), "normalization_modes": action_norm}
        if self.binary_threshold is not None:
            state_kwargs["binary_threshold"] = self.binary_threshold
            action_kwargs["binary_threshold"] = self.binary_threshold

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=list(self.state_keys)),
                StateActionTransform(**state_kwargs),
                StateActionToTensor(apply_to=list(self.action_keys)),
                StateActionTransform(**action_kwargs),
            ]
        )


_ROBOTWIN32_NORM = {
    "state": {
        "state.left_joints": "mean_std",
        "state.right_joints": "mean_std",
        "state.left_gripper": "min_max",
        "state.right_gripper": "min_max",
    },
    "action": {
        "action.left_joints": "mean_std",
        "action.right_joints": "mean_std",
        "action.left_gripper": "min_max",
        "action.right_gripper": "min_max",
    },
}


def _robotwin32_config(robot_type: str, horizon: int = 32) -> RobotVizDataConfig:
    return RobotVizDataConfig(
        robot_type=robot_type,
        action_indices=list(range(horizon)),
        left_joints_target_dim=7,
        right_joints_target_dim=7,
        state_norm_modes=dict(_ROBOTWIN32_NORM["state"]),
        action_norm_modes=dict(_ROBOTWIN32_NORM["action"]),
        binary_threshold=0.49,
    )


# Per-robot configs. Each robot may use a different action horizon.
# Native joint dims: aloha/arx/piper/ur5 use 6+6 (padded to 7+7); franka uses 7+7 (16-D).
ALOHA_AGILEX_CONFIG = _robotwin32_config("aloha_agilex", horizon=16)
ARX_X5_CONFIG = _robotwin32_config("arx_x5", horizon=16)
PIPER_CONFIG = _robotwin32_config("piper", horizon=16)
UR5_CONFIG = _robotwin32_config("ur5", horizon=16)
FRANKA_CONFIG = _robotwin32_config("franka", horizon=16)

# Legacy / alternate horizons (optional).
ROBOTWIN16_CONFIG = _robotwin32_config("robotwin", horizon=16)

ROBOTWIN50_CONFIG = RobotVizDataConfig(
    robot_type="robotwin50",
    action_indices=list(range(50)),
    left_joints_target_dim=7,
    right_joints_target_dim=7,
    state_norm_modes={
        "state.left_joints": "q99",
        "state.right_joints": "q99",
        "state.left_gripper": "binary",
        "state.right_gripper": "binary",
    },
    action_norm_modes={
        "action.left_joints": "q99",
        "action.right_joints": "q99",
        "action.left_gripper": "binary",
        "action.right_gripper": "binary",
    },
    binary_threshold=0.49,
)


ROBOT_TYPE_CONFIG_MAP: dict[str, RobotVizDataConfig] = {
    "aloha_agilex": ALOHA_AGILEX_CONFIG,
    "arx_x5": ARX_X5_CONFIG,
    "piper": PIPER_CONFIG,
    "ur5": UR5_CONFIG,
    "franka": FRANKA_CONFIG,
    "robotwin": ROBOTWIN16_CONFIG,
    "robotwin50": ROBOTWIN50_CONFIG,
    # Backward-compatible alias.
    "robotwin32": ALOHA_AGILEX_CONFIG,
}


def get_robot_type_config(robot_type: str) -> RobotVizDataConfig:
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        available = ", ".join(sorted(ROBOT_TYPE_CONFIG_MAP))
        raise KeyError(f"Unknown robot_type={robot_type!r}. Available: {available}")
    return ROBOT_TYPE_CONFIG_MAP[robot_type]
