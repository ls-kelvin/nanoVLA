from collections import deque
from typing import Dict, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 20,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        infer_every_steps: Optional[int] = None,
        robotwin_action_type: str = "auto",
        image_channel_order: str = "rgb",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None
        self.last_infer_step = None
        self.infer_count = 0

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        self.infer_every_steps = self._resolve_infer_every_steps(infer_every_steps)
        self.robotwin_action_type = self._resolve_robotwin_action_type(robotwin_action_type, server_meta)
        self.image_channel_order = self._resolve_image_channel_order(image_channel_order)
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"infer_every_steps: {self.infer_every_steps}, "
            f"robotwin_action_type: {self.robotwin_action_type}, "
            f"image_channel_order: {self.image_channel_order}, server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.last_infer_step = None
        self.infer_count = 0
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

    def _resolve_infer_every_steps(self, infer_every_steps: Optional[int]) -> int:
        action_chunk_size = int(self.action_chunk_size)
        if infer_every_steps is None:
            return action_chunk_size

        infer_every_steps = int(infer_every_steps)
        if infer_every_steps < 1:
            raise ValueError("infer_every_steps must be >= 1")
        if infer_every_steps > action_chunk_size:
            raise ValueError(
                f"infer_every_steps ({infer_every_steps}) cannot exceed action_chunk_size ({action_chunk_size})"
            )
        return infer_every_steps

    def _resolve_robotwin_action_type(self, robotwin_action_type: str, server_meta: dict) -> str:
        requested = str(robotwin_action_type or "auto").lower()
        if requested in {"qpos", "ee"}:
            return requested
        if requested != "auto":
            raise ValueError("robotwin_action_type must be one of: auto, qpos, ee")

        action_keys = [str(key) for key in server_meta.get("action_keys", [])]
        if any("endpose" in key for key in action_keys):
            return "ee"
        return "qpos"

    def _resolve_image_channel_order(self, image_channel_order: str) -> str:
        image_channel_order = str(image_channel_order or "rgb").lower()
        if image_channel_order not in {"rgb", "bgr"}:
            raise ValueError("image_channel_order must be one of: rgb, bgr")
        return image_channel_order

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if self.image_channel_order == "bgr":
            return image[..., ::-1].copy()
        return image

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        example_copy = example.copy()
        if state is not None:
            example_copy["state"] = np.asarray(state).reshape(1, -1)
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        needs_infer = (
            self.raw_actions is None
            or self.last_infer_step is None
            or step - self.last_infer_step >= self.infer_every_steps
        )

        if needs_infer:
            self.infer_count += 1
            print(
                f"[RobotwinEval] policy inference #{self.infer_count}: "
                f"env_step={step}, infer_every_steps={self.infer_every_steps}, "
                f"action_chunk_size={self.action_chunk_size}, task={task_description!r}",
                flush=True,
            )
            response = self.client.predict_action(vla_input)
            # server already un-normalized via training-time transform
            raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

            if len(raw_actions) < self.infer_every_steps:
                raise ValueError(
                    f"Policy returned {len(raw_actions)} actions, fewer than infer_every_steps={self.infer_every_steps}"
                )

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions
            self.last_infer_step = step

        action_idx = step - self.last_infer_step
        if action_idx >= len(self.raw_actions):
            raise IndexError(
                f"Action index {action_idx} is out of range for cached action chunk of length {len(self.raw_actions)}"
            )

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = self._to_robotwin_action(current_action)
        return current_action

    def _to_robotwin_action(self, action: np.ndarray) -> np.ndarray:
        if self.robotwin_action_type == "ee":
            if action.shape[-1] != 16:
                raise ValueError(f"RoboTwin ee action expects 16 dims, got {action.shape[-1]}")
            return action[[0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]]

        if action.shape[-1] != 14:
            raise ValueError(f"RoboTwin qpos action expects 14 dims, got {action.shape[-1]}")
        return action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )
    infer_every_steps = usr_args.get("infer_every_steps", None)
    robotwin_action_type = usr_args.get("robotwin_action_type", "auto")
    image_channel_order = usr_args.get("image_channel_order", "rgb")

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
        infer_every_steps=infer_every_steps,
        robotwin_action_type=robotwin_action_type,
        image_channel_order=image_channel_order,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = model._prepare_image(observation["observation"]["head_camera"]["rgb"])
    left_img = model._prepare_image(observation["observation"]["left_camera"]["rgb"])
    right_img = model._prepare_image(observation["observation"]["right_camera"]["rgb"])

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    if model.robotwin_action_type == "ee":
        if "endpose" not in observation:
            raise KeyError(
                "RoboTwin ee eval requires observation['endpose']. "
                "Enable endpose in the RoboTwin task config."
            )
        endpose = observation["endpose"]
        state = np.concatenate(
            [
                np.asarray(endpose["left_endpose"], dtype=np.float32),
                np.asarray([endpose["left_gripper"]], dtype=np.float32),
                np.asarray(endpose["right_endpose"], dtype=np.float32),
                np.asarray([endpose["right_gripper"]], dtype=np.float32),
            ]
        )
    else:
        state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action, action_type=model.robotwin_action_type)
