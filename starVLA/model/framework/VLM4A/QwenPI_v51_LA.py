# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""QwenPI_v51_LA: state token inside the VLM prefix, LA queries visible to actions.

Layout change over QwenPI_v5_LA:

- Sequence: ``[prompt, <state>?, latent bridge tokens]`` -- the state rides the
  VLM (processed by VLM parameters) via a ``<state>`` placeholder token whose
  embedding is overwritten by ``state_vlm_proj``. Pure-video batches omit the
  token and the mrope positions simply continue ("position encoding skips").
- The action expert's prefix K/V span the full sequence, so the action suffix
  attends to the LA queries' per-layer hidden states (v5 sliced them off).
- The expert suffix is actions-only; the state token is removed from it.
- Inference appends the same number of bridge tokens as training (no latent
  loss) so the suffix positions and visible K/V match the training layout.
"""

from typing import List, Optional

import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v5_LA import Qwen_PI_v5_LA
from starVLA.model.modules.action_model.dual_stream_expert import DualStreamFlowMatchingV51
from starVLA.model.modules.vlm.QWen3 import merge_qwen3_vl_inputs
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

STATE_TOKEN = "<state>"


@FRAMEWORK_REGISTRY.register("QwenPI_v51_LA")
class Qwen_PI_v51_LA(Qwen_PI_v5_LA):
    """QwenPI_v5_LA with the state token moved into the VLM prefix."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._expand_state_token()

    def _build_action_model(self) -> DualStreamFlowMatchingV51:
        return DualStreamFlowMatchingV51(global_config=self.config)

    # ------------------------------------------------------------------ #
    # state token
    # ------------------------------------------------------------------ #
    def _expand_state_token(self) -> None:
        tokenizer = self.processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [STATE_TOKEN]})
        self.action_model.qwenvl_with_expert.qwenvl.resize_token_embeddings(len(tokenizer))
        ids = tokenizer(STATE_TOKEN, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"State token `{STATE_TOKEN}` must map to one token id, got {ids}.")
        self.state_token_id = int(ids[0])
        self.action_model.state_token_id = self.state_token_id

    @staticmethod
    def _batch_has_state(examples: List[dict]) -> bool:
        return "state" in examples[0]

    def _append_state_token(self, inputs: dict) -> dict:
        """Append one ``<state>`` placeholder per row (prompt tail, before bridge tokens)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        state_col = torch.full(
            (input_ids.shape[0], 1), self.state_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        inputs = dict(inputs)
        inputs["input_ids"] = torch.cat([input_ids, state_col], dim=1)
        inputs["attention_mask"] = torch.cat(
            [attention_mask, torch.ones_like(state_col, dtype=attention_mask.dtype)], dim=1
        )
        return inputs

    def _inference_window_count(self) -> int:
        """Bridge-token window count used at inference to match the training layout."""
        override = self.latent_action_cfg.get("inference_window_count", None)
        if override is not None:
            return int(override)
        la_data_cfg = self.config.datasets.vla_data.get("latent_action", {})
        horizon = int(la_data_cfg.get("horizon", self.action_horizon))
        stride = int(la_data_cfg.get("stride", 1))
        return max(horizon // stride, 1)

    # ------------------------------------------------------------------ #
    # stream inputs: [prompt, <state>?, bridge tokens]
    # ------------------------------------------------------------------ #
    def _build_stream_inputs(
        self, examples: List[dict], instructions: list[str], compute_latent: bool
    ) -> tuple[dict, int]:
        inputs = self._build_vlm_inputs(
            [example["image"] for example in examples],
            instructions,
            prebuilt_inputs=examples[0].get("vlm_inputs", None),
        )
        if self._batch_has_state(examples):
            inputs = self._append_state_token(inputs)
        if not compute_latent:
            return inputs, 0
        window_count = self._collect_latent_window_counts(examples)[0]
        inputs, num_latent_tokens = self._append_bridge_tokens(inputs, window_count)
        self._print_training_sample_once(examples, instructions)
        return inputs, num_latent_tokens

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        loss_mode = str(kwargs.pop("loss_mode", "joint")).lower()
        if loss_mode == "dual":
            return self._forward_dual(examples, kwargs.pop("dual_loss_modes", None))

        compute_action, compute_latent = self._resolve_loss_flags(loss_mode)

        instructions = [example["lang"] for example in examples]
        inputs, num_latent_tokens = self._build_stream_inputs(examples, instructions, compute_latent)
        device = inputs["input_ids"].device

        action_dit_loss = None
        latent_hidden = None
        action_train_batch_size = 0

        if compute_action:
            state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                examples, device
            )
            out = self.action_model(
                inputs,
                state,
                actions,
                action_mask,
                num_latent_tokens=num_latent_tokens,
                num_repeats=self.repeated_diffusion_steps,
            )
            action_dit_loss = out["action_loss"]
            if compute_latent:
                latent_hidden = out["latent_hidden"]
        elif compute_latent:
            state = self._prepare_state(examples, device)
            latent_hidden = self.action_model.encode_prefix(inputs, num_latent_tokens, state=state)

        return self._assemble_losses(
            examples, instructions, action_dit_loss, latent_hidden, action_train_batch_size
        )

    # ------------------------------------------------------------------ #
    # merged dual-dataloader forward
    # ------------------------------------------------------------------ #
    def _forward_dual(self, batches: dict, dual_loss_modes: dict | None) -> dict:
        """Merged VLM pass over both streams, extended with per-stream state tokens.

        Merging stays valid only when both streams share the exact tail layout
        (same bridge-token count and same state-token presence); otherwise fall
        back to one forward per stream.
        """
        if not isinstance(batches, dict):
            raise TypeError(f"loss_mode='dual' expects a dict of streams, got {type(batches)!r}.")
        modes = dual_loss_modes or {}
        names = [name for name in ("latent", "action") if batches.get(name)]
        if len(names) < 2:
            raise ValueError(f"loss_mode='dual' needs both streams, got {names}.")

        streams = []
        for name in names:
            examples = batches[name]
            loss_mode = str(modes.get(name, "joint")).lower()
            compute_action, compute_latent = self._resolve_loss_flags(loss_mode)
            instructions = [example["lang"] for example in examples]
            inputs, num_latent_tokens = self._build_stream_inputs(examples, instructions, compute_latent)
            streams.append(
                {
                    "name": name,
                    "examples": examples,
                    "instructions": instructions,
                    "loss_mode": loss_mode,
                    "compute_action": compute_action,
                    "compute_latent": compute_latent,
                    "inputs": inputs,
                    "num_latent_tokens": num_latent_tokens,
                    "has_state": self._batch_has_state(examples),
                }
            )

        if len({(stream["num_latent_tokens"], stream["has_state"]) for stream in streams}) != 1:
            return {
                "streams": {
                    stream["name"]: self.forward(stream["examples"], loss_mode=stream["loss_mode"])
                    for stream in streams
                }
            }

        pad_token_id = int(self.processor.tokenizer.pad_token_id)
        merged = merge_qwen3_vl_inputs([stream["inputs"] for stream in streams], pad_token_id)
        device = merged["input_ids"].device
        if streams[0]["has_state"]:
            merged_state = torch.cat(
                [self._prepare_state(stream["examples"], device) for stream in streams], dim=0
            )
        else:
            merged_state = None
        prefix, last_hidden, layer_inputs = self.action_model.encode_prefix_hidden(
            merged, streams[0]["num_latent_tokens"], state=merged_state
        )
        latent_start = prefix["latent_start"]

        row = 0
        outputs = {}
        for stream in streams:
            batch_size = stream["inputs"]["input_ids"].shape[0]
            rows = slice(row, row + batch_size)
            row += batch_size

            latent_hidden = last_hidden[rows, latent_start:] if stream["compute_latent"] else None
            action_dit_loss = None
            action_train_batch_size = 0
            if stream["compute_action"]:
                state, actions, action_mask, action_train_batch_size = self._prepare_action_inputs(
                    stream["examples"], device
                )
                stream_prefix = {
                    "position_ids": prefix["position_ids"][:, rows],
                    "prompt_pad_masks": prefix["prompt_pad_masks"][rows],
                    "prompt_len": prefix["prompt_len"],
                }
                action_dit_loss = self.action_model.flow_matching_loss(
                    stream_prefix,
                    self.action_model.build_prefix_kvs(prefix, layer_inputs, rows),
                    state,
                    actions,
                    action_mask,
                    num_repeats=self.repeated_diffusion_steps,
                )

            outputs[stream["name"]] = self._assemble_losses(
                stream["examples"],
                stream["instructions"],
                action_dit_loss,
                latent_hidden,
                action_train_batch_size,
            )

        return {"streams": outputs}

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        inputs = self._build_vlm_inputs(batch_images, instructions)
        device = inputs["input_ids"].device

        state = self._prepare_state(examples, device)
        if state is not None:
            inputs = self._append_state_token(inputs)
        # Match the training layout: the action suffix is positioned after the
        # LA bridge tokens and attends to their hidden states.
        if self.latent_action_enabled:
            inputs, _ = self._append_bridge_tokens(inputs, self._inference_window_count())

        if state is None:
            state = torch.zeros(len(examples), self.state_dim, device=device, dtype=torch.float32)

        # Training reaches the model through the accelerate/DeepSpeed wrapper, which
        # runs forward under autocast; inference is called on the unwrapped module,
        # so the same autocast has to be re-established for low-precision weights.
        param_dtype = self.action_model.state_proj.weight.dtype
        with torch.autocast(
            device.type,
            dtype=param_dtype,
            enabled=param_dtype in (torch.bfloat16, torch.float16),
        ):
            pred_actions = self.action_model.sample_actions(inputs, state)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
