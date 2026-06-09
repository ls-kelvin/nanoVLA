# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenMetaQuery Framework
=======================
Standalone VLA framework porting the *action-loss* architecture of
``Mantis_flow_depth_use_metaquery`` into starVLA.

Difference from QwenPI / QwenPI_v3 (the only structural novelty):
  Instead of conditioning the layer-wise flow-matching Action DiT on the **full**
  VLM token sequence, this framework conditions it on a fixed set of learnable
  **metaquery** tokens.  We append ``<begin_of_img><img0>..<imgN-1><end_of_img>``
  to the prompt, resize the VLM token embeddings, run the VLM, and extract the
  per-layer hidden states at the metaquery positions as the DiT cross-attention
  context (``encode_condition_action``).  The context is fixed-length and fully
  valid, so no encoder attention mask is needed.

DiT shape: ``input_embedding_dim`` (DiT inner) stays small (1024) while
``cross_attention_dim`` equals the VLM hidden size — the VL→DiT reduction happens
*inside* the DiT cross-attention, so there is **no external projector** and we do
NOT call ``populate_layerwise_dit_cfg`` (which would force cross == inner).

Reused from starVLA: ``_QWen3_VL_Interface`` (VLM), ``LayerwiseFM`` low-level
modules (ActionEncoder / DiT, via the metaquery head subclass), dataloader,
trainer, registry.  Metaquery token-expansion mirrors ``QwenPI_v3_LA``.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.LayerwiseMetaqueryFM_ActionHeader import (
    LayerwiseMetaqueryFlowmatchingActionHead,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenMetaQuery (RoboTwin eef defaults, from
#  Mantis configs/robotwin_all_wo_pretrain_image_action_eef_64_metaqueries.yaml)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenMetaQueryDefaultConfig:
    name: str = "QwenMetaQuery"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 28,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # Metaquery conditioning
    metaquery: dict = field(
        default_factory=lambda: {
            "num_metaqueries": 64,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_dim": 16,
            "state_dim": 16,
            "action_horizon": 32,  # Mantis chunk_size
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 0,  # no register/future tokens (Mantis)
            "repeated_diffusion_steps": 16,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 32,
            # DiT shape: cross_attention_dim / num_layers / output_dim are set at
            # runtime in __init__ (cross = VLM hidden, num_layers = VLM layers,
            # output_dim = action_dim).  input_embedding_dim = heads * head_dim.
            "diffusion_model_cfg": {
                "input_embedding_dim": 1024,
                "attention_head_dim": 32,
                "num_attention_heads": 32,
                "norm_type": "norm",
                "dropout": 0.0,
                "final_dropout": False,
                "interleave_self_attention": True,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenMetaQuery")
class Qwen_MetaQuery(baseframework):
    """VLM + metaquery-conditioned layer-wise flow-matching action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenMetaQueryDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read true hidden size / layer count from the loaded VLM (Qwen3-VL-2B).
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Expand metaquery tokens and resize embeddings (mirror QwenPI_v3_LA).
        self.num_metaqueries = int(self.config.framework.metaquery.num_metaqueries)
        self._expand_metaquery_tokens()

        # Set DiT shape directly (no external projector, no populate_layerwise_dit_cfg):
        #   cross_attention_dim = VLM hidden, num_layers = VLM layers, output_dim = action_dim.
        action_cfg = self.config.framework.action_model
        dit_cfg = action_cfg.diffusion_model_cfg
        dit_cfg.cross_attention_dim = llm_hidden_size
        dit_cfg.num_layers = num_vl_layers
        dit_cfg.output_dim = int(action_cfg.action_dim)

        self.action_model = LayerwiseMetaqueryFlowmatchingActionHead(global_config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.action_horizon = int(action_cfg.action_horizon)
        self.repeated_diffusion_steps = int(action_cfg.get("repeated_diffusion_steps", 16))

    # ------------------------------------------------------------------ #
    # Metaquery setup + extraction
    # ------------------------------------------------------------------ #
    def _expand_metaquery_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        tokens = ["<begin_of_img>", "<end_of_img>"] + [f"<img{i}>" for i in range(self.num_metaqueries)]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))

        self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
        self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")
        img_ids = [tokenizer.convert_tokens_to_ids(f"<img{i}>") for i in range(self.num_metaqueries)]
        # Suffix appended after the prompt: <begin_of_img> <img0..N-1> <end_of_img>
        self.register_buffer(
            "metaquery_suffix_ids",
            torch.tensor([self.boi_token_id, *img_ids, self.eoi_token_id], dtype=torch.long),
            persistent=False,
        )

    def _build_metaquery_inputs(self, batch_images: List, instructions: List[str]) -> dict:
        """Reuse the shared VLM tokenizer, then append the metaquery suffix ids."""
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        suffix = self.metaquery_suffix_ids.to(input_ids.device).unsqueeze(0).expand(B, -1)
        inputs["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        inputs["attention_mask"] = torch.cat(
            [attention_mask, torch.ones((B, suffix.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        return inputs

    def encode_condition_action(self, input_ids: torch.Tensor, hidden_states) -> List[torch.Tensor]:
        """Slice the metaquery positions (between BOI/EOI) from every VLM layer.

        Returns a list (one tensor per DiT layer) of shape
        (B, num_metaqueries, vl_hidden), keeping the last ``num_action_dit_layers``.
        """
        boi_pos = torch.where(input_ids == self.boi_token_id)[1]
        eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]
        batch_size, seq_len = input_ids.shape
        indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(batch_size, -1)
        mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
        mask = mask.to(hidden_states[0].device)

        per_layer = []
        for layer in hidden_states:
            per_layer.append(layer[mask].view(batch_size, -1, layer.size(-1)))
        return per_layer[-self.num_action_dit_layers :]

    def _encode_metaquery_hidden_states(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        inputs = self._build_metaquery_inputs(batch_images, instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            meta_embs = self.encode_condition_action(inputs["input_ids"], outputs.hidden_states)
        return meta_embs

    # ------------------------------------------------------------------ #
    # Train / inference
    # ------------------------------------------------------------------ #
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        meta_embs = self._encode_metaquery_hidden_states(batch_images, instructions)
        base_hidden = meta_embs[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            r = self.repeated_diffusion_steps
            actions_target = actions_target.repeat(r, 1, 1)
            meta_embs = [h.repeat(r, 1, 1) for h in meta_embs]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(r, 1, 1)

            action_loss = self.action_model(meta_embs, actions_target, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        meta_embs = self._encode_metaquery_hidden_states(batch_images, instructions)
        base_hidden = meta_embs[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(meta_embs, state_t)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/Robotwin/train_files/starvla_metaquery_robotwin.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()

    if os.getenv("DEBUG_MODE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct"

    model = Qwen_MetaQuery(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    action_dim = int(cfg.framework.action_model.action_dim)
    state_dim = int(cfg.framework.action_model.state_dim)
    horizon = int(cfg.framework.action_model.action_horizon)
    sample = {
        "action": np.random.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16),
    }
    sample2 = dict(sample)
    sample2["lang"] = "Another fake instruction for testing."
    batch = [sample, sample2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Forward + gradient-aliveness check (loss-down is not sufficient).
    out = model(batch)
    loss = out["action_loss"]
    print(f"Action Loss: {loss.item()}")
    loss.backward()
    dit_grad = sum(
        p.grad.norm().item() for p in model.action_model.model.parameters() if p.grad is not None
    )
    emb_grad = model.qwen_vl_interface.model.get_input_embeddings().weight.grad
    emb_grad_norm = emb_grad.norm().item() if emb_grad is not None else 0.0
    print(f"[grad] action DiT grad-norm sum = {dit_grad:.4e} | input-embedding grad-norm = {emb_grad_norm:.4e}")
    assert dit_grad > 0, "Action DiT received zero gradient!"

    predict_output = model.predict_action([sample])
    print(f"Predicted normalized actions shape: {predict_output['normalized_actions'].shape}")
    print("Finished")
