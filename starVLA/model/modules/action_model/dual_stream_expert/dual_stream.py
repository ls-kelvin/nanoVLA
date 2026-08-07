# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Causal Qwen3-VL prefix + Qwen2 action expert conditioned on prefix K/V.

The VLM runs as a stock causal forward (flash_attention_2). Per-layer prefix
K/V are recomputed from each layer's real input (post-deepstack) and fed to the
matching action-expert layer, which attends over ``[prefix_prompt, suffix]``
with flex_attention. Training and inference share the same two-pass path.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    apply_rotary_pos_emb,
)

from .joint_attention import build_dense_flash_meta, build_expert_block_mask, build_suffix_attention_mask
from .qwen2_expert import build_qwen2_expert


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation != "flash_attention_2":
        return attn_implementation
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[WARNING] flash_attn not installed, falling back to sdpa for QwenPI_v5 prefix")
        return "sdpa"
    return attn_implementation


class QwenVLWithExpert(nn.Module):
    def __init__(
        self,
        base_vlm: str,
        *,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        expert_num_attention_heads: int = 16,
        expert_num_key_value_heads: int = 8,
        expert_head_dim: int = 128,
        num_expert_layers: Optional[int] = None,
        adanorm_time: bool = True,
        final_norm_adanorm: bool = False,
        attn_implementation: str = "flash_attention_2",
        expert_attention: str = "flex",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.attn_implementation = _resolve_attn_implementation(attn_implementation)
        self.expert_attention = expert_attention
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.final_norm_adanorm = final_norm_adanorm

        self.qwenvl = Qwen3VLForConditionalGeneration.from_pretrained(
            base_vlm,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        self.qwenvl.model.config.use_sliding_window = False
        self.qwenvl.model.config.sliding_window = None
        if hasattr(self.qwenvl.config, "text_config"):
            self.qwenvl.config.text_config.use_sliding_window = False
            self.qwenvl.config.text_config.sliding_window = None

        vlm_text_layers = int(self.qwenvl.config.text_config.num_hidden_layers)
        num_layers = num_expert_layers or vlm_text_layers
        if num_layers != vlm_text_layers:
            raise ValueError(
                f"Action expert layers ({num_layers}) must equal VLM text layers ({vlm_text_layers}) "
                "because each expert layer is conditioned on the matching VLM layer's K/V."
            )

        vlm_text_cfg = self.qwenvl.config.text_config
        if expert_num_attention_heads != int(vlm_text_cfg.num_attention_heads):
            raise ValueError(
                f"action_expert.num_attention_heads must match the VLM "
                f"({expert_num_attention_heads} vs {vlm_text_cfg.num_attention_heads})."
            )
        if expert_num_key_value_heads != int(vlm_text_cfg.num_key_value_heads):
            raise ValueError(
                f"action_expert.num_key_value_heads must match the VLM "
                f"({expert_num_key_value_heads} vs {vlm_text_cfg.num_key_value_heads})."
            )
        vlm_head_dim = int(
            getattr(vlm_text_cfg, "head_dim", vlm_text_cfg.hidden_size // vlm_text_cfg.num_attention_heads)
        )
        if expert_head_dim != vlm_head_dim:
            raise ValueError(f"action_expert.head_dim must match the VLM ({expert_head_dim} vs {vlm_head_dim}).")

        self.qwen_expert = build_qwen2_expert(
            hidden_size=expert_hidden_size,
            intermediate_size=expert_intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=expert_num_attention_heads,
            num_key_value_heads=expert_num_key_value_heads,
            head_dim=expert_head_dim,
            adanorm_time=adanorm_time,
            final_norm_adanorm=final_norm_adanorm,
        )
        # Action expert is full fp32 regardless of VLM dtype.
        self.qwen_expert = self.qwen_expert.float()

        self.num_layers = num_layers
        self.num_attention_heads = expert_num_attention_heads
        self.set_requires_grad()

    # -- setup helpers ----------------------------------------------------
    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
            for param in self.qwenvl.model.visual.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.qwenvl.eval()
            for param in self.qwenvl.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.qwenvl.model.visual.eval()
        if self.train_expert_only:
            self.qwenvl.eval()
        return self

    # -- embedding helpers ------------------------------------------------
    def embed_image(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """Return flat visual embeddings plus the per-layer deepstack features."""
        return self.qwenvl.model.visual(pixel_values, grid_thw=image_grid_thw)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def build_position_ids(self, input_ids, attention_mask, image_grid_thw=None) -> torch.Tensor:
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids

    # -- prefix encode (stock causal VLM) ---------------------------------
    def encode_prefix_layers(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the stock Qwen3-VL text tower and return ``(last_hidden, layer_inputs)``.

        ``layer_inputs[i]`` is the real input to layer ``i`` (already includes any
        deepstack injection from layer ``i-1``). Collecting these ourselves avoids
        the stock ``output_hidden_states`` path, which records layer outputs
        *before* deepstack and would therefore give the wrong K/V inputs.
        """
        language_model = self.qwenvl.model.language_model
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        cache_position = torch.arange(seq_len, device=device)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            rope_position_ids = position_ids
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=language_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(hidden_states, rope_position_ids)

        layer_inputs: List[torch.Tensor] = []
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            layer_inputs.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        last_hidden = language_model.norm(hidden_states)
        return last_hidden, layer_inputs

    def build_prefix_kv(
        self,
        layer_inputs: List[torch.Tensor],
        position_ids: torch.Tensor,
        prompt_len: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Recompute per-layer prefix K/V from each layer's real input.

        Returns a list of ``(key, value)`` with shape ``[B, Hkv, prompt_len, D]``,
        already RoPE'd with the VLM mrope. Bridge tokens past ``prompt_len`` are
        dropped so the action expert never sees them.
        """
        language_model = self.qwenvl.model.language_model
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rope_position_ids = position_ids[1:, :, :prompt_len]
        else:
            rope_position_ids = position_ids[:, :, :prompt_len]

        prefix_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = layer_inputs[layer_idx][:, :prompt_len]

            attn = decoder_layer.self_attn
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)

            key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            position_embeddings = language_model.rotary_emb(key_states, rope_position_ids)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, *position_embeddings)

            prefix_kvs.append((key_states, value_states))
        return prefix_kvs

    # -- expert forward ---------------------------------------------------
    def run_expert(
        self,
        suffix_embs: torch.Tensor,
        prefix_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        suffix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        ada_cond: Optional[torch.Tensor] = None,
        attention_implementation: Optional[str] = None,
    ) -> torch.Tensor:
        """Run the action expert over ``[prefix_kv, suffix]``. Returns suffix hidden.

        ``attention_implementation`` overrides ``self.expert_attention`` for this
        call so independent streams (e.g. action vs. latent) sharing one expert
        can pick different backends (see ``joint_attention.py`` for ``"flash"``,
        which only fits a fully bidirectional suffix with no block-causal mask).
        """
        if len(prefix_kvs) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} prefix K/V pairs, got {len(prefix_kvs)}.")

        if suffix_position_ids.ndim == 3 and suffix_position_ids.shape[0] == 4:
            rope_position_ids = suffix_position_ids[1:]
        elif suffix_position_ids.ndim == 3 and suffix_position_ids.shape[0] == 3:
            rope_position_ids = suffix_position_ids
        else:
            raise ValueError(f"suffix_position_ids must be 3D mrope ids, got {tuple(suffix_position_ids.shape)}")

        attention_implementation = attention_implementation or self.expert_attention

        # Build attention structure once; all layers share the same mask/meta.
        block_mask = None
        attention_mask = None
        flash_meta = None
        if attention_implementation == "flex":
            block_mask = build_expert_block_mask(
                prefix_pad_masks, suffix_pad_masks, suffix_att_masks
            )
        elif attention_implementation == "sdpa":
            attention_mask = build_suffix_attention_mask(
                prefix_pad_masks, suffix_pad_masks, suffix_att_masks
            )
        elif attention_implementation == "flash":
            if not bool(suffix_pad_masks.all()) or bool((suffix_att_masks[:, 1:] != 0).any()):
                raise ValueError(
                    "flash expert attention only supports a fully valid, fully bidirectional "
                    "suffix (no padding, no block-causal structure)."
                )
            flash_meta = build_dense_flash_meta(prefix_pad_masks, suffix_pad_masks.shape[1])
        else:
            raise ValueError(f"Unknown expert attention implementation: {attention_implementation}")

        # Suffix RoPE via the VLM mrope (same frequencies as the prefix K/V).
        language_model = self.qwenvl.model.language_model
        dummy = suffix_embs[:, :, :1]  # rotary_emb only needs dtype/device from x
        position_embeddings = language_model.rotary_emb(dummy, rope_position_ids)
        # Expert is fp32; cast RoPE tables to match.
        position_embeddings = (position_embeddings[0].float(), position_embeddings[1].float())

        hidden_states = suffix_embs.float()
        ada_cond = ada_cond.float() if ada_cond is not None else None
        for layer_idx, expert_layer in enumerate(self.qwen_expert.layers):
            prefix_key, prefix_value = prefix_kvs[layer_idx]
            hidden_states = expert_layer(
                hidden_states,
                prefix_key=prefix_key.float(),
                prefix_value=prefix_value.float(),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                block_mask=block_mask,
                flash_meta=flash_meta,
                ada_cond=ada_cond,
                attention_implementation=attention_implementation,
            )
        return self.qwen_expert.apply_final_norm(hidden_states, ada_cond)
