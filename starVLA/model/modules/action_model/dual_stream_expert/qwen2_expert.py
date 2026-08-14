# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Qwen2 action expert conditioned on per-layer VLM prefix K/V.

Each layer projects Q/K/V for the suffix, concatenates prefix K/V along the
sequence axis, and runs flex_attention (or SDPA) with a rectangular BlockMask
that preserves the pi0 state/action block semantics. Time enters through
AdaRMSNorm. No shared joint attention with the VLM; the VLM is a frozen-shape
prefix whose K/V are recomputed from its layer inputs.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from .joint_attention import (
    FlashDenseAttentionMeta,
    flash_attention_dense,
    flex_attention_with_block_mask,
    sdpa_attention_with_mask,
)


class AdaRMSNorm(nn.Module):
    """RMSNorm + optional FiLM.

    ``cond`` is ``None`` (no FiLM, plain RMSNorm), ``[B, D]`` (broadcast, typical
    time embedding), or ``[B, S, D]`` (per-token). ``gamma`` / ``beta`` are
    zero-initialised (DiT style) so FiLM starts as identity.
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)
        self.reset_film_parameters()

    def reset_film_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        if cond is None:
            return hidden_states.to(input_dtype)
        if cond.ndim not in (2, 3):
            raise ValueError(
                f"AdaRMSNorm cond must be None, [B, D], or [B, S, D], got {tuple(cond.shape)}."
            )
        gamma = self.gamma(cond).to(torch.float32)
        beta = self.beta(cond).to(torch.float32)
        if gamma.ndim == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        hidden_states = (1 + gamma) * hidden_states + beta
        return hidden_states.to(input_dtype)


def _make_norm(hidden_size: int, eps: float, adanorm_time: bool) -> nn.Module:
    return AdaRMSNorm(hidden_size, hidden_size, eps=eps) if adanorm_time else Qwen2RMSNorm(hidden_size, eps=eps)


def _apply_norm(norm: nn.Module, hidden_states: torch.Tensor, ada_cond: Optional[torch.Tensor]) -> torch.Tensor:
    if isinstance(norm, AdaRMSNorm):
        return norm(hidden_states, ada_cond)
    return norm(hidden_states)


class Qwen2ExpertDecoderLayer(GradientCheckpointingLayer):
    """Suffix decoder layer attending to ``[prefix_kv, suffix_kv]``."""

    def __init__(self, config: Qwen2Config, layer_idx: int, adanorm_time: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = _make_norm(config.hidden_size, config.rms_norm_eps, adanorm_time)
        self.post_attention_layernorm = _make_norm(config.hidden_size, config.rms_norm_eps, adanorm_time)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        block_mask=None,
        flash_meta: Optional[FlashDenseAttentionMeta] = None,
        ada_cond: Optional[torch.Tensor] = None,
        attention_implementation: str = "flex",
        return_suffix_kv: bool = False,
    ) -> torch.Tensor:
        # Expert is full fp32 end-to-end.
        hidden_states = hidden_states.float()
        if ada_cond is not None:
            ada_cond = ada_cond.float()
        prefix_key = prefix_key.float()
        prefix_value = prefix_value.float()

        residual = hidden_states
        hidden_states = _apply_norm(self.input_layernorm, hidden_states, ada_cond)

        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.float()
        sin = sin.float()
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Prefix K/V are already RoPE'd; concat along the sequence axis.
        suffix_key, suffix_value = key_states, value_states
        key_states = torch.cat([prefix_key, suffix_key], dim=2)
        value_states = torch.cat([prefix_value, suffix_value], dim=2)

        if attention_implementation == "flex":
            if block_mask is None:
                raise ValueError("flex expert attention requires a pre-built block_mask.")
            attn_output = flex_attention_with_block_mask(
                query_states, key_states, value_states, block_mask, scaling=self.scaling
            )
        elif attention_implementation == "sdpa":
            if attention_mask is None:
                raise ValueError("sdpa expert attention requires an attention_mask.")
            attn_output = sdpa_attention_with_mask(
                query_states, key_states, value_states, attention_mask, scaling=self.scaling
            )
        elif attention_implementation == "flash":
            if flash_meta is None:
                raise ValueError("flash expert attention requires a pre-built FlashDenseAttentionMeta.")
            attn_output = flash_attention_dense(
                query_states, key_states, value_states, flash_meta, scaling=self.scaling
            )
        else:
            raise ValueError(f"Unknown expert attention implementation: {attention_implementation}")

        attn_output = self.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = _apply_norm(self.post_attention_layernorm, hidden_states, ada_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if return_suffix_kv:
            return hidden_states, suffix_key, suffix_value
        return hidden_states


class Qwen2ExpertPreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2ExpertDecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, AdaRMSNorm):
            # ``apply`` visits the gamma/beta Linears first, so restore the zero
            # init that makes AdaRMSNorm start out as a plain RMSNorm.
            module.reset_film_parameters()


class Qwen2ExpertModel(Qwen2ExpertPreTrainedModel):
    """Qwen2 decoder stack without token embeddings or LM head."""

    def __init__(self, config: Qwen2Config, adanorm_time: bool = True, final_norm_adanorm: bool = False):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Qwen2ExpertDecoderLayer(config, layer_idx, adanorm_time)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _make_norm(config.hidden_size, config.rms_norm_eps, adanorm_time and final_norm_adanorm)
        self.gradient_checkpointing = False
        self.post_init()

    def apply_final_norm(self, hidden_states: torch.Tensor, ada_cond: Optional[torch.Tensor]) -> torch.Tensor:
        return _apply_norm(self.norm, hidden_states, ada_cond)


def build_qwen2_expert(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    adanorm_time: bool,
    final_norm_adanorm: bool,
) -> Qwen2ExpertModel:
    config = Qwen2Config(
        attention_dropout=0.0,
        hidden_act="silu",
        hidden_size=hidden_size,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        max_position_embeddings=32768,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        sliding_window=32768,
        use_sliding_window=False,
        tie_word_embeddings=True,
        vocab_size=151936,
        initializer_range=0.02,
    )
    config._attn_implementation = "eager"
    return Qwen2ExpertModel(config, adanorm_time=adanorm_time, final_norm_adanorm=final_norm_adanorm)
