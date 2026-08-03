# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Attention helpers for the dual-stream action expert.

Prefix runs as a stock causal Qwen3-VL forward (flash_attention_2). The action
expert attends to ``[prefix_prompt, suffix]`` with a rectangular BlockMask that
keeps the pi0 state/action block semantics, so suffix uses flex_attention.
"""

import math

import torch
import torch.nn.functional as F
from packaging.version import Version

FLEX_SPARSE_BLOCK_SIZE = 128
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

if Version(torch.__version__) > Version("2.5.0"):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # Without compilation flex_attention falls back to an unfused kernel that
    # materializes the full scores matrix and ignores BlockMask sparsity.
    # dynamic=True avoids recompiles as q/kv lengths vary per batch.
    compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
else:
    compiled_flex_attention = None


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions (flow-matching time)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device) -> torch.Tensor:
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """big_vision block-causal 2D mask.

    A query attends to a key iff the key's cumulative block id (cumsum of
    ``att_masks``) is <= the query's, and the key is not padding.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def build_suffix_attention_mask(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
) -> torch.Tensor:
    """Bool mask ``[B, Lq, Lp+Ls]`` for suffix queries over ``[prefix, suffix]`` keys."""
    batch_size, prefix_len = prefix_pad_masks.shape
    suffix_len = suffix_pad_masks.shape[1]
    prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    return torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)


@torch.compiler.disable
def build_expert_block_mask(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    block_size: int = FLEX_SPARSE_BLOCK_SIZE,
):
    """Build a reusable flex BlockMask for expert attention over ``[prefix, suffix]``.

    ``Q_LEN`` / ``KV_LEN`` are the real sequence lengths (no 128-alignment padding).
    Partial trailing blocks are handled by flex_attention itself.

    ``mask_mod`` does not read ``h``, so the mask is built with ``H=None`` and
    broadcast across heads instead of being materialized once per head.
    """
    batch_size, prefix_len = prefix_pad_masks.shape
    suffix_len = suffix_pad_masks.shape[1]
    kv_len = prefix_len + suffix_len
    device = prefix_pad_masks.device

    # tokenizer padding_side="left" => contiguous left pads; offset = #pads.
    pad_offset = (~prefix_pad_masks).long().sum(dim=1)  # [B]
    suffix_block = torch.cumsum(suffix_att_masks.long(), dim=1)  # [B, Ls]
    # Invalid / padded suffix positions get a huge block id so nothing attends to them
    # via the block comparison; prefix padding is handled by pad_offset.
    suffix_block = suffix_block.masked_fill(~suffix_pad_masks, 10**9)

    def mask_mod(b, h, q_idx, kv_idx):
        is_prefix = kv_idx < prefix_len
        prefix_ok = kv_idx >= pad_offset[b]
        suffix_kv_idx = kv_idx - prefix_len
        # Clamp for gather safety on any partial-block probe past the real length.
        q_clamped = torch.minimum(q_idx, torch.full_like(q_idx, suffix_len - 1))
        sk_clamped = torch.minimum(
            torch.clamp(suffix_kv_idx, min=0),
            torch.full_like(suffix_kv_idx, suffix_len - 1),
        )
        suffix_ok = suffix_block[b, sk_clamped] <= suffix_block[b, q_clamped]
        visible = torch.where(is_prefix, prefix_ok, suffix_ok)
        return (q_idx < suffix_len) & (kv_idx < kv_len) & visible

    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=suffix_len,
        KV_LEN=kv_len,
        BLOCK_SIZE=block_size,
        device=device,
        _compile=False,
    )


def flex_attention_with_block_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_mask,
    scaling=None,
) -> torch.Tensor:
    """Run flex_attention with a pre-built BlockMask.

    Inputs are ``[B, H, L, D]`` in fp32. Returns ``[B, L, H*D]`` in fp32.
    """
    batch_size, _, seq_len, head_dim = query_states.shape

    attn_output = compiled_flex_attention(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        block_mask=block_mask,
        enable_gqa=True,
        scale=head_dim**-0.5 if scaling is None else scaling,
        kernel_options=FLEX_KERNEL_OPTIONS,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.reshape(batch_size, seq_len, -1)


def sdpa_attention_with_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
) -> torch.Tensor:
    """SDPA fallback with a bool mask ``[B, Lq, Lk]``. Inputs ``[B, H, L, D]``."""
    batch_size, num_heads, seq_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        key_states = (
            key_states[:, :, None, :, :]
            .expand(batch_size, num_kv_heads, n_rep, key_states.shape[2], head_dim)
            .reshape(batch_size, num_heads, key_states.shape[2], head_dim)
        )
        value_states = (
            value_states[:, :, None, :, :]
            .expand(batch_size, num_kv_heads, n_rep, value_states.shape[2], head_dim)
            .reshape(batch_size, num_heads, value_states.shape[2], head_dim)
        )

    scale = head_dim**-0.5 if scaling is None else scaling
    attn_mask = attention_mask[:, None, :, :]  # [B, 1, Lq, Lk]
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output.reshape(batch_size, seq_len, num_heads * head_dim)
