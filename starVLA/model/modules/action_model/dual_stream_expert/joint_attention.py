# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Attention helpers for the dual-stream action expert.

Prefix runs as a stock causal Qwen3-VL forward (flash_attention_2). The action
expert attends to ``[prefix_prompt, suffix]`` with a rectangular BlockMask that
keeps the pi0 state/action block semantics, so suffix uses flex_attention.
"""

import math
from typing import NamedTuple

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


class FlashDenseAttentionMeta(NamedTuple):
    """Precomputed unpad bookkeeping for the dense (no block-causal) latent
    flash-attention path. Built once per forward (shared by every expert
    layer, since it only depends on prefix padding / suffix length, not on
    layer-specific K/V content).
    """

    gather_index: torch.Tensor  # [total_valid] long, into flattened [B*(Lp+Ls)]
    cu_seqlens_q: torch.Tensor  # [B+1] int32
    cu_seqlens_k: torch.Tensor  # [B+1] int32
    max_seqlen_q: int
    max_seqlen_k: int
    query_len: int


def build_dense_flash_meta(prefix_pad_masks: torch.Tensor, suffix_len: int) -> FlashDenseAttentionMeta:
    """Unpad bookkeeping for latent (fully bidirectional) expert attention.

    ``prefix_pad_masks`` is ``[B, Lp]`` with left padding (tokenizer
    ``padding_side="left"``); the suffix (latent target tokens) is always
    fully valid and has the same length across the batch (enforced upstream
    by the same-window-count constraint), so only the prefix needs unpadding.
    """
    batch_size, prefix_len = prefix_pad_masks.shape
    kv_len = prefix_len + suffix_len
    device = prefix_pad_masks.device

    pad_offset = (~prefix_pad_masks).long().sum(dim=1)  # [B]
    valid_len = (prefix_len - pad_offset) + suffix_len  # [B]

    row_base = torch.arange(batch_size, device=device) * kv_len
    gather_rows = []
    for b in range(batch_size):
        start = int(pad_offset[b].item())
        base = int(row_base[b].item())
        prefix_idx = torch.arange(start, prefix_len, device=device) + base
        suffix_idx = torch.arange(prefix_len, kv_len, device=device) + base
        gather_rows.append(torch.cat([prefix_idx, suffix_idx], dim=0))
    gather_index = torch.cat(gather_rows, dim=0)

    cu_seqlens_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seqlens_k[1:] = torch.cumsum(valid_len, dim=0).to(torch.int32)
    max_seqlen_k = int(valid_len.max().item())

    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * suffix_len, suffix_len, device=device, dtype=torch.int32
    )

    return FlashDenseAttentionMeta(
        gather_index=gather_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=suffix_len,
        max_seqlen_k=max_seqlen_k,
        query_len=suffix_len,
    )


def flash_attention_dense(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta: FlashDenseAttentionMeta,
    scaling=None,
) -> torch.Tensor:
    """Variable-length flash-attention for the latent (dense, no block-causal) stream.

    Inputs are ``[B, H, L, D]`` (fp32; the expert runs fp32 end-to-end). flash_attn
    requires fp16/bf16, so only the attention matmul itself is computed in bf16;
    RoPE and the surrounding residual path stay fp32. Returns ``[B, Lq, H*D]`` fp32.
    """
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    batch_size, num_q_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    kv_len = key_states.shape[2]
    if q_len != meta.query_len:
        raise ValueError(f"Latent query length {q_len} does not match flash meta ({meta.query_len}).")

    q = query_states.transpose(1, 2).reshape(batch_size * q_len, num_q_heads, head_dim)
    k = key_states.transpose(1, 2).reshape(batch_size * kv_len, num_kv_heads, head_dim)
    v = value_states.transpose(1, 2).reshape(batch_size * kv_len, num_kv_heads, head_dim)

    gather_index = meta.gather_index.to(k.device)
    k = k.index_select(0, gather_index)
    v = v.index_select(0, gather_index)

    compute_dtype = torch.bfloat16
    attn_output = flash_attn_varlen_func(
        q.to(compute_dtype),
        k.to(compute_dtype),
        v.to(compute_dtype),
        cu_seqlens_q=meta.cu_seqlens_q.to(q.device),
        cu_seqlens_k=meta.cu_seqlens_k.to(q.device),
        max_seqlen_q=meta.max_seqlen_q,
        max_seqlen_k=meta.max_seqlen_k,
        causal=False,
        softmax_scale=scaling,
    )
    attn_output = attn_output.to(query_states.dtype)
    return attn_output.reshape(batch_size, q_len, num_q_heads * head_dim)


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
