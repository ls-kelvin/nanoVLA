"""Self-contained VillaX (IgorModel) implementation for latent-action encoding.

Extracted from third_party/villa-x to avoid external path dependencies.
Only the encoder + VQ path (``idm``) is used at inference; the decoder is
included so ``from_pretrained`` can load the full checkpoint without errors.

Key fix vs. original: VectorQuantizer2.forward casts ``self.embedding.weight``
to the same dtype as the input to prevent the BFloat16/Float mismatch under
AMP autocast.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pydantic import BaseModel, field_serializer
from torchvision import transforms as T


# ===================================================================
#  Utilities
# ===================================================================
def _flatten_internal(fn, flatten_ndim=3):
    def wrapper(x: torch.Tensor, *args, **kwargs):
        dim = x.shape[:-flatten_ndim]
        x = x.reshape(-1, *x.shape[-flatten_ndim:])
        x = fn(x, *args, **kwargs)
        x = x.reshape(*dim, *x.shape[-x.ndim + 1 :])
        return x
    return wrapper


def _hwc2chw(imgs: torch.Tensor):
    return rearrange(imgs, "... h w c -> ... c h w")


def _chw2hwc(imgs: torch.Tensor):
    return rearrange(imgs, "... c h w -> ... h w c")


def _hwc_internal(fn):
    def wrapper(imgs, *args, **kwargs):
        imgs = _chw2hwc(imgs)
        imgs = fn(imgs, *args, **kwargs)
        imgs = _hwc2chw(imgs)
        return imgs
    return wrapper


@_flatten_internal
def _resize(imgs: torch.Tensor, size: tuple[int] | int):
    if isinstance(size, int):
        size = (size, size)
    return T.Resize(size)(imgs)


@_hwc_internal
def _normalize_images(imgs: torch.Tensor):
    imgs = imgs.to(torch.float32) / 255
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)
    return (imgs - mean) / std


def _patching(images: torch.Tensor, patch_size: int):
    return rearrange(
        images,
        "... c (nh ph) (nw pw) -> ... (nh nw) (ph pw c)",
        pw=patch_size,
        ph=patch_size,
    )


# ===================================================================
#  Position embeddings
# ===================================================================
def _get_1d_sine_cosine(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_1d_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return _get_1d_sine_cosine(embed_dim, np.arange(length))


def get_2d_position_embeddings(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sine_cosine(embed_dim // 2, grid[0])
    emb_w = _get_1d_sine_cosine(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


# ===================================================================
#  Building blocks
# ===================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(self, resolution: int, patch_size: int, embed_dim: int, in_channels: int = 3) -> None:
        super().__init__()
        self.grid_size = (resolution // patch_size, resolution // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, patches: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        @_flatten_internal
        def embed(x: torch.Tensor) -> torch.Tensor:
            x = self.proj(x)
            return rearrange(x, "... c h w -> ... (h w) c")

        if isinstance(patches, list):
            return [embed(patch) for patch in patches]
        return embed(patches)


# ===================================================================
#  Attention modules
# ===================================================================
class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm
        if qk_norm:
            per_head = embed_dim // n_heads
            self.q_layernorm = nn.LayerNorm(per_head, elementwise_affine=True)
            self.k_layernorm = nn.LayerNorm(per_head, elementwise_affine=True)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, N_q, C = q_in.shape
        N_kv = kv_in.shape[1]
        kv = self.kv(kv_in).reshape(B, N_kv, 2, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q = self.q(q_in).reshape(B, N_q, 1, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = q[0]
        if self.qk_norm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)
        vals = F.scaled_dot_product_attention(q, k, v)
        vals = vals.transpose(1, 2).reshape(B, N_q, C)
        return self.dropout(self.proj(vals))


class AttentionWithMask(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor | list[torch.Tensor], causal: bool) -> torch.Tensor | list[torch.Tensor]:
        if causal:
            ret = []
            for xi in x:
                qkv = rearrange(self.qkv(xi), "n t (m h d) -> m n h t d", h=self.num_heads, m=3)
                q, k, v = qkv.unbind(0)
                q, k = self.q_norm(q), self.k_norm(k)
                xa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                xa = rearrange(xa, "n h t d -> n t (h d)")
                ret.append(self.proj(xa))
            return ret

        qkv = rearrange(self.qkv(x), "b n (m h d) -> m b h n d", h=self.num_heads, m=3)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj(x)


# ===================================================================
#  Transformer blocks
# ===================================================================
def _approx_gelu():
    return nn.GELU(approximate="tanh")


class Mlp(nn.Module):
    """timm Mlp-compatible block with stable fc1/fc2 parameter names."""

    def __init__(self, in_features: int, hidden_features: int, act_layer, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def _t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class STBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, st_use_qk_norm: bool,
                 d_s: int, d_t: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = AttentionWithMask(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=st_use_qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden,
            act_layer=_approx_gelu,
            drop=0.0,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.d_s, self.d_t = d_s, d_t
        self.attn_temp = AttentionWithMask(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=st_use_qk_norm)

    def forward(self, concat_x, pad_len, tpe=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.scale_shift_table[None].chunk(6, dim=1)
        x_m = _t2i_modulate(self.norm1(concat_x), shift_msa, scale_msa)
        x_s = self.attn(x_m, causal=False)
        concat_x = concat_x + gate_msa * x_s

        split_x = list(torch.split(concat_x, split_size_or_sections=pad_len, dim=0))
        split_x_ = []
        for i, sxi in enumerate(split_x):
            if tpe is None:
                split_x_.append(sxi.transpose(0, 1))
            else:
                split_x_.append(sxi.transpose(0, 1) + tpe[:, : pad_len[i]])
        x_t = self.attn_temp(split_x_, causal=True)
        x_t = [xti.transpose(0, 1) for xti in x_t]
        x_t = torch.cat(x_t, dim=0)
        x = concat_x + gate_msa * x_t
        x = x + gate_mlp * self.mlp(_t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MAPBlock(nn.Module):
    def __init__(self, n_latents: int, embed_dim: int, n_heads: int, output_dim: int,
                 mlp_ratio: float = 4.0, do_rms_norm: bool = True, do_swish_glu: bool = True,
                 qk_norm: bool = False) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.embed_dim = embed_dim
        self.pre_projection = nn.Linear(embed_dim, embed_dim)
        self.latents = nn.Parameter(torch.zeros(n_latents, embed_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.attn_norm = RMSNorm(embed_dim) if do_rms_norm else nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = FlashAttention(embed_dim, n_heads=n_heads, qk_norm=qk_norm)
        self.mlp_norm = RMSNorm(embed_dim) if do_rms_norm else nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_inner = int(mlp_ratio * embed_dim)
        self.mlp = nn.Sequential(
            SwishGLU(embed_dim, mlp_inner) if do_swish_glu else nn.Sequential(nn.Linear(embed_dim, mlp_inner), nn.GELU()),
            nn.Linear(mlp_inner, embed_dim),
        )
        self.final_proj = nn.Sequential(nn.Linear(embed_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])
        latents = self.attn_norm(latents + self.attn(q_in=latents, kv_in=self.pre_projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        latents = latents.squeeze(dim=1)
        return self.final_proj(latents)


class AttnBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 do_rms_norm: bool = False, do_swish_glu: bool = False,
                 do_layer_scale: bool = False, qk_norm: bool = False) -> None:
        super().__init__()
        self.do_layer_scale = do_layer_scale
        self.pre_norm_attn = RMSNorm(embed_dim) if do_rms_norm else nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = FlashAttention(embed_dim, n_heads=n_heads, qk_norm=qk_norm)
        if do_layer_scale:
            self.layer_scale_attn = LayerScale(embed_dim)
        self.pre_norm_mlp = RMSNorm(embed_dim) if do_rms_norm else nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_inner = int(mlp_ratio * embed_dim)
        self.mlp = nn.Sequential(
            SwishGLU(embed_dim, mlp_inner) if do_swish_glu else nn.Sequential(nn.Linear(embed_dim, mlp_inner), nn.GELU()),
            nn.Dropout(0.0),
            nn.Linear(mlp_inner, embed_dim),
        )
        if do_layer_scale:
            self.layer_scale_mlp = LayerScale(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if self.do_layer_scale:
            q = q + self.layer_scale_attn(self.attn(q_in=self.pre_norm_attn(q), kv_in=self.pre_norm_attn(kv)))
            q = q + self.layer_scale_mlp(self.mlp(self.pre_norm_mlp(q)))
        else:
            q = q + self.attn(q_in=self.pre_norm_attn(q), kv_in=self.pre_norm_attn(kv))
            q = q + self.mlp(self.pre_norm_mlp(q))
        return q


# ===================================================================
#  VectorQuantizer (with dtype fix)
# ===================================================================
class VectorQuantizer2(nn.Module):
    def __init__(self, n_e: int, e_dim: int, beta: float, remap=None, sane_index_shape: bool = False, legacy: bool = False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = nn.Embedding(n_e, e_dim)
        self.remap = remap
        self.re_embed = n_e
        self.sane_index_shape = sane_index_shape

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, use_entropy_loss=False):
        z_flattened = z.reshape(-1, self.e_dim)
        emb_weight = self.embedding.weight.to(z_flattened.dtype)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(emb_weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(emb_weight, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape).to(z_flattened.dtype)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.contiguous()

        return z_q, loss, (None, None, min_encoding_indices)


# ===================================================================
#  Config & pretrained base
# ===================================================================
class _PretrainedConfig(BaseModel):
    _config_path: str | None = None

    @classmethod
    def load(cls, path: str):
        if path.endswith(".json"):
            with open(path) as f:
                obj = cls.model_validate_json(f.read())
        elif path.endswith(".yaml"):
            import yaml
            with open(path) as f:
                obj = cls.model_validate(yaml.safe_load(f))
        else:
            raise ValueError(f"Unknown file format: {path}")
        obj._config_path = pathlib.Path(path).parent.absolute().as_posix()
        obj.resolve_path()
        return obj

    def resolve_path(self):
        for k, v in self.model_dump().items():
            if isinstance(v, str) and "$CONFIG_PATH" in v:
                setattr(self, k, v.replace("$CONFIG_PATH", self._config_path))


class _ModelConfig(BaseModel):
    architecture: str
    description: str = ""
    version: int = 0
    config: dict | _PretrainedConfig

    @field_serializer("config")
    def _serialize_config(self, value):
        if isinstance(value, _PretrainedConfig):
            return value.model_dump()
        return value

    @classmethod
    def load(cls, path: str | pathlib.Path):
        if isinstance(path, pathlib.Path):
            path = path.as_posix()
        if path.endswith(".json"):
            with open(path) as f:
                return cls.model_validate_json(f.read())
        elif path.endswith(".yaml"):
            import yaml
            with open(path) as f:
                return cls.model_validate(yaml.safe_load(f))
        raise ValueError(f"Unknown file format: {path}")


class _PretrainedModel(nn.Module):
    config_class: type[_PretrainedConfig]

    def __init__(self, config: _PretrainedConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def _transformer_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _init_weights(self, module):
        self._transformer_init(module)

    def _initialize_weights(self, module):
        if getattr(module, "_igor_initialized", False):
            return
        self._init_weights(module)
        module._igor_initialized = True

    def init_weights(self):
        self.apply(self._initialize_weights)

    def post_init(self):
        self.init_weights()

    def preprocess(self, clips: torch.Tensor, augment_type="no"):
        if clips.shape[-1] == 3:
            clips = _hwc2chw(clips)
        clips = _resize(clips, size=self.config.resolution)
        return _normalize_images(clips)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, strict: bool = True, **kwargs):
        from safetensors.torch import load_model

        model_dir = pathlib.Path(pretrained_model_path)
        config_path = model_dir / "model_config.json"
        weights_path = model_dir / "model.safetensors"
        assert config_path.exists(), f"Config not found: {config_path}"
        assert weights_path.exists(), f"Weights not found: {weights_path}"

        model_config = _ModelConfig.load(config_path)
        assert model_config.architecture == cls.__name__, (
            f"Architecture mismatch: expected {cls.__name__}, got {model_config.architecture}"
        )
        config = cls.config_class.model_validate(model_config.config)
        config._config_path = model_dir.absolute().as_posix()
        config.resolve_path()

        model = cls(config, **kwargs)
        model.eval()
        load_model(model=model, filename=weights_path, strict=strict)
        return model


# ===================================================================
#  IgorConfig
# ===================================================================
class IgorConfig(_PretrainedConfig):
    resolution: int = 224
    patch_size: int = 14
    in_channels: int = 3
    d_t: int = 8
    mlp_ratio: float = 4.0
    encoder_depth: int = 12
    encoder_embed_dim: int = 768
    encoder_n_heads: int = 8
    action_latent_dim: int = 128
    st_use_qk_norm: bool = True
    num_learned_tokens: int = 4
    map_heads: int = 24
    decoder_depth: int = 8
    decoder_embed_dim: int = 768
    decoder_n_heads: int = 16
    use_qk_norm: bool = True
    n_codes: int = 32
    grid_size: int | None = None
    embed_tokens: int | None = None
    augment_type: str = "resize_crop"
    augment_level: str = "clip"
    random_crop_scale: list[float] = [0.8, 1.0]  # noqa: RUF012
    random_crop_ratio: list[float] = [0.75, 4.0 / 3.0]  # noqa: RUF012

    def model_post_init(self, __context):
        self.grid_size = self.resolution // self.patch_size
        self.embed_tokens = self.grid_size**2


# ===================================================================
#  Helpers used by IgorEncoder
# ===================================================================
def _select_last_k(x: list[list], k: list[int], loffset: int = 0, roffset: int | None = None):
    roffset = roffset if roffset is None else -roffset
    return [x[i][-k[i] + loffset : roffset] for i in range(len(k))]


# ===================================================================
#  IgorEncoder
# ===================================================================
class IgorEncoder(_PretrainedModel):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.embed = PatchEmbed(config.resolution, config.patch_size, config.encoder_embed_dim)
        self.pe_spatial = nn.Parameter(
            torch.from_numpy(get_2d_position_embeddings(config.encoder_embed_dim, config.grid_size)).float().unsqueeze(0),
            requires_grad=False,
        )
        self.pe_temporal = nn.Parameter(
            torch.from_numpy(get_1d_position_embeddings(config.encoder_embed_dim, config.d_t)).float().unsqueeze(0),
            requires_grad=False,
        )
        self.layers = nn.ModuleList([
            STBlock(config.encoder_embed_dim, config.encoder_n_heads,
                    d_s=config.embed_tokens, d_t=config.d_t,
                    mlp_ratio=config.mlp_ratio, st_use_qk_norm=config.st_use_qk_norm)
            for _ in range(config.encoder_depth)
        ])
        self.norm = RMSNorm(config.encoder_embed_dim)
        self.map_block = MAPBlock(
            n_latents=config.num_learned_tokens, embed_dim=config.encoder_embed_dim,
            n_heads=config.map_heads, mlp_ratio=config.mlp_ratio,
            output_dim=config.action_latent_dim,
            do_rms_norm=True, do_swish_glu=True, qk_norm=False,
        )
        self.post_init()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed.proj.weight.data.view([self.embed.proj.weight.data.shape[0], -1]))
        return super().init_weights()

    def forward(self, clips: torch.Tensor, clip_len: list[int] | None = None):
        embedding = self.embed(clips)
        embedding = embedding + self.pe_spatial
        clip_len = clip_len if clip_len is not None else [clips.shape[1]] * clips.shape[0]
        embeddings = _select_last_k(embedding, clip_len)
        x = torch.cat(embeddings, dim=0)
        for idx, layer in enumerate(self.layers):
            x = layer(x, clip_len, tpe=self.pe_temporal if idx == 0 else None)
        x = self.norm(x)
        xs = list(torch.split(x, split_size_or_sections=clip_len, dim=0))
        xs = torch.cat([(xi[1:] + xi[:-1]) / 2.0 for xi in xs], dim=0)
        action = self.map_block(xs)
        action = rearrange(action, "b n d -> b 1 (n d)")
        return action

    @torch.inference_mode()
    def idm(self, clips: torch.Tensor | list[torch.Tensor], mask: torch.Tensor | None = None):
        if isinstance(clips, list):
            max_len = max(x.shape[1] for x in clips)
            clips = [F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in clips]
            clips = torch.stack(clips, dim=0)
        elif len(clips.shape) == 4:
            clips = clips.unsqueeze(0)
            mask = mask.unsqueeze(0) if mask is not None else None
        mask = torch.ones(*clips.shape[:2], device=clips.device).bool() if mask is None else mask
        clips = self.preprocess(clips, augment_type="no")

        t = clips.shape[1]
        if t <= self.config.d_t:
            clips = F.pad(clips, (0, 0, 0, self.config.d_t - t))
            mask = F.pad(mask, (0, self.config.d_t - t))
            t = self.config.d_t
        actions = None
        for i in range(t - self.config.d_t + 1):
            clips_ = clips[:, i : i + self.config.d_t]
            mask_ = mask[:, i : i + self.config.d_t]
            action = self.forward(clips_, mask_.sum(dim=1).tolist())
            if i == 0:
                actions = list(torch.split(action, split_size_or_sections=(mask_.sum(dim=1) - 1).tolist(), dim=0))
            else:
                new_actions = torch.split(action, split_size_or_sections=(mask_.sum(dim=1) - 1).tolist(), dim=0)
                new_action = [xi[-1].unsqueeze(0) for xi in new_actions]
                actions = [torch.cat([x, y], dim=0) for x, y in zip(actions, new_action)]
        return actions


# ===================================================================
#  IgorDecoder (needed for from_pretrained weight loading)
# ===================================================================
class IgorDecoder(_PretrainedModel):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.embed = PatchEmbed(config.resolution, config.patch_size, config.decoder_embed_dim)
        self.pe = nn.Parameter(
            torch.from_numpy(get_2d_position_embeddings(config.decoder_embed_dim, config.grid_size)).float().unsqueeze(0),
            requires_grad=False,
        )
        self.action_embed = nn.Linear(config.action_latent_dim * config.num_learned_tokens, config.decoder_embed_dim)
        self.layers = nn.ModuleList([
            AttnBlock(config.decoder_embed_dim, config.decoder_n_heads, config.mlp_ratio,
                      do_rms_norm=True, do_swish_glu=True, do_layer_scale=True, qk_norm=config.use_qk_norm)
            for _ in range(config.decoder_depth)
        ])
        self.norm = RMSNorm(config.decoder_embed_dim)
        head_dim = config.patch_size**2 * config.in_channels
        self.pred_head = nn.Linear(config.decoder_embed_dim, head_dim)
        self.post_init()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embed.proj.weight.data.view([self.embed.proj.weight.data.shape[0], -1]))
        return super().init_weights()


# ===================================================================
#  IgorModel
# ===================================================================
class IgorModel(_PretrainedModel):
    config_class = IgorConfig

    def __init__(self, config: IgorConfig) -> None:
        super().__init__(config)
        self.encoder = IgorEncoder(config)
        self.vq = VectorQuantizer2(config.n_codes, config.action_latent_dim, beta=0.25)
        self.decoder = IgorDecoder(config)
        self.post_init()

    def init_weights(self):
        nn.init.uniform_(self.vq.embedding.weight, -1.0 / self.vq.n_e, 1.0 / self.vq.n_e)
        return super().init_weights()

    @torch.inference_mode()
    def idm(self, clips: torch.Tensor, *, return_dict: bool = False):
        tokens = self.encoder.idm(clips)
        vq_tokens, _, (_, _, indices) = self.vq(torch.stack(tokens).contiguous())
        if not return_dict:
            return vq_tokens
        return {"tokens": tokens, "vq_tokens": vq_tokens, "indices": indices}
