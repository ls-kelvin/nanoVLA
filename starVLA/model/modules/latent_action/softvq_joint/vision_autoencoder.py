"""Vision pair autoencoder with soft categorical latent (FDM structure).

Architecture
------------
Frame encoder (shared, frozen if pretrained):
    frame_i   (B,3,H,W)  →  src_tokens  (B,N,D_frame)
    frame_i+k (B,3,H,W)  →  tgt_tokens  (B,N,D_frame)

    Supported types:
      - "cnn"  : lightweight Conv2d patchifier trained from scratch.
      - "dino" : frozen DINOv2 loaded from a local path via HuggingFace
                 transformers.  Set frame_encoder.pretrained_path in config.

frame_to_hidden : Linear(D_frame, H)  [Identity when D_frame == H]

Delta encoder (Transformer):
    [cls, src_tokens+type_src+pos, tgt_tokens+type_tgt+pos]
    → TransformerEncoder (norm_first)
    → CLS output → SoftVectorQuantizer (cosine soft-VQ):
        encode_proj(H→cd) → L2-norm → softmax(-dist/τ) → weights (B,K)
        embedding = decode_proj(weights @ L2-norm(codebook))  (B,E)
    Both encoder output and codebook are unit vectors, so the assignment logits
    are bounded → the softmax cannot saturate → no index/saturation collapse.

Decoder (FDM, AdaLN-Zero):
    query = src_tokens_h (MAE-masked) + dec_pos_embed   (B,N,H)
    condition c = latent_proj(embedding)   (B,H)
    Each AdaLNDecoderBlock: c → (shift, scale, gate) per sub-layer
    → output_head → recon_tokens (B,N,D_frame)   [feature mode]
    or PixelHead  → recon_image  (B,3,H,W)        [pixel mode]

    src_mask_ratio: at train time a fraction of source tokens are replaced by a
    learnable mask token, forcing the decoder to use the latent (breaks the
    near-identity reconstruction shortcut that drives latent collapse).

Loss (single-coefficient dual-entropy regulariser):
    recon_loss   MSE against tgt_frame_tokens (feature) or frame_future (pixel)
    entropy_loss = mi_beta * (H(z|x) - H(z))   [computed in SoftVectorQuantizer,
                   on a separate sharper temperature]
    total = recon_loss + entropy_loss = recon_loss - mi_beta * I(z; x)
      maximise I(z;x): sharpen each sample (low H(z|x)) while keeping batch
      codebook usage uniform (high H(z)).
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Frame encoders
# ---------------------------------------------------------------------------

class CNNFrameEncoder(nn.Module):
    """Simple Conv2d patchifier; trainable from scratch."""

    def __init__(self, token_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.token_dim = token_dim
        self.proj = nn.Conv2d(3, token_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor):
        """(B,3,H,W) → tokens (B,N,D), grid (H//p, W//p)."""
        x = self.proj(images)               # (B,D,h,w)
        grid = (x.shape[2], x.shape[3])
        tokens = x.flatten(2).transpose(1, 2)  # (B,N,D)
        return tokens, grid


class DINOFrameEncoder(nn.Module):
    """Frozen DINOv2 loaded from a local HuggingFace checkpoint directory.

    Parameters
    ----------
    pretrained_path : str
        Local path to a dinov2-{small|base|large} model directory downloaded
        from HuggingFace (facebook/dinov2-small / facebook/dinov2-base /
        facebook/dinov2-large).  No network access is performed; the path must
        exist on disk.
    """

    def __init__(self, pretrained_path: str) -> None:
        super().__init__()
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"DINOv2 pretrained_path not found: {pretrained_path}\n"
                "Download it from HuggingFace:\n"
                "  facebook/dinov2-small  (token_dim=384)\n"
                "  facebook/dinov2-base   (token_dim=768)\n"
                "  facebook/dinov2-large  (token_dim=1024)\n"
                "and set frame_encoder.pretrained_path in your config."
            )
        self.model = AutoModel.from_pretrained(pretrained_path)
        self.model.requires_grad_(False)
        self.model.eval()
        # Both attributes come from the pretrained model's own config; the
        # user-supplied config value (if any) is intentionally ignored here.
        self.token_dim:  int = self.model.config.hidden_size
        self.patch_size: int = self.model.config.patch_size

    def forward(self, images: torch.Tensor):
        """(B,3,H,W) → tokens (B,N,D), grid (h,w)."""
        with torch.no_grad():
            out = self.model(pixel_values=images)
        # last_hidden_state: (B, 1+N, D); drop the CLS token at index 0
        tokens = out.last_hidden_state[:, 1:]              # (B, N, D)
        h = images.shape[2] // self.patch_size
        w = images.shape[3] // self.patch_size
        return tokens, (h, w)


def build_frame_encoder(cfg) -> tuple[nn.Module, int]:
    """Build frame encoder from config; return (encoder, token_dim)."""
    enc_type = str(cfg.get("type", "cnn"))
    if enc_type == "cnn":
        token_dim = int(cfg.get("token_dim", 256))
        patch_size = int(cfg.get("patch_size", 16))
        enc = CNNFrameEncoder(token_dim=token_dim, patch_size=patch_size)
        return enc, token_dim
    elif enc_type == "dino":
        pretrained_path = str(cfg.pretrained_path)
        enc = DINOFrameEncoder(pretrained_path=pretrained_path)
        return enc, enc.token_dim
    else:
        raise ValueError(f"Unknown frame_encoder type: {enc_type!r}. Choose 'cnn' or 'dino'.")


# ---------------------------------------------------------------------------
# AdaLN-Zero decoder block
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply affine modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNDecoderBlock(nn.Module):
    """Transformer decoder block with adaLN-Zero conditioning (DiT style).

    The condition vector ``c`` (B,H) is projected to 6*H parameters:
      (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    All projections initialised to zero so that at init the block is an
    identity function (stable training start).

    Self-attention only (no cross-attention); the latent is injected purely
    through AdaLN modulation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: 6 parameters per token position (shift/scale/gate × 2 sub-layers)
        self.ada_proj = nn.Linear(hidden_dim, 6 * hidden_dim)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, H)  token sequence
        c : (B, H)     condition (projected latent embedding)
        """
        ada = self.ada_proj(F.silu(c))          # (B, 6H)
        shift1, scale1, gate1, shift2, scale2, gate2 = ada.chunk(6, dim=-1)  # each (B,H)

        # Attention sub-layer
        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate1.unsqueeze(1) * attn_out

        # MLP sub-layer
        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_norm)

        return x


# ---------------------------------------------------------------------------
# Pixel head (used in pixel reconstruction mode)
# ---------------------------------------------------------------------------

class PixelHead(nn.Module):
    """Project patch tokens back to pixels via ConvTranspose2d + Sigmoid."""

    def __init__(self, token_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 3, kernel_size=patch_size, stride=patch_size),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """(B, N, D) + grid (h,w) → (B, 3, H, W)."""
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, grid[0], grid[1])
        return self.net(x)


# ---------------------------------------------------------------------------
# Soft vector quantizer (cosine, dual-entropy)  — ported from nanovla
# ---------------------------------------------------------------------------

class SoftVectorQuantizer(nn.Module):
    """Cosine-similarity soft VQ with single-coefficient dual-entropy reg.

    Ported from ``tok_act.model.vq.SoftVectorQuantizer``. Assignment weights are
    a softmax over the *negative L2 distance between L2-normalised vectors*, i.e.
    a temperature-scaled cosine similarity. Because both the encoder output and
    the codebook are unit vectors, the distance lies in [0, 4] and the logits
    ``-distance/temperature`` are **bounded** → the softmax can never saturate to
    a degenerate one-hot fixed point → no index/saturation collapse.

    The dual-entropy (mutual-information) regulariser is computed on a *separate,
    sharper* temperature and uses a single coefficient ``entropy_loss_ratio``:

        entropy_loss = ratio * (E_b[H(p)] - H(E_b[p])) = -ratio * I(z; x)

    so minimising it maximises I(z;x): sharpen each sample (low conditional
    entropy) while keeping the batch-marginal usage uniform (high marginal
    entropy).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        codebook_size: int,
        codebook_dim: int,
        temperature: float = 0.25,
        entropy_temperature: float = 0.1,
        entropy_loss_ratio: float = 0.01,
        l2_norm: bool = True,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.temperature = float(temperature)
        self.entropy_temperature = float(entropy_temperature)
        self.entropy_loss_ratio = float(entropy_loss_ratio)
        self.l2_norm = bool(l2_norm)

        self.encode_proj = nn.Identity() if in_dim == self.codebook_dim else nn.Linear(in_dim, self.codebook_dim)
        self.decode_proj = nn.Identity() if self.codebook_dim == out_dim else nn.Linear(self.codebook_dim, out_dim)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encode_proj(z)                                   # (B, cd)
        flat_q = F.normalize(z_e, dim=-1) if self.l2_norm else z_e
        codebook = F.normalize(self.codebook.weight, dim=-1) if self.l2_norm else self.codebook.weight

        distance = flat_q.pow(2).sum(1, keepdim=True) - 2 * flat_q @ codebook.t() + codebook.pow(2).sum(1)
        weights = F.softmax(-distance / self.temperature, dim=1)    # (B, K), bounded logits
        z_q = weights @ codebook                                    # (B, cd)
        quantized = self.decode_proj(z_q)                           # (B, out_dim)

        # --- dual-entropy on a separate (sharper) temperature ---------------
        entropy_logits = -distance / self.entropy_temperature
        entropy_probs = F.softmax(entropy_logits, dim=1)
        entropy_log_probs = F.log_softmax(entropy_logits + 1e-5, dim=1)
        sample_entropy = -(entropy_probs * entropy_log_probs).sum(dim=1).mean()  # E_b[H(p)] = H(z|x)
        avg_probs = entropy_probs.mean(dim=0)                                    # (K,) local marginal
        # 训练时跨卡聚合得到全局边缘分布(保留 autograd);eval/单卡跳过避免死锁
        if self.training and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            avg_probs = dist_nn.all_reduce(avg_probs, op=dist.ReduceOp.SUM) / dist.get_world_size()
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()            # H(E_b[p]) = H(z)
        # 单系数双熵: ratio*(H(z|x) - H(z)) = -ratio*I(z;x)
        entropy_loss = self.entropy_loss_ratio * (sample_entropy - avg_entropy)

        return {
            "quantized":      quantized,
            "weights":        weights,
            "entropy_loss":   entropy_loss,
            "sample_entropy": sample_entropy,   # H(z|x)
            "avg_entropy":    avg_entropy,      # H(z)
            "max_prob":       weights.max(dim=1).values.mean().detach(),
        }


# ---------------------------------------------------------------------------
# VisionAutoencoder
# ---------------------------------------------------------------------------

class VisionAutoencoder(nn.Module):
    """Vision pair autoencoder with soft categorical latent and FDM decoder.

    Parameters
    ----------
    cfg : OmegaConf DictConfig (or attribute-accessible object) with keys:

        frame_encoder:
            type            : "cnn" | "dino"
            # cnn only:
            token_dim       : int   (e.g. 256)
            patch_size      : int   (e.g. 16)
            # dino only:
            pretrained_path : str   (local path to facebook/dinov2-* checkout)

        hidden_dim          : int   (transformer hidden size H)
        embedding_dim       : int   (latent embedding dim E fed to the decoder)
        codebook_size       : int   (number of codes K, align with action AE)
        codebook_dim        : int   (cosine-VQ code dim, default 32; low-dim+L2
                                     keeps codes spread on the unit sphere)
        quant_temperature   : float (soft assignment temperature, default 0.25)
        entropy_temperature : float (separate sharper temp for the dual-entropy
                                     regulariser, default 0.1)
        l2_norm             : bool  (L2-normalise encoder output & codebook so the
                                     assignment logits are bounded; default True)
        enc_layers          : int
        enc_heads           : int
        dec_layers          : int
        dec_heads           : int
        dropout             : float
        mi_beta             : float (single coefficient of the dual-entropy /
                                     I(z;x) regulariser; = entropy_loss_ratio)
        src_mask_ratio      : float (fraction of source tokens masked at decode)
        reconstruction_target : "feature" | "pixel"  (default "feature")
        adv_beta            : float (placeholder, currently unused; set 0)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        H  = int(cfg.hidden_dim)
        K  = int(cfg.codebook_size)
        E  = int(cfg.embedding_dim)
        recon_target = str(getattr(cfg, "reconstruction_target", "feature"))
        self.reconstruction_target = recon_target

        # ---- Frame encoder ------------------------------------------------
        self.frame_encoder, frame_dim = build_frame_encoder(cfg.frame_encoder)
        self.frame_to_hidden = nn.Linear(frame_dim, H)
        self.hidden_to_frame = nn.Linear(H, frame_dim)

        # ---- Delta encoder ------------------------------------------------
        # Type embeddings: distinguish src vs tgt tokens fed to encoder
        self.type_embed_src = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.type_embed_tgt = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        # Positional embedding is allocated for a max of 1024 tokens; actual
        # length is capped at runtime to the number of patch tokens N.
        self.enc_pos_embed = nn.Parameter(torch.randn(1, 1024, H) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = H,
            nhead           = int(cfg.enc_heads),
            dim_feedforward = H * 4,
            dropout         = float(cfg.dropout),
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.enc_layers))

        # ---- Soft vector quantizer (cosine, dual-entropy) -----------------
        # Replaces the unbounded "Linear(H,K)+softmax" categorical head: the
        # cosine assignment uses L2-normalised vectors so the logits are bounded
        # and the softmax cannot saturate → no collapse. Outputs an E-dim latent
        # so the downstream latent_proj(E→H) stays unchanged.
        self.quantizer = SoftVectorQuantizer(
            in_dim              = H,
            out_dim             = E,
            codebook_size       = K,
            codebook_dim        = int(getattr(cfg, "codebook_dim", 32)),
            temperature         = float(getattr(cfg, "quant_temperature", 0.25)),
            entropy_temperature = float(getattr(cfg, "entropy_temperature", 0.1)),
            entropy_loss_ratio  = float(getattr(cfg, "mi_beta", 0.01)),
            l2_norm             = bool(getattr(cfg, "l2_norm", True)),
        )

        # ---- Decoder (FDM, AdaLN-Zero) ------------------------------------
        self.latent_proj  = nn.Linear(E, H)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, 1024, H) * 0.02)

        # MAE-style source-token masking: at decode time a fraction of the
        # source patch tokens are replaced by a learnable mask token so the
        # decoder is forced to rely on the latent to infer the future frame
        # (breaks the trivial near-identity reconstruction shortcut).
        self.src_mask_ratio = float(getattr(cfg, "src_mask_ratio", 0.0))
        self.mask_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        self.decoder_blocks = nn.ModuleList([
            AdaLNDecoderBlock(
                hidden_dim = H,
                num_heads  = int(cfg.dec_heads),
                dropout    = float(cfg.dropout),
            )
            for _ in range(int(cfg.dec_layers))
        ])
        self.dec_norm = nn.LayerNorm(H)

        # ---- Reconstruction head ------------------------------------------
        if recon_target == "pixel":
            patch_size = int(getattr(cfg.frame_encoder, "patch_size", 16))
            self.output_head = None
            self.pixel_head  = PixelHead(token_dim=frame_dim, patch_size=patch_size)
        else:
            # feature: predict tgt frame tokens in original frame_dim space
            self.output_head = nn.Linear(H, frame_dim)
            self.pixel_head  = None

        # adv_beta placeholder (domain-adversarial not implemented in v1)
        self.adv_beta = float(getattr(cfg, "adv_beta", 0.0))

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # AdaLNDecoderBlock.ada_proj is re-zeroed in its own __init__,
            # but xavier above would overwrite it, so restore zero init:
            if isinstance(m, AdaLNDecoderBlock):
                nn.init.zeros_(m.ada_proj.weight)
                nn.init.zeros_(m.ada_proj.bias)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        frame_cur: torch.Tensor,
        frame_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Encode a frame pair into categorical weights and latent embedding.

        Parameters
        ----------
        frame_cur    : (B, 3, H, W)
        frame_future : (B, 3, H, W)

        Returns
        -------
        weights      : (B, K)      soft categorical weights (sums to 1)
        embedding    : (B, E)      quantized latent embedding
        q            : dict        quantizer outputs (entropy terms, max_prob)
        src_tokens_h : (B, N, H)   projected src patch tokens (for decoder)
        tgt_tokens   : (B, N, D_frame)  raw tgt patch tokens (for feature-mode loss)
        grid         : (h, w)      spatial grid of patch tokens
        """
        # --- frame encoding (shared encoder) --------------------------------
        src_tokens, grid = self.frame_encoder(frame_cur)    # (B,N,D_frame)
        tgt_tokens, _    = self.frame_encoder(frame_future) # (B,N,D_frame)

        src_h = self.frame_to_hidden(src_tokens)  # (B,N,H)
        tgt_h = self.frame_to_hidden(tgt_tokens)  # (B,N,H)

        B, N, H = src_h.shape

        # --- positional + type embeddings ----------------------------------
        pos = self.enc_pos_embed[:, :N]             # (1,N,H)
        src_in = src_h + pos + self.type_embed_src  # (B,N,H)
        tgt_in = tgt_h + pos + self.type_embed_tgt  # (B,N,H)

        cls = self.cls_token.expand(B, -1, -1)      # (B,1,H)
        x   = torch.cat([cls, src_in, tgt_in], dim=1)  # (B, 1+2N, H)

        # --- transformer encoder -------------------------------------------
        x       = self.encoder(x)                   # (B, 1+2N, H)
        cls_out = x[:, 0]                           # (B,H)

        # cosine soft-VQ: bounded logits → no saturation collapse
        q = self.quantizer(cls_out)
        weights   = q["weights"]                     # (B,K) soft assignment
        embedding = q["quantized"]                   # (B,E) latent for decoder

        return weights, embedding, q, src_h, tgt_tokens, grid

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _mask_source_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Randomly replace ``src_mask_ratio`` of tokens with the mask token.

        Per-sample random masking (different positions for each sample).
        Positional information is preserved because ``dec_pos_embed`` is added
        afterwards in :meth:`decode`.

        Parameters
        ----------
        tokens : (B, N, H)

        Returns
        -------
        (B, N, H) with a fraction of positions replaced by ``mask_token``.
        """
        B, N, _ = tokens.shape
        num_mask = int(round(N * self.src_mask_ratio))
        if num_mask <= 0:
            return tokens
        # smallest-noise positions are masked → uniform random subset per sample
        noise = torch.rand(B, N, device=tokens.device)
        mask_ids = noise.argsort(dim=1)[:, :num_mask]            # (B, num_mask)
        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        mask.scatter_(1, mask_ids, True)                         # (B, N)
        mask_tokens = self.mask_token.expand(B, N, -1)           # (B, N, H)
        return torch.where(mask.unsqueeze(-1), mask_tokens, tokens)

    def decode(
        self,
        embedding: torch.Tensor,
        src_tokens_h: torch.Tensor,
        grid: tuple[int, int],
    ) -> dict:
        """FDM decode: reconstruct target from source + latent.

        Parameters
        ----------
        embedding    : (B, E)
        src_tokens_h : (B, N, H)   projected src patch tokens
        grid         : (h, w)

        Returns
        -------
        dict with 'recon_tokens' (B,N,H) always, plus:
          'recon_frame_tokens' (B,N,D_frame)  for feature mode
          'recon_image'        (B,3,H,W)      for pixel mode
        """
        B, N, H = src_tokens_h.shape
        c = self.latent_proj(embedding)             # (B,H)

        # MAE-style masking (train only): drop a fraction of source tokens so
        # those positions carry no current-frame info; the decoder must use the
        # latent to reconstruct them. Mask token keeps positional embedding.
        query = src_tokens_h
        if self.training and self.src_mask_ratio > 0.0:
            query = self._mask_source_tokens(query)

        # Query: source tokens + positional embedding (latent injected via AdaLN)
        x = query + self.dec_pos_embed[:, :N]       # (B,N,H)

        for block in self.decoder_blocks:
            x = block(x, c)
        x = self.dec_norm(x)                        # (B,N,H)

        out: dict = {"recon_tokens": x}
        if self.reconstruction_target == "pixel":
            frame_tokens = self.hidden_to_frame(x)  # (B,N,D_frame)
            out["recon_frame_tokens"] = frame_tokens
            out["recon_image"] = self.pixel_head(frame_tokens, grid)
        else:
            out["recon_frame_tokens"] = self.output_head(x)  # (B,N,D_frame)

        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def entropy(weights: torch.Tensor) -> torch.Tensor:
        """Per-sample entropy of categorical weights, averaged over batch."""
        return -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

    def reconstruction_loss(
        self,
        dec_out: dict,
        frame_future: torch.Tensor,
        tgt_frame_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.reconstruction_target == "pixel":
            return F.mse_loss(dec_out["recon_image"], frame_future)
        else:
            return F.mse_loss(dec_out["recon_frame_tokens"], tgt_frame_tokens.detach())

    def compute_loss(
        self,
        dec_out: dict,
        frame_future: torch.Tensor,
        tgt_frame_tokens: torch.Tensor,
        q: dict,
    ) -> Dict[str, torch.Tensor]:
        recon = self.reconstruction_loss(dec_out, frame_future, tgt_frame_tokens)
        # 单系数双熵正则(由 quantizer 计算): entropy_loss = mi_beta*(H(z|x) - H(z))
        # total = recon + entropy_loss = recon - mi_beta * I(z; x)
        h_cond     = q["sample_entropy"]                              # H(z|x)
        h_marginal = q["avg_entropy"]                                 # H(z)
        mi_zx      = h_marginal - h_cond                              # I(z; x)
        total      = recon + q["entropy_loss"]
        return {
            "loss":       total,
            "recon":      recon.detach(),
            "h_cond":     h_cond.detach(),
            "h_marginal": h_marginal.detach(),
            "mi_zx":      mi_zx.detach(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_cur: torch.Tensor,
        frame_future: torch.Tensor,
        embodiment_ids: Optional[torch.Tensor] = None,  # placeholder for future adv
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → quantize → decode → loss.

        Parameters
        ----------
        frame_cur    : (B, 3, H, W) float, current frame
        frame_future : (B, 3, H, W) float, future frame at step i+k
        embodiment_ids : (B,) int | None  – reserved for domain-adversarial
                         (not used in v1; adv_beta=0)

        Returns
        -------
        dict with keys: weights, embedding, loss, recon, mi_zx, h_cond,
                        h_marginal, max_prob, recon_frame_tokens (always),
                        recon_image (pixel mode only)
        """
        weights, embedding, q, src_h, tgt_tokens, grid = self.encode(frame_cur, frame_future)

        dec_out = self.decode(embedding, src_h, grid)
        losses  = self.compute_loss(dec_out, frame_future, tgt_tokens, q)

        # adv placeholder: if adv_beta > 0 a GRL + embodiment classifier would go here
        # TODO(future): implement domain-adversarial when adv_beta > 0

        return {
            "weights":   weights,
            "embedding": embedding,
            "max_prob":  q["max_prob"],
            **dec_out,
            **losses,
        }


# ---------------------------------------------------------------------------
# Quick shape self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    import torch
    from omegaconf import OmegaConf

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

    # ------------------------------------------------------------------ CNN
    print("=" * 60)
    print("TEST 1: type=cnn, reconstruction_target=feature")
    print("=" * 60)
    cfg_cnn = OmegaConf.create({
        "frame_encoder": {"type": "cnn", "token_dim": 64, "patch_size": 16},
        "hidden_dim": 128,
        "embedding_dim": 128,
        "codebook_size": 64,
        "enc_layers": 2,
        "enc_heads": 4,
        "dec_layers": 2,
        "dec_heads": 4,
        "dropout": 0.0,
        "mi_beta": 0.001,
        "reconstruction_target": "feature",
        "adv_beta": 0.0,
    })
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionAutoencoder(cfg_cnn).to(device)
    B = 4
    fi  = torch.randn(B, 3, 224, 224, device=device)
    fik = torch.randn(B, 3, 224, 224, device=device)
    out = model(fi, fik)
    print(f"  weights     : {tuple(out['weights'].shape)}  sum={out['weights'][0].sum():.4f}")
    print(f"  embedding   : {tuple(out['embedding'].shape)}")
    print(f"  recon_frame : {tuple(out['recon_frame_tokens'].shape)}")
    print(f"  loss={out['loss'].item():.4f}  recon={out['recon'].item():.4f}  mi={out['mi_zx'].item():.4f}")
    out["loss"].backward()
    print("  [OK] backward passed")
    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n:,}")

    # ------------------------------------------------------------------ CNN pixel
    print()
    print("=" * 60)
    print("TEST 2: type=cnn, reconstruction_target=pixel")
    print("=" * 60)
    cfg_pix = OmegaConf.create({
        "frame_encoder": {"type": "cnn", "token_dim": 64, "patch_size": 16},
        "hidden_dim": 128,
        "embedding_dim": 128,
        "codebook_size": 64,
        "enc_layers": 2,
        "enc_heads": 4,
        "dec_layers": 2,
        "dec_heads": 4,
        "dropout": 0.0,
        "mi_beta": 0.001,
        "reconstruction_target": "pixel",
        "adv_beta": 0.0,
    })
    model2 = VisionAutoencoder(cfg_pix).to(device)
    out2 = model2(fi, fik)
    print(f"  recon_image  : {tuple(out2['recon_image'].shape)}")
    print(f"  loss={out2['loss'].item():.4f}")
    out2["loss"].backward()
    print("  [OK] backward passed")

    print()
    print("All shape tests PASSED.")
