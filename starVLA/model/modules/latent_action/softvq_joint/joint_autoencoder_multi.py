"""Multi-embodiment joint vision+action tokenizer (slim, encoder-free action branch).

Architecture
------------
Vision branch (shared, all embodiments):
    VisionAutoencoder : (frame_cur, frame_future) → weights_v (B,M,K), embedding_v (B,M,E)
    M = num_latent_tokens (config key ``vision.num_latent_tokens``, default 1).
    Each of the M slots is an independent learnable latent query whose output goes
    through the shared SoftVectorQuantizer → (B,M,K) soft weights & (B,M,E) embeddings.
    The M embeddings are prepended as prefix tokens to the vision FDM decoder.

Action branch (per-embodiment, decoder only):
    ActionDecoderBranch[emb] : codebook(K,E) + Transformer decoder
    No action encoder. Driven *entirely* by the vision→action cross-modal term:
        emb_va  = einsum("bmk,ke->bme", weights_v[emb_indices], codebook[emb])  (B_e, M, E)
        The M slot embeddings are prepended as prefix tokens to the action decoder.
        recon_va = MSE(decoder[emb](emb_va, state), action_delta)

    Gradients flow into: vision encoder/quantizer + action codebook[emb] + decoder[emb].
    The K-dim soft distribution is shared across embodiments; each embodiment learns
    what the same K visual change codes mean in its own action space.

Training objective
------------------
    total = w_vision * vision_loss
          + w_cross_va * Σ_emb (count_emb/total) * recon_va[emb]

MI / entropy: vision self-recon loss already includes the dual-entropy I(z;x)
regulariser from SoftVectorQuantizer (averaged over M slots).  No separate
action-side MI term is needed.

Collate convention
------------------
The trainer expects a batch dict with:
    frame_cur      : (B, 3, H, W)
    frame_future   : (B, 3, H, W)
    by_embodiment  : {emb: {"indices": LongTensor(B_e,),
                             "state":   (B_e, D),
                             "action_delta": (B_e, L, D),
                             "valid_mask":   (B_e, L) | None}}

Eval helpers
------------
    decode_action_from_vision(weights_v, state, embodiment)
        → recon_delta : (B, L, D)  in normalised delta space
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.model.modules.latent_action.softvq_joint.vision_autoencoder import VisionAutoencoder


# ---------------------------------------------------------------------------
# ActionDecoderBranch  (per-embodiment, no encoder)
# ---------------------------------------------------------------------------

class ActionDecoderBranch(nn.Module):
    """Slim action branch: codebook + Transformer decoder only.

    Sub-module names are kept identical to the *decoder* portion of
    ``ActionAutoencoder`` so that partial checkpoints from a pre-trained
    ActionAutoencoder can be loaded via shape-filtered ``load_state_dict``.

    Parameters
    ----------
    action_dim   : output action dimension D  (per-embodiment)
    state_dim    : state dimension S          (per-embodiment, defaults to action_dim)
    chunk_len    : action horizon L
    hidden_dim   : transformer hidden size H
    embedding_dim: latent embedding dim E  (must equal vision embedding_dim)
    codebook_size: number of codes K        (must equal vision codebook_size)
    dec_layers   : number of transformer layers
    dec_heads    : number of attention heads
    dropout      : dropout rate
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        H = int(cfg.hidden_dim)
        K = int(cfg.codebook_size)
        E = int(cfg.embedding_dim)
        L = int(cfg.chunk_len)
        D = int(cfg.action_dim)
        S = int(getattr(cfg, "state_dim", D))

        # ---- Codebook  (K, E) ------------------------------------------------
        # Named `codebook` to match ActionAutoencoder.codebook for ckpt compat.
        self.codebook = nn.Parameter(torch.randn(K, E) * 0.02)

        # ---- Decoder (identical names to ActionAutoencoder decoder) ----------
        self.latent_proj = nn.Linear(E, H)
        self.dec_state_proj = nn.Linear(S, H)
        self.dec_pos_queries = nn.Parameter(torch.randn(1, L, H) * 0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=int(cfg.dec_heads),
            dim_feedforward=H * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(cfg.dec_layers))
        self.output_head = nn.Linear(H, D)

        self._chunk_len = L
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, embedding: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Decode latent embeddings + state to action chunk.

        Parameters
        ----------
        embedding : (B, M, E)   M latent slot embeddings; M=1 is the original single-token case.
                    Also accepts (B, E) for backward compatibility (treated as M=1).
        state     : (B, S)

        Returns
        -------
        recon : (B, L, D)
        """
        B = embedding.shape[0]
        L = self._chunk_len

        # Support legacy (B, E) input
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)   # (B, 1, E)
        M = embedding.shape[1]

        # Project each slot to hidden dim; nn.Linear broadcasts over the M dim
        latent_h = self.latent_proj(embedding)                        # (B, M, H)
        queries  = self.dec_pos_queries[:, :L].expand(B, -1, -1)     # (B, L, H)
        st       = self.dec_state_proj(state).unsqueeze(1)            # (B, 1, H)
        # Sequence: [state_token | M latent prefix tokens | L action queries]
        x = torch.cat([st, latent_h, queries], dim=1)                 # (B, 1+M+L, H)
        x = self.decoder(x)[:, 1 + M:]                                # (B, L, H)
        return self.output_head(x)                                     # (B, L, D)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked MSE in normalised delta space.

        Parameters
        ----------
        recon, target : (B, L, D)
        valid_mask    : (B, L)  1.0 = real, 0.0 = padding
        """
        mse = (recon - target).pow(2).mean(dim=-1)   # (B, L)
        if valid_mask is not None:
            denom = valid_mask.sum().clamp(min=1.0)
            return (mse * valid_mask).sum() / denom
        return mse.mean()


# ---------------------------------------------------------------------------
# MultiJointTokenizer
# ---------------------------------------------------------------------------

class MultiJointTokenizer(nn.Module):
    """Multi-embodiment joint tokenizer.

    Parameters
    ----------
    cfg : OmegaConf DictConfig with sub-keys:
        vision : passed verbatim to VisionAutoencoder
        action :
            codebook_size  : int    (must align with vision)
            embedding_dim  : int    (must align with vision)
            chunk_len      : int
            hidden_dim     : int
            dec_layers     : int
            dec_heads      : int
            dropout        : float
            embodiment_dims: dict   {emb_name: action_dim}   (per-embodiment I/O dim)
        joint :
            w_vision   : float  vision self-recon weight  (default 1.0)
            w_cross_va : float  vision→action cross-recon weight (default 1.0)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # ---- Shared vision branch -----------------------------------------
        self.vision = VisionAutoencoder(cfg.vision)

        # ---- Per-embodiment action branches (decoder only) ----------------
        emb_dims: Dict[str, int] = dict(cfg.action.embodiment_dims)
        self.embodiments: List[str] = list(emb_dims.keys())

        # Per-embodiment chunk_len must match data.embodiment_stride exactly.
        # Falls back to a global cfg.action.chunk_len if the per-emb dict is absent.
        emb_chunk_lens: Dict[str, int] = {}
        if hasattr(cfg.action, "embodiment_chunk_lens") and cfg.action.embodiment_chunk_lens:
            emb_chunk_lens = {k: int(v) for k, v in cfg.action.embodiment_chunk_lens.items()}

        action_branches = {}
        for emb, action_dim in emb_dims.items():
            chunk_len = emb_chunk_lens.get(emb, int(getattr(cfg.action, "chunk_len", 16)))
            branch_cfg = OmegaConf.merge(
                cfg.action,
                OmegaConf.create({
                    "action_dim": action_dim,
                    "state_dim": action_dim,
                    "chunk_len": chunk_len,
                }),
            )
            action_branches[emb] = ActionDecoderBranch(branch_cfg)
        # ModuleDict key must not contain '.'; '-' is fine.
        self.action = nn.ModuleDict(action_branches)

        # ---- Loss weights -------------------------------------------------
        joint_cfg = cfg.joint
        self.w_vision    = float(getattr(joint_cfg, "w_vision",    1.0))
        self.w_cross_va  = float(getattr(joint_cfg, "w_cross_va",  1.0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_cur: torch.Tensor,
        frame_future: torch.Tensor,
        by_embodiment: Dict[str, dict],
    ) -> Dict[str, torch.Tensor]:
        """Joint forward over a mixed-embodiment batch.

        Parameters
        ----------
        frame_cur      : (B, 3, H, W)
        frame_future   : (B, 3, H, W)
        by_embodiment  : {emb: {"indices":     LongTensor(B_e,),
                                "state":        (B_e, D),
                                "action_delta": (B_e, L, D),
                                "valid_mask":   (B_e, L) | None}}

        Returns
        -------
        dict with keys:
            loss, vision_loss, recon_va,
            recon_v, mi_zx_v, h_cond_v, h_marginal_v,
            weights_v (B,M,K), embedding_v (B,M,E),
            by_embodiment_out : {emb: {"recon_va": scalar}},
            recon_image (pixel mode), recon_frame_tokens (feature mode),
            code_perplexity_v, code_usage_v, code_max_prob_v,
        """
        # ---- Vision self-recon (full batch) --------------------------------
        weights_v, embedding_v, q_v, src_h, tgt_tokens, grid = self.vision.encode(
            frame_cur, frame_future
        )
        dec_v    = self.vision.decode(embedding_v, src_h, grid)
        losses_v = self.vision.compute_loss(dec_v, frame_future, tgt_tokens, q_v)
        vision_loss = losses_v["loss"]

        # ---- Cross-modal: vision → action  (per embodiment) ----------------
        total_count = frame_cur.shape[0]
        recon_va_sum = vision_loss.new_zeros(())
        by_embodiment_out: Dict[str, dict] = {}

        for emb, group in by_embodiment.items():
            act: ActionDecoderBranch = self.action[emb]
            indices    = group["indices"]                          # (B_e,)
            state      = group["state"]                           # (B_e, D)
            act_delta  = group["action_delta"]                    # (B_e, L, D)
            valid_mask = group.get("valid_mask", None)            # (B_e, L) | None

            # Select vision weights for this embodiment's samples
            # weights_v : (B, M, K)  →  wv_e : (B_e, M, K)
            wv_e   = weights_v.index_select(0, indices)
            # Weighted sum over codebook for each slot → (B_e, M, E)
            emb_va = torch.einsum("bmk,ke->bme", wv_e, act.codebook)
            recon_pred = act.decode(emb_va, state)                # (B_e, L, D)
            recon_va_e = act.reconstruction_loss(recon_pred, act_delta, valid_mask)

            weight = indices.shape[0] / total_count
            recon_va_sum = recon_va_sum + recon_va_e * weight
            by_embodiment_out[emb] = {"recon_va": recon_va_e.detach()}

        # ---- Total loss ----------------------------------------------------
        total = self.w_vision * vision_loss + self.w_cross_va * recon_va_sum

        # ---- Code stats (detached) -----------------------------------------
        with torch.no_grad():
            # weights_v : (B, M, K) — flatten B*M for aggregate stats
            w      = weights_v.detach().float()
            w_flat = w.view(-1, w.shape[-1])                       # (B*M, K)
            counts = torch.bincount(w_flat.argmax(dim=-1), minlength=w_flat.shape[-1]).float()
            code_usage    = (counts > 0).sum().to(w)
            marginal      = w_flat.mean(0)                         # (K,)
            code_perplexity = (-(marginal * (marginal + 1e-8).log()).sum()).exp()

        result: Dict[str, torch.Tensor] = {
            "loss":          total,
            "vision_loss":   vision_loss,
            "recon_va":      recon_va_sum.detach() if not recon_va_sum.requires_grad else recon_va_sum,
            # vision diagnostics
            "recon_v":       losses_v["recon"],
            "mi_zx_v":       losses_v["mi_zx"],
            "h_cond_v":      losses_v["h_cond"],
            "h_marginal_v":  losses_v["h_marginal"],
            "weights_v":     weights_v,
            "embedding_v":   embedding_v,
            # code stats
            "code_perplexity_v": code_perplexity,
            "code_usage_v":      code_usage,
            "code_max_prob_v":   q_v["max_prob"],
            # per-embodiment breakdown
            "by_embodiment_out": by_embodiment_out,
        }

        # Attach vision reconstruction outputs for preview/log
        if "recon_image" in dec_v:
            result["recon_image"] = dec_v["recon_image"]
        if "recon_frame_tokens" in dec_v:
            result["recon_frame_tokens"] = dec_v["recon_frame_tokens"]

        return result

    # ------------------------------------------------------------------
    # Eval helpers  (no gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_action_from_vision(
        self,
        weights_vision: torch.Tensor,
        state: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        """Reconstruct action chunk using vision-branch distribution.

        Parameters
        ----------
        weights_vision : (B, M, K)   per-slot soft categorical weights from vision encoder
        state          : (B, D)      normalised current state (embodiment-specific D)
        embodiment     : str         key into self.action ModuleDict

        Returns
        -------
        recon_delta : (B, L, D)  in normalised delta space
        """
        act: ActionDecoderBranch = self.action[embodiment]
        # weights_vision : (B, M, K);  codebook : (K, E)  →  emb_va : (B, M, E)
        emb_va = torch.einsum("bmk,ke->bme", weights_vision, act.codebook)
        return act.decode(emb_va, state)             # (B, L, D)


__all__ = ["ActionDecoderBranch", "MultiJointTokenizer"]
