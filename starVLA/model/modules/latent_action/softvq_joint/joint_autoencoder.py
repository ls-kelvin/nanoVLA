"""Joint vision+action tokenizer with cross-modal reconstruction.

Architecture
------------
Two independent branches share no parameters:
  - VisionAutoencoder  : (frame_cur, frame_future) → weights_v (B,K), embedding_v (B,E)
  - ActionAutoencoder  : (action_delta, state)      → weights_a (B,K), embedding_a (B,E)

Both use K=64 soft-categorical distributions over their own codebooks.

Training objective
------------------
    total = w_vision * vision_loss + w_action * action_loss
          + kl_beta  * KL_sym(weights_v, weights_a)    # set kl_beta=0 to disable
          + w_cross_av * recon_av                       # action→vision cross-recon
          + w_cross_va * recon_va                       # vision→action cross-recon  ← v2a target

Cross-modal reconstruction (WITH gradient through source encoder):
    action→vision:
        emb_av = vision_codebook_proj(weights_a @ normalize(vision_codebook))
        recon_av = MSE(vision_decoder(emb_av, src_h, grid), frame_future)
        Gradients flow into: action encoder + vision codebook/decoder.

    vision→action:
        emb_va = weights_v @ action_codebook
        recon_va = MSE(action_decoder(emb_va, state), action_delta)
        Gradients flow into: vision encoder + action codebook/decoder.
        This term directly optimises the v2a transfer metric.

By making z_act a causal prerequisite for the cross-modal decoder, neither
branch can collapse to a constant without incurring large cross-recon losses.

Cross-modal inference (no gradient, eval time)
----------------------------------------------
    decode_vision_from_action(weights_action, src_h, grid)
        → reconstructed future frame

    decode_action_from_vision(weights_vision, state)
        → reconstructed action chunk
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.latent_action.softvq_joint.action_autoencoder import ActionAutoencoder
from starVLA.model.modules.latent_action.softvq_joint.vision_autoencoder import VisionAutoencoder


# ---------------------------------------------------------------------------
# KL / JS helpers  (kept for optional use; set kl_beta=0 to disable)
# ---------------------------------------------------------------------------

def _sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def _js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q).clamp(min=eps)
    js_pq = 0.5 * (p * (p.log() - m.log())).sum(dim=-1)
    js_qp = 0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    return (js_pq + js_qp).mean()


# ---------------------------------------------------------------------------
# JointTokenizer
# ---------------------------------------------------------------------------

class JointTokenizer(nn.Module):
    """Joint vision + action tokenizer with cross-modal reconstruction.

    Parameters
    ----------
    cfg : OmegaConf DictConfig with sub-keys:
        vision      : passed verbatim to VisionAutoencoder
        action      : passed verbatim to ActionAutoencoder
        joint       :
            kl_beta    : float  KL alignment coefficient (set 0 to disable)
            kl_form    : str    "sym_kl" (default) | "js"
            w_vision   : float  weight on vision self-recon loss (default 1.0)
            w_action   : float  weight on action self-recon loss (default 1.0)
            w_cross_av : float  weight on action→vision cross-recon (default 1.0)
            w_cross_va : float  weight on vision→action cross-recon (default 1.0)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionAutoencoder(cfg.vision)
        self.action = ActionAutoencoder(cfg.action)

        joint_cfg = cfg.joint
        self.kl_beta     = float(getattr(joint_cfg, "kl_beta",     0.0))
        self.kl_form     = str(getattr(joint_cfg,   "kl_form",     "sym_kl"))
        self.w_vision    = float(getattr(joint_cfg, "w_vision",    1.0))
        self.w_action    = float(getattr(joint_cfg, "w_action",    1.0))
        self.w_cross_av  = float(getattr(joint_cfg, "w_cross_av",  1.0))
        self.w_cross_va  = float(getattr(joint_cfg, "w_cross_va",  1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alignment_loss(self, weights_v: torch.Tensor, weights_a: torch.Tensor) -> torch.Tensor:
        if self.kl_form == "js":
            return _js(weights_v, weights_a)
        return _sym_kl(weights_v, weights_a)

    def _vision_codebook(self) -> torch.Tensor:
        """Return (K, cd) vision codebook, L2-normalised if configured."""
        vq = self.vision.quantizer
        cb = vq.codebook.weight          # (K, cd)
        return F.normalize(cb, dim=-1) if vq.l2_norm else cb

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_cur: torch.Tensor,
        frame_future: torch.Tensor,
        action_delta: torch.Tensor,
        state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Joint forward: both self-recons + two cross-modal recons + optional KL.

        Parameters
        ----------
        frame_cur    : (B, 3, H, W)
        frame_future : (B, 3, H, W)
        action_delta : (B, L, D)   normalised delta (action_chunk - state)
        state        : (B, D)      normalised current state
        valid_mask   : (B, L) | None

        Returns
        -------
        dict keys:
            loss, vision_loss, action_loss, kl_align,
            recon_v, recon_a, recon_av, recon_va,
            mi_zx_v, mi_zx_a, h_cond_v, h_cond_a, h_marginal_v, h_marginal_a,
            weights_v, weights_a, embedding_v, embedding_a,
            max_prob_v, max_prob_a, argmax_agree,
            recon_action  (from self-recon, delta space),
            recon_action_va  (from cross-recon, delta space),
            recon_image  (pixel mode, from vision self-recon),
            recon_image_av  (pixel mode, from action→vision cross-recon),
        """
        # ---- Vision self-recon -----------------------------------------
        weights_v, embedding_v, q_v, src_h, tgt_tokens, grid = self.vision.encode(
            frame_cur, frame_future
        )
        dec_v    = self.vision.decode(embedding_v, src_h, grid)
        losses_v = self.vision.compute_loss(dec_v, frame_future, tgt_tokens, q_v)
        vision_loss = losses_v["loss"]

        # ---- Action self-recon -----------------------------------------
        weights_a, embedding_a, _ = self.action.encode(action_delta, state)
        recon_a_pred = self.action.decode(embedding_a, state)
        losses_a = self.action.compute_loss(recon_a_pred, action_delta, weights_a, valid_mask)
        action_loss = losses_a["loss"]

        # ---- Optional KL alignment  (kl_beta=0 → zero cost, no collapse risk)
        kl_align = self._alignment_loss(weights_v, weights_a)

        # ---- Cross-modal: action → vision  (weights_a drives vision decoder)
        # Gradient flows into: action encoder + vision codebook + vision decoder.
        cb_v  = self._vision_codebook()                       # (K, cd), normalised
        z_q_av = weights_a @ cb_v                             # (B, cd)
        emb_av = self.vision.quantizer.decode_proj(z_q_av)   # (B, E)
        dec_av = self.vision.decode(emb_av, src_h, grid)
        recon_av = self.vision.reconstruction_loss(dec_av, frame_future, tgt_tokens)

        # ---- Cross-modal: vision → action  (weights_v drives action decoder)
        # Gradient flows into: vision encoder + action codebook + action decoder.
        # This is the v2a training signal.
        emb_va       = weights_v @ self.action.codebook       # (B, E)
        recon_va_pred = self.action.decode(emb_va, state)
        recon_va = self.action.reconstruction_loss(recon_va_pred, action_delta, valid_mask)

        # ---- Total loss ------------------------------------------------
        total = (
            self.w_vision   * vision_loss
            + self.w_action * action_loss
            + self.kl_beta  * kl_align
            + self.w_cross_av * recon_av
            + self.w_cross_va * recon_va
        )

        # ---- Misc metrics ----------------------------------------------
        argmax_agree = (
            (weights_v.argmax(dim=-1) == weights_a.argmax(dim=-1))
            .float().mean()
        )

        result: Dict[str, torch.Tensor] = {
            "loss":           total,
            "vision_loss":    vision_loss,
            "action_loss":    action_loss,
            "kl_align":       kl_align,
            "recon_v":        losses_v["recon"],
            "recon_a":        losses_a["recon_loss"],
            "recon_av":       recon_av,
            "recon_va":       recon_va,
            "mi_zx_v":        losses_v["mi_zx"],
            "mi_zx_a":        losses_a["mi_zx"],
            "h_cond_v":       losses_v["h_cond"],
            "h_cond_a":       losses_a["h_cond"],
            "h_marginal_v":   losses_v["h_marginal"],
            "h_marginal_a":   losses_a["h_marginal"],
            "weights_v":      weights_v,
            "weights_a":      weights_a,
            "embedding_v":    embedding_v,
            "embedding_a":    embedding_a,
            "max_prob_v":     q_v["max_prob"],
            "max_prob_a":     weights_a.max(dim=-1).values.mean().detach(),
            "argmax_agree":   argmax_agree,
            "recon_action":   recon_a_pred.detach(),
            "recon_action_va": recon_va_pred.detach(),
        }
        # Vision pixel-mode outputs
        if "recon_image" in dec_v:
            result["recon_image"] = dec_v["recon_image"]
        if "recon_frame_tokens" in dec_v:
            result["recon_frame_tokens"] = dec_v["recon_frame_tokens"]
        # Cross action→vision pixel output
        if "recon_image" in dec_av:
            result["recon_image_av"] = dec_av["recon_image"]
        return result

    # ------------------------------------------------------------------
    # Cross-modal inference helpers  (no gradient, eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_vision_from_action(
        self,
        weights_action: torch.Tensor,
        src_h: torch.Tensor,
        grid: tuple,
    ) -> dict:
        """Reconstruct future frame using action-branch distribution (eval).

        Parameters
        ----------
        weights_action : (B, K)
        src_h          : (B, N, H)  projected source patch tokens
        grid           : (h, w)

        Returns
        -------
        dict from vision.decode
        """
        cb_v  = self._vision_codebook()
        z_q   = weights_action @ cb_v
        emb   = self.vision.quantizer.decode_proj(z_q)
        return self.vision.decode(emb, src_h, grid)

    @torch.no_grad()
    def decode_action_from_vision(
        self,
        weights_vision: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct action chunk using vision-branch distribution (eval).

        Parameters
        ----------
        weights_vision : (B, K)
        state          : (B, D)

        Returns
        -------
        recon_delta : (B, L, D)
        """
        emb = weights_vision @ self.action.codebook
        return self.action.decode(emb, state)


__all__ = ["JointTokenizer"]
