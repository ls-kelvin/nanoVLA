"""Action chunk autoencoder with soft categorical latent.

Architecture
------------
Encoder (Transformer):
    action_chunk (B, L, D) + state (B, S) →
    Linear → +pos_enc → prepend [CLS, state_token, delta_tokens] →
    TransformerEncoder → CLS output → Linear → softmax → weights (B, K)

Codebook:
    embedding = weights @ codebook  →  (B, E)

Decoder (Transformer):
    state (B, S) + position_queries + projected embedding →
    [state_token, queries] → TransformerEncoder → Linear → (B, L, D)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F


class ActionAutoencoder(nn.Module):
    """Action chunk autoencoder with soft categorical latent.

    Parameters
    ----------
    cfg : OmegaConf DictConfig (or any attribute-accessible object) with keys:
        action_dim, chunk_len, hidden_dim, embedding_dim, codebook_size,
        enc_layers, enc_heads, dec_layers, dec_heads, dropout, mi_beta,
        state_dim (optional, defaults to action_dim)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        K = cfg.codebook_size
        E = cfg.embedding_dim
        L = cfg.chunk_len
        D = cfg.action_dim
        S = getattr(cfg, "state_dim", D)

        # ---- Encoder ----
        self.input_proj = nn.Linear(D, H)
        self.enc_state_proj = nn.Linear(S, H)
        self.enc_pos_embed = nn.Parameter(torch.randn(1, L, H) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg.enc_heads,
            dim_feedforward=H * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.enc_layers)
        self.to_logits = nn.Linear(H, K)

        # ---- Codebook ----
        self.codebook = nn.Parameter(torch.randn(K, E) * 0.02)

        # ---- Decoder ----
        self.latent_proj = nn.Linear(E, H)
        self.dec_state_proj = nn.Linear(S, H)
        self.dec_pos_queries = nn.Parameter(torch.randn(1, L, H) * 0.02)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=cfg.dec_heads,
            dim_feedforward=H * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=cfg.dec_layers)
        self.output_head = nn.Linear(H, D)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self, action_chunk: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode action chunk into categorical weights and latent embedding.

        Parameters
        ----------
        action_chunk : (B, L, D)
        state : (B, S)

        Returns
        -------
        weights : (B, K)   soft categorical weights (sums to 1)
        embedding : (B, E) weighted-sum codebook embedding
        logits : (B, K)    raw logits before softmax
        """
        B, L, D = action_chunk.shape
        x = self.input_proj(action_chunk)  # (B, L, H)
        x = x + self.enc_pos_embed[:, :L]

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        st = self.enc_state_proj(state).unsqueeze(1)  # (B, 1, H)
        x = torch.cat([cls, st, x], dim=1)  # (B, 2+L, H)

        x = self.encoder(x)  # (B, 2+L, H)
        cls_out = x[:, 0]  # (B, H)

        logits = self.to_logits(cls_out)  # (B, K)
        weights = F.softmax(logits, dim=-1)  # (B, K)
        embedding = weights @ self.codebook  # (B, E)

        return weights, embedding, logits

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, embedding: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Decode latent embedding back to action chunk.

        Parameters
        ----------
        embedding : (B, E)
        state : (B, S)

        Returns
        -------
        recon : (B, L, D)
        """
        B = embedding.shape[0]
        L = self.cfg.chunk_len

        h = self.latent_proj(embedding)  # (B, H)
        queries = self.dec_pos_queries[:, :L].expand(B, -1, -1)  # (B, L, H)
        queries = queries + h.unsqueeze(1)  # (B, L, H)  broadcast add
        st = self.dec_state_proj(state).unsqueeze(1)  # (B, 1, H)
        x = torch.cat([st, queries], dim=1)  # (B, 1+L, H)

        x = self.decoder(x)[:, 1:]  # (B, L, H)  drop state token
        recon = self.output_head(x)  # (B, L, D)
        return recon

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked MSE loss.

        Parameters
        ----------
        recon, target : (B, L, D)
        valid_mask : (B, L)  1.0 for real steps, 0.0 for padding
        """
        mse = (recon - target).pow(2).mean(dim=-1)  # (B, L)
        if valid_mask is not None:
            denom = valid_mask.sum().clamp(min=1.0)
            return (mse * valid_mask).sum() / denom
        return mse.mean()

    @staticmethod
    def entropy(weights: torch.Tensor) -> torch.Tensor:
        """Per-sample entropy of categorical weights, averaged over batch.

        Parameters
        ----------
        weights : (B, K)
        """
        return -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

    def compute_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = self.reconstruction_loss(recon, target, valid_mask)
        # h_cond: 每样本熵均值——线性均值,DDP/DeepSpeed backward 自动跨卡平均梯度
        h_cond   = self.entropy(weights)                              # E_b[H(w)]
        # h_marginal: 边缘分布的非线性函数,需跨卡聚合得到全局边缘分布
        marginal = weights.mean(0)                                    # (K,) 本卡边缘
        # 仅训练时跨卡聚合;eval 只在单卡 forward,聚合会与其他卡的 barrier 撞死锁
        if self.training and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            marginal = dist_nn.all_reduce(marginal, op=dist.ReduceOp.SUM) / dist.get_world_size()
        h_marginal = -(marginal * (marginal + 1e-8).log()).sum()      # H(E_b[w])
        mi_zx  = h_marginal - h_cond                                  # I(z; x)
        total  = recon_loss - self.cfg.mi_beta * mi_zx
        return {
            "loss":       total,
            "recon_loss": recon_loss.detach(),
            "h_cond":     h_cond.detach(),
            "h_marginal": h_marginal.detach(),
            "mi_zx":      mi_zx.detach(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        action_chunk: torch.Tensor,
        state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → codebook lookup → decode → loss.

        Parameters
        ----------
        action_chunk : (B, L, D)
        state : (B, S)
        valid_mask : (B, L) or None

        Returns
        -------
        dict with keys: recon, weights, embedding, logits, loss, recon_loss, h_cond, h_marginal, mi_zx
        """
        weights, embedding, logits = self.encode(action_chunk, state)
        recon = self.decode(embedding, state)
        losses = self.compute_loss(recon, action_chunk, weights, valid_mask)
        return {
            "recon": recon,
            "weights": weights,
            "embedding": embedding,
            "logits": logits,
            **losses,
        }

