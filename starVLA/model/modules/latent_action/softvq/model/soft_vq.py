import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = int(config.get("codebook_size", 256))
        self.dim = int(config.get("dim", config.get("token_dim", 1024)))
        self.codebook_dim = int(config.get("codebook_dim", self.dim))
        self.commitment_weight = float(config.get("commitment_weight", 0.25))
        self.l2_norm = bool(config.get("l2_norm", False))
        self.ema_update = bool(config.get("ema_update", True))
        self.ema_decay = float(config.get("ema_decay", 0.99))
        self.eps = float(config.get("eps", 1e-5))
        self.code_restart = bool(config.get("code_restart", True))
        self.restart_threshold = float(config.get("restart_threshold", 1.0))

        self.encode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.dim, self.codebook_dim)
        self.decode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.codebook_dim, self.dim)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.codebook.weight.requires_grad_(not self.ema_update)
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size), persistent=False)
        self.register_buffer("ema_embed", torch.zeros(self.codebook_size, self.codebook_dim), persistent=False)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)
        self.ema_embed.copy_(self.codebook.weight.data)

    def forward(self, z: torch.Tensor) -> dict:
        z_e = self.encode_proj(z)
        flat = z_e.reshape(-1, self.codebook_dim)
        flat_q = F.normalize(flat, dim=-1) if self.l2_norm else flat
        codebook = F.normalize(self.codebook.weight, dim=-1) if self.l2_norm else self.codebook.weight

        distance = flat_q.pow(2).sum(1, keepdim=True) - 2 * flat_q @ codebook.t() + codebook.pow(2).sum(1)
        weights = F.one_hot(distance.argmin(dim=1), self.codebook_size).type_as(flat)
        indices = weights.argmax(dim=1)
        z_q = (weights @ codebook).view_as(z_e)

        if self.training and self.ema_update:
            self._ema_update(flat_q, weights)

        vq_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        quantized = self.decode_proj(z_e + (z_q - z_e).detach())
        loss = self.commitment_weight * commit_loss if self.ema_update else vq_loss + self.commitment_weight * commit_loss

        entropy = -(weights * (weights + 1e-10).log()).sum(dim=1).mean()
        uniform_kl = torch.log(weights.new_tensor(self.codebook_size)) - entropy
        max_prob = weights.max(dim=1).values.mean()
        counts = weights.float().sum(0)
        probs = counts / counts.sum().clamp_min(1)
        batch_entropy = -(probs * (probs + 1e-10).log()).sum()
        entropy_loss = z_e.new_zeros(())
        perplexity = torch.exp(batch_entropy)
        dead_codes = (self.ema_cluster_size < self.restart_threshold).sum() if self.ema_update else (counts == 0).sum()
        return {
            "quantized": quantized,
            "indices": indices.view(z.shape[:-1]),
            "weights": weights.view(*z.shape[:-1], self.codebook_size),
            "loss": loss,
            "vq_loss": vq_loss,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
            "categorical_entropy": entropy,
            "uniform_kl": uniform_kl,
            "max_prob": max_prob,
            "entropy_loss": entropy_loss,
            "dead_codes": dead_codes,
        }

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, weights: torch.Tensor) -> None:
        weights = weights.type_as(flat)
        counts = weights.sum(0)
        embed_sum = weights.t() @ flat
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(embed_sum)

        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_embed.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
        self._restart_dead_codes(flat)

        n = self.ema_cluster_size.sum()
        size = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        weight = self.ema_embed / size.clamp_min(self.eps).unsqueeze(1)
        self.codebook.weight.data.copy_(F.normalize(weight, dim=-1) if self.l2_norm else weight)

    @torch.no_grad()
    def _restart_dead_codes(self, flat: torch.Tensor) -> None:
        if not self.code_restart:
            return
        dead = torch.nonzero(self.ema_cluster_size < self.restart_threshold).flatten()
        if dead.numel() == 0:
            return
        samples = flat[torch.randint(flat.shape[0], (dead.numel(),), device=flat.device)].to(self.ema_embed.dtype)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(samples, src=0)
        self.ema_embed[dead] = samples
        self.ema_cluster_size[dead] = self.restart_threshold


class SoftVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = int(config.get("codebook_size", 256))
        self.dim = int(config.get("dim", config.get("token_dim", 1024)))
        self.codebook_dim = int(config.get("codebook_dim", self.dim))
        self.l2_norm = bool(config.get("l2_norm", True))
        self.temperature = float(config.get("temperature", 0.07))
        self.entropy_loss_ratio = float(config.get("entropy_loss_ratio", 0.01))
        self.entropy_temperature = float(config.get("entropy_temperature", 0.01))

        self.encode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.dim, self.codebook_dim)
        self.decode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.codebook_dim, self.dim)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)

    def forward(self, z: torch.Tensor) -> dict:
        z_e = self.encode_proj(z)
        flat = z_e.reshape(-1, self.codebook_dim)
        flat_q = F.normalize(flat, dim=-1) if self.l2_norm else flat
        codebook = F.normalize(self.codebook.weight, dim=-1) if self.l2_norm else self.codebook.weight

        distance = flat_q.pow(2).sum(1, keepdim=True) - 2 * flat_q @ codebook.t() + codebook.pow(2).sum(1)
        weights = F.softmax(-distance / self.temperature, dim=1)
        indices = weights.argmax(dim=1)
        z_q = (weights @ codebook).view_as(z_e)
        quantized = self.decode_proj(z_q)

        entropy = -(weights * (weights + 1e-10).log()).sum(dim=1).mean()
        uniform_kl = torch.log(weights.new_tensor(self.codebook_size)) - entropy
        max_prob = weights.max(dim=1).values.mean()
        counts = weights.float().sum(0)
        probs = counts / counts.sum().clamp_min(1)
        batch_entropy = -(probs * (probs + 1e-10).log()).sum()

        entropy_logits = -distance / self.entropy_temperature
        entropy_probs = F.softmax(entropy_logits, dim=1)
        entropy_log_probs = F.log_softmax(entropy_logits + 1e-5, dim=1)
        sample_entropy = -(entropy_probs * entropy_log_probs).sum(dim=1).mean()
        avg_probs = entropy_probs.mean(dim=0)
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        entropy_loss = self.entropy_loss_ratio * (sample_entropy - avg_entropy)

        vq_loss = F.mse_loss(z_q.detach(), z_e.detach())
        dead_codes = (counts == 0).sum()
        return {
            "quantized": quantized,
            "indices": indices.view(z.shape[:-1]),
            "weights": weights.view(*z.shape[:-1], self.codebook_size),
            "loss": entropy_loss,
            "vq_loss": vq_loss,
            "commit_loss": vq_loss,
            "perplexity": torch.exp(batch_entropy),
            "categorical_entropy": entropy,
            "uniform_kl": uniform_kl,
            "max_prob": max_prob,
            "entropy_loss": entropy_loss,
            "dead_codes": dead_codes,
        }


def build_vq(config_model):
    config = dict(config_model.get("vq") or {})
    config["dim"] = int(config_model.hidden_size)
    return SoftVectorQuantizer(config) if config.get("soft", False) else VectorQuantizer(config)


__all__ = ["VectorQuantizer", "SoftVectorQuantizer", "build_vq"]
