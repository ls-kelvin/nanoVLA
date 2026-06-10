import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = int(config.get("codebook_size", 1024))
        self.dim = int(config.get("dim", config.get("token_dim", 256)))
        self.codebook_dim = int(config.get("codebook_dim", self.dim))
        self.commitment_weight = float(config.get("commitment_weight", 0.25))
        self.code_restart = bool(config.get("code_restart", True))
        self.restart_interval = int(config.get("restart_interval", 100))
        self.max_restart_steps = int(config.get("max_restart_steps", 50000))
        self.encode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.dim, self.codebook_dim)
        self.decode_proj = nn.Identity() if self.codebook_dim == self.dim else nn.Linear(self.codebook_dim, self.dim)
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer("usage", torch.zeros(self.codebook_size), persistent=False)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long), persistent=False)
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)

    def forward(self, z: torch.Tensor) -> dict:
        z_e = self.encode_proj(z)
        flat = z_e.reshape(-1, self.codebook_dim)
        self._restart_dead_codes(flat)
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1)
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).view_as(z_e)
        vq_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        quantized = self.decode_proj(z_e + (z_q - z_e).detach())
        counts = F.one_hot(indices, self.codebook_size).float().sum(0)
        if self.training and self.code_restart:
            self.usage += counts
        probs = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return {
            "quantized": quantized,
            "indices": indices.view(z.shape[:-1]),
            "loss": vq_loss + self.commitment_weight * commit_loss,
            "vq_loss": vq_loss,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
            "dead_codes": (counts == 0).sum(),
        }

    @torch.no_grad()
    def _restart_dead_codes(self, flat: torch.Tensor) -> None:
        if not (self.training and self.code_restart):
            return
        self.steps += 1
        if self.steps % self.restart_interval or self.steps > self.max_restart_steps:
            return
        usage = self.usage.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(usage)
        dead = torch.nonzero(usage == 0).flatten()
        if dead.numel():
            samples = flat[torch.randint(flat.shape[0], (dead.numel(),), device=flat.device)].to(self.codebook.weight.dtype)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(samples, src=0)
            self.codebook.weight[dead] = samples
        self.usage.zero_()
