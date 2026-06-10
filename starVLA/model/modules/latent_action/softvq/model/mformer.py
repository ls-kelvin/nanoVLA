import torch
from torch import nn


class MFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.get("token_dim", config.get("hidden_size", 256)))
        depth = int(config.get("depth", 4))
        heads = int(config.get("num_heads", 8))
        mlp_ratio = float(config.get("mlp_ratio", 4.0))
        dropout = float(config.get("dropout", 0.0))
        max_seq_len = int(config.get("max_seq_len", 1024))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=heads,
            dim_feedforward=int(self.hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.sep = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, self.hidden_size))
        self.type_embed = nn.Embedding(4, self.hidden_size)
        nn.init.trunc_normal_(self.sep, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

    def forward(self, query: torch.Tensor, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bsz = query.shape[0]
        sep = self.sep.expand(bsz, -1, -1)
        parts = [query, cond, sep, target]
        x = torch.cat(parts, dim=1)
        type_ids = torch.cat([
            torch.zeros(query.shape[1], device=x.device, dtype=torch.long),
            torch.ones(cond.shape[1], device=x.device, dtype=torch.long),
            torch.full((1,), 2, device=x.device, dtype=torch.long),
            torch.full((target.shape[1],), 3, device=x.device, dtype=torch.long),
        ])
        x = x + self.pos[:, : x.shape[1]] + self.type_embed(type_ids)[None]
        return self.blocks(x)[:, : query.shape[1]]
