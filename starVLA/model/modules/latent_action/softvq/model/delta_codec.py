import torch
from torch import nn

from starVLA.model.modules.latent_action.softvq.model.mformer import MFormer


class MFormerDeltaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_delta_tokens = int(config.get("num_delta_tokens", 8))
        dim = int(config.get("token_dim", config.get("hidden_size", 256)))
        self.query = nn.Parameter(torch.empty(1, self.num_delta_tokens, dim))
        self.mformer = MFormer(config)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(src_tokens.shape[0], -1, -1)
        return self.mformer(query, src_tokens, tgt_tokens)


class MFormerDeltaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = int(config.get("token_dim", config.get("hidden_size", 256)))
        self.query_bias = nn.Parameter(torch.zeros(1, 1, dim))
        self.mformer = MFormer(config)
        self.out = nn.LayerNorm(dim)

    def forward(self, src_tokens: torch.Tensor, delta_tokens: torch.Tensor) -> torch.Tensor:
        query = src_tokens + self.query_bias
        return self.out(self.mformer(query, src_tokens, delta_tokens))
