import torch
from torch import nn


class CNNFrameEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = int(config.get("patch_size", 16))
        self.token_dim = int(config.get("token_dim", 256))
        self.proj = nn.Conv2d(3, self.token_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(images)
        grid = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), grid


class ReservedFrameEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, images: torch.Tensor):
        raise NotImplementedError(f"frame_encoder '{self.name}' is reserved but not implemented yet")


def build_frame_encoder(config) -> nn.Module:
    name = str(config.get("type", "cnn"))
    if name == "cnn":
        return CNNFrameEncoder(config)
    if name in {"qwen", "dino", "wan_vae"}:
        return ReservedFrameEncoder(name)
    raise ValueError(f"unknown frame_encoder: {name}")
