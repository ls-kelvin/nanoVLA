import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.latent_action.softvq.model.delta_codec import MFormerDeltaDecoder, MFormerDeltaEncoder
from starVLA.model.modules.latent_action.softvq.model.frame_encoder import build_frame_encoder
from starVLA.model.modules.latent_action.softvq.model.soft_vq import SoftVectorQuantizer
from starVLA.model.modules.latent_action.softvq.model.vq import VectorQuantizer


class PixelHead(nn.Module):
    def __init__(self, token_dim: int, patch_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(token_dim, 3, kernel_size=patch_size, stride=patch_size), nn.Sigmoid())

    def forward(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        bsz, _, dim = tokens.shape
        x = tokens.transpose(1, 2).reshape(bsz, dim, *grid)
        return self.net(x)


class VisualVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reconstruction_target = str(config.get("reconstruction_target", "pixel"))
        self.frame_encoder = build_frame_encoder(config.frame_encoder)
        frame_dim = int(config.frame_encoder.get("token_dim", 256))
        hidden_dim = int(config.delta_encoder.get("token_dim", frame_dim))
        delta_cfg = dict(config.delta_encoder)
        delta_cfg["token_dim"] = hidden_dim
        quant_cfg = dict(config.quantizer)
        quant_cfg["dim"] = hidden_dim
        self.frame_to_hidden = nn.Identity() if frame_dim == hidden_dim else nn.Linear(frame_dim, hidden_dim)
        self.hidden_to_frame = nn.Identity() if frame_dim == hidden_dim else nn.Linear(hidden_dim, frame_dim)
        self.use_quantizer = bool(quant_cfg.get("enabled", True))
        self.delta_encoder = MFormerDeltaEncoder(delta_cfg)
        quantizer_cls = SoftVectorQuantizer if quant_cfg.get("type") == "softvq" or quant_cfg.get("soft") else VectorQuantizer
        self.quantizer = quantizer_cls(quant_cfg) if self.use_quantizer else None
        self.delta_decoder = MFormerDeltaDecoder(delta_cfg)
        self.pixel_head = PixelHead(frame_dim, int(config.frame_encoder.get("patch_size", 16)))

    def encode(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        src_frame_tokens, grid = self.frame_encoder(source)
        tgt_frame_tokens, _ = self.frame_encoder(target)
        src_tokens = self.frame_to_hidden(src_frame_tokens)
        tgt_tokens = self.frame_to_hidden(tgt_frame_tokens)
        delta = self.delta_encoder(src_tokens, tgt_tokens)
        vq = self.quantizer(delta) if self.use_quantizer else self._continuous(delta)
        return {
            **vq,
            "delta": delta,
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "src_frame_tokens": src_frame_tokens,
            "tgt_frame_tokens": tgt_frame_tokens,
            "grid": grid,
        }

    def _continuous(self, delta: torch.Tensor) -> dict:
        zero = delta.new_zeros(())
        return {
            "quantized": delta,
            "indices": torch.zeros(delta.shape[:-1], device=delta.device, dtype=torch.long),
            "loss": zero,
            "vq_loss": zero,
            "commit_loss": zero,
            "perplexity": zero,
            "dead_codes": zero,
        }

    def reconstruct(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        out = self.encode(source, target)
        rec_tokens = self.delta_decoder(out["src_tokens"], out["quantized"])
        out["recon_tokens"] = rec_tokens
        out["recon_frame_tokens"] = self.hidden_to_frame(rec_tokens)
        out["recon_image"] = self.pixel_head(out["recon_frame_tokens"], out["grid"])
        return out

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict:
        out = self.reconstruct(source, target)
        if self.reconstruction_target == "feature":
            recon_loss = F.mse_loss(out["recon_tokens"], out["tgt_tokens"].detach())
        else:
            recon_loss = F.mse_loss(out["recon_image"], target)
        out["recon_loss"] = recon_loss
        out["total_loss"] = recon_loss + out["loss"]
        return out
