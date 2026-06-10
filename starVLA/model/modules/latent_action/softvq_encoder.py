from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder
from starVLA.model.modules.latent_action.softvq import VisOnly


DEFAULT_SOFTVQ_MODEL_CONFIG = {
    "visual_model": {
        "reconstruction_target": "pixel",
        "frame_encoder": {
            "type": "cnn",
            "image_size": [256, 256],
            "patch_size": 16,
            "token_dim": 256,
        },
        "delta_encoder": {
            "type": "mformer",
            "num_delta_tokens": 1,
            "token_dim": 1024,
            "depth": 16,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "max_seq_len": 1024,
        },
        "quantizer": {
            "enabled": True,
            "type": "softvq",
            "soft": True,
            "codebook_size": 64,
            "codebook_dim": 32,
            "l2_norm": True,
            "temperature": 0.25,
            "entropy_loss_ratio": 0.01,
            "entropy_temperature": 0.1,
        },
    }
}


def _get_softvq_model_config(softvq_cfg):
    inline_model_cfg = softvq_cfg.get("model", None)
    if inline_model_cfg is not None:
        return inline_model_cfg

    config_path = softvq_cfg.get("config_path", None)
    if config_path:
        if not Path(str(config_path)).exists():
            raise FileNotFoundError(f"SoftVQ config must be a local path: {config_path}")
        return OmegaConf.load(str(config_path)).model

    return OmegaConf.create(DEFAULT_SOFTVQ_MODEL_CONFIG)


class SoftVQLatentActionEncoder(BaseLatentActionEncoder):
    """Frozen vision-only SoftVQ wrapper returning categorical distributions."""

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        softvq_cfg = la_cfg.get("softvq", {})
        ckpt_path = softvq_cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError("framework.latent_action.softvq.ckpt_path is required for backend='softvq'.")
        if not Path(str(ckpt_path)).exists():
            raise FileNotFoundError(f"SoftVQ checkpoint must be a local path: {ckpt_path}")

        self.image_size = tuple(softvq_cfg.get("image_size", [256, 256]))
        self.model = VisOnly(_get_softvq_model_config(softvq_cfg))
        ckpt = self._load_checkpoint(str(ckpt_path))
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=bool(softvq_cfg.get("strict_load", False)))
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _load_checkpoint(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = to_pil_preserve(image).convert("RGB").resize(self.image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.inference_mode()
    def encode_distribution(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if not frame_pairs:
            return torch.empty((0, 0, 0), dtype=torch.float32, device=self.device)

        source_images = []
        target_images = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each SoftVQ frame pair to contain 2 frames, got {len(pair)}.")
            source_images.append(self._image_to_tensor(pair[0]))
            target_images.append(self._image_to_tensor(pair[1]))

        batch = {
            "source_image": torch.stack(source_images, dim=0).to(self.device),
            "target_image": torch.stack(target_images, dim=0).to(self.device),
        }
        self.model.eval()
        out = self.model(batch)
        if "weights" not in out:
            raise RuntimeError("SoftVQ model output does not contain `weights`.")
        return out["weights"].float()

    @torch.inference_mode()
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        return self.encode_distribution(frame_pairs, instructions).argmax(dim=-1).long()
