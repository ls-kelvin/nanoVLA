from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder
from starVLA.model.modules.latent_action.univla.lam import ControllableDINOLatentActionModel


class UniVLALatentActionEncoder(BaseLatentActionEncoder):
    """Frozen UniVLA stage-2 LAM wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        univla_cfg = la_cfg.get("univla", {})
        ckpt_path = univla_cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError("framework.latent_action.univla.ckpt_path is required for backend='univla'.")

        self.image_size = tuple(univla_cfg.get("image_size", [224, 224]))
        self.model = ControllableDINOLatentActionModel(
            in_dim=3,
            model_dim=int(univla_cfg.get("model_dim", 768)),
            latent_dim=int(univla_cfg.get("latent_dim", 128)),
            num_latents=int(la_cfg.get("codebook_size", 16)),
            patch_size=int(univla_cfg.get("patch_size", 14)),
            enc_blocks=int(univla_cfg.get("enc_blocks", 12)),
            dec_blocks=int(univla_cfg.get("dec_blocks", 12)),
            num_heads=int(univla_cfg.get("num_heads", 12)),
            dropout=float(univla_cfg.get("dropout", 0.0)),
        )
        state = torch.load(Path(ckpt_path), map_location="cpu")
        state_dict = state.get("state_dict", state)
        state_dict = {k.replace("lam.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=bool(univla_cfg.get("strict_load", True)))
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.inference_mode()
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        if not frame_pairs:
            return torch.empty((0, 0), dtype=torch.long, device=self.device)

        videos = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each frame pair to contain 2 frames, got {len(pair)}.")
            videos.append(torch.stack([self._image_to_tensor(pair[0]), self._image_to_tensor(pair[1])], dim=0))
        videos = torch.stack(videos, dim=0).to(self.device)
        outputs = self.model.vq_encode(videos)
        indices = outputs["indices"].long()
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        return indices
