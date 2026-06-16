from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder


DEFAULT_SOFTVQ_CONFIG_PATH = Path(__file__).resolve().parent / "softvq_joint" / "0614_joint_aloha_xrecon_scratch.yaml"
DEFAULT_SOFTVQ_CKPT_PATH = (
    "/inspire/qb-ilm/project/qproject-fundationmodel/public/jjc/exp/softla/"
    "0614_joint_aloha_xrecon_scratch/checkpoints/partial_step_100000.pt"
)


class SoftVQLatentActionEncoder(BaseLatentActionEncoder):
    """Frozen JointTokenizer vision encoder returning ``weights_v`` distributions."""

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        softvq_cfg = la_cfg.get("softvq", {})

        config_path = Path(str(softvq_cfg.get("config_path", DEFAULT_SOFTVQ_CONFIG_PATH))).expanduser()
        ckpt_path = Path(str(softvq_cfg.get("ckpt_path", DEFAULT_SOFTVQ_CKPT_PATH))).expanduser()

        if not config_path.exists():
            raise FileNotFoundError(f"SoftVQ JointTokenizer config must be a local path: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SoftVQ JointTokenizer checkpoint must be a local path: {ckpt_path}")

        from starVLA.model.modules.latent_action.softvq_joint.joint_autoencoder import JointTokenizer

        cfg = OmegaConf.load(str(config_path))
        self.image_size = tuple(softvq_cfg.get("image_size", [224, 224]))
        self.model = JointTokenizer(cfg.model)
        ckpt = self._load_checkpoint(str(ckpt_path))
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=bool(softvq_cfg.get("strict_load", True)))
        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _load_checkpoint(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = to_pil_preserve(image).convert("RGB").resize(self.image_size, Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.inference_mode()
    def encode_distribution(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if not frame_pairs:
            return torch.empty((0, 1, 0), dtype=torch.float32, device=self.device)

        frame_cur = []
        frame_future = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each SoftVQ frame pair to contain 2 frames, got {len(pair)}.")
            frame_cur.append(self._image_to_tensor(pair[0]))
            frame_future.append(self._image_to_tensor(pair[1]))

        fc = torch.stack(frame_cur, dim=0).to(self.device)
        ff = torch.stack(frame_future, dim=0).to(self.device)
        self.model.eval()
        weights_v, _, _, _, _, _ = self.model.vision.encode(fc, ff)
        if weights_v.ndim != 2:
            raise RuntimeError(f"JointTokenizer vision.encode weights_v must be [B, K], got {tuple(weights_v.shape)}.")
        return weights_v.float().unsqueeze(1)

    @torch.inference_mode()
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        return self.encode_distribution(frame_pairs, instructions).argmax(dim=-1).long()
