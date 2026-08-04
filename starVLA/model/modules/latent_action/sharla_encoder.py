import sys
from importlib import import_module
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder


SHARLA_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "sharla"
if str(SHARLA_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARLA_ROOT))

build_latent_action_tokenizer = import_module("sharla.model.tokenizer").build_latent_action_tokenizer


class SharlaLatentActionEncoder(BaseLatentActionEncoder):
    """Frozen Sharla tokenizer returning soft targets or quantized embeddings."""

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        sharla_cfg = la_cfg.get("sharla", {})
        config_path = sharla_cfg.get("config_path", None)
        ckpt_path = sharla_cfg.get("ckpt_path", None)
        if not config_path:
            raise ValueError("framework.latent_action.sharla.config_path is required for backend='sharla'.")
        if not ckpt_path:
            raise ValueError("framework.latent_action.sharla.ckpt_path is required for backend='sharla'.")

        config_path = Path(str(config_path)).expanduser()
        ckpt_path = Path(str(ckpt_path)).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Sharla tokenizer config must be a local file: {config_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Sharla tokenizer checkpoint must be a local file: {ckpt_path}")

        tokenizer_config = OmegaConf.load(config_path).model.tokenizer
        self.image_size = tuple(sharla_cfg.get("image_size", [224, 224]))
        self.model = build_latent_action_tokenizer(tokenizer_config)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_encoder.", "quantizer."))
        }
        # Partial ckpts omit frozen DINO frame_encoder; it is loaded from dinov2_path at build time.
        # Match sharla.model.unit_tokenizer.load_unit_teacher: allow missing non-trainable keys.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if bool(sharla_cfg.get("strict_load", True)):
            trainable = {name for name, param in self.model.named_parameters() if param.requires_grad}
            missing_trainable = trainable.intersection(missing)
            if missing_trainable or unexpected:
                raise RuntimeError(
                    "Failed to load Sharla tokenizer checkpoint: "
                    f"missing_trainable={sorted(missing_trainable)}, unexpected={sorted(unexpected)}"
                )
        self.model.requires_grad_(False)
        self.model.eval()
        self._enable_dinov2_sdpa()

    def _enable_dinov2_sdpa(self) -> None:
        """Switch frozen Dinov2 backbone to SDPA when the frame encoder is DINO."""
        frame_encoder = getattr(self.model.vision_encoder, "frame_encoder", None)
        dino = getattr(frame_encoder, "model", None)
        if dino is None:
            return
        if hasattr(dino, "set_attn_implementation"):
            dino.set_attn_implementation("sdpa")
            return
        from transformers import Dinov2Model

        name_or_path = getattr(dino.config, "_name_or_path", None)
        if not name_or_path:
            return
        replacement = Dinov2Model.from_pretrained(
            name_or_path,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        replacement.requires_grad_(False)
        frame_encoder.model = replacement

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def query_num(self) -> int:
        return int(self.model.vision_encoder.num_latent)

    @property
    def latent_dim(self) -> int:
        return int(self.model.quantizer.input_proj.out_features)

    @property
    def codebook_size(self) -> int:
        return int(self.model.quantizer.codebook.num_embeddings)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = to_pil_preserve(image)
        if not isinstance(image, Image.Image):
            raise TypeError("Sharla supports one image per timestep; configure a single latent-action video key.")
        image = image.convert("RGB").resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _prepare_vision_input(self, frame_pairs: Sequence[Sequence[Image.Image]]) -> dict[str, torch.Tensor]:
        frame_cur = []
        frame_future = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each Sharla frame pair to contain 2 frames, got {len(pair)}.")
            frame_cur.append(self._image_to_tensor(pair[0]))
            frame_future.append(self._image_to_tensor(pair[1]))
        return {
            "f0": torch.stack(frame_cur).to(device=self.device, dtype=self.dtype),
            "f1": torch.stack(frame_future).to(device=self.device, dtype=self.dtype),
        }

    @torch.inference_mode()
    def encode_distribution(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.Tensor:
        del instructions
        if not frame_pairs:
            return torch.empty((0, self.query_num, self.codebook_size), device=self.device)
        self.model.eval()
        return self.model(self._prepare_vision_input(frame_pairs)).float()

    @torch.inference_mode()
    def encode_distribution_from_tensors(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        fmid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode preprocessed RGB tensors ``[B, 3, H, W]`` in ``[0, 1]`` to soft weights.

        Optional ``fmid`` has shape ``[B, M, 3, H, W]`` for full-stride Perceiver
        teachers; omitted for two-frame teachers.
        """
        if f0.ndim != 4 or f1.ndim != 4:
            raise ValueError(f"Expected f0/f1 as [B,3,H,W], got {tuple(f0.shape)} / {tuple(f1.shape)}")
        if f0.shape != f1.shape:
            raise ValueError(f"f0/f1 shape mismatch: {tuple(f0.shape)} vs {tuple(f1.shape)}")
        if f0.shape[0] == 0:
            return torch.empty((0, self.query_num, self.codebook_size), device=self.device, dtype=torch.float32)
        if fmid is not None:
            if fmid.ndim != 5:
                raise ValueError(f"Expected fmid as [B,M,3,H,W], got {tuple(fmid.shape)}")
            if fmid.shape[0] != f0.shape[0] or fmid.shape[2:] != f0.shape[1:]:
                raise ValueError(
                    f"fmid shape {tuple(fmid.shape)} is incompatible with f0 {tuple(f0.shape)}"
                )
        self.model.eval()
        inputs = {
            "f0": f0.to(device=self.device, dtype=self.dtype, non_blocking=True),
            "f1": f1.to(device=self.device, dtype=self.dtype, non_blocking=True),
        }
        if fmid is not None:
            inputs["fmid"] = fmid.to(device=self.device, dtype=self.dtype, non_blocking=True)
        return self.model(inputs).float()

    @torch.inference_mode()
    def encode_continuous(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.Tensor:
        del instructions
        if not frame_pairs:
            return torch.empty((0, self.query_num, self.latent_dim), device=self.device, dtype=self.dtype)
        self.model.eval()
        latent, _, _ = self.model.vision_encoder(self._prepare_vision_input(frame_pairs))
        latent = F.normalize(self.model.quantizer.input_proj(latent), dim=-1)
        return latent

    @torch.inference_mode()
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        return self.encode_distribution(frame_pairs, instructions).argmax(dim=-1).long()
