"""villa-x (IgorModel) latent action encoder for QwenMetaQuery_LA."""

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder

_VILLAX_DIR = str(Path(__file__).resolve().parents[4] / "third_party" / "villa-x")


class VillaXLatentActionEncoder(BaseLatentActionEncoder):
    """Frozen villa-x IgorModel wrapper for clip-based latent action encoding.

    Each clip of d_t frames produces (d_t - 1) latent actions, each represented
    by ``num_learned_tokens`` VQ codes from a codebook of size ``n_codes``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        villax_cfg = la_cfg.get("villax", {})

        ckpt_path = villax_cfg.get("ckpt_path", None)
        if not ckpt_path:
            raise ValueError("framework.latent_action.villax.ckpt_path is required for backend='villax'.")
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"villa-x checkpoint not found: {ckpt_path}")

        self.image_size = tuple(villax_cfg.get("image_size", [224, 224]))
        self.d_t = int(villax_cfg.get("d_t", 8))

        if _VILLAX_DIR not in sys.path:
            sys.path.insert(0, _VILLAX_DIR)
        from lam import IgorModel

        self.model: IgorModel = IgorModel.from_pretrained(ckpt_path)
        self.model.eval()
        self.model.requires_grad_(False)

        self._num_learned_tokens = self.model.config.num_learned_tokens
        self._n_codes = self.model.config.n_codes
        self._e_dim = self.model.config.action_latent_dim

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def num_learned_tokens(self) -> int:
        return self._num_learned_tokens

    @property
    def n_codes(self) -> int:
        return self._n_codes

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to [C, H, W] float32 tensor in [0, 255] range.

        villa-x's internal preprocess applies ImageNet normalization (divides
        by 255 then normalizes), so we keep the raw pixel scale here.
        """
        image = image.convert("RGB").resize(self.image_size)
        arr = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W], values in [0, 255]

    def _sample_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        """Uniformly sample d_t frames from a frame list using linspace."""
        n = len(frames)
        if n == self.d_t:
            return frames
        indices = np.linspace(0, n - 1, self.d_t).astype(int)
        return [frames[i] for i in indices]

    @torch.inference_mode()
    def encode_clips(self, clips: list[list[Image.Image]]) -> torch.LongTensor:
        """Encode a batch of clips into VQ indices.

        Args:
            clips: List of frame-lists, one per stride window.
                   Each frame-list may have variable length; it will be
                   uniformly sampled down to d_t frames.

        Returns:
            Tensor of shape ``[num_clips, d_t-1, num_learned_tokens]`` with
            VQ indices in ``[0, n_codes)``.
        """
        if not clips:
            return torch.empty((0, self.d_t - 1, self._num_learned_tokens), dtype=torch.long, device=self.device)

        tensors = []
        for clip_frames in clips:
            sampled = self._sample_frames(clip_frames)
            frame_tensors = torch.stack([self._image_to_tensor(f) for f in sampled], dim=0)  # [d_t, C, H, W]
            tensors.append(frame_tensors)

        batch = torch.stack(tensors, dim=0).to(device=self.device, dtype=self.dtype)  # [num_clips, d_t, C, H, W]

        result = self.model.idm(batch, return_dict=True)
        indices = result["indices"]  # flat: [num_clips * (d_t-1) * num_learned_tokens]

        num_clips = len(clips)
        transitions = self.d_t - 1
        indices = indices.reshape(num_clips, transitions, self._num_learned_tokens)
        return indices.long()

    @torch.inference_mode()
    def encode(
        self,
        frame_pairs: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        """Fallback pair-wise interface: treat each pair as a 2-frame clip.

        Returns:
            Tensor of shape ``[num_pairs, num_learned_tokens]`` with VQ indices.
        """
        if not frame_pairs:
            return torch.empty((0, self._num_learned_tokens), dtype=torch.long, device=self.device)

        clips = [list(pair) for pair in frame_pairs]

        tensors = []
        for pair in clips:
            if len(pair) != 2:
                raise ValueError(f"Expected 2 frames per pair, got {len(pair)}.")
            frame_tensors = torch.stack([self._image_to_tensor(f) for f in pair], dim=0)  # [2, C, H, W]
            tensors.append(frame_tensors)

        batch = torch.stack(tensors, dim=0).to(device=self.device, dtype=self.dtype)  # [num_pairs, 2, C, H, W]

        result = self.model.idm(batch, return_dict=True)
        indices = result["indices"]  # flat: [num_pairs * 1 * num_learned_tokens]

        num_pairs = len(frame_pairs)
        indices = indices.reshape(num_pairs, self._num_learned_tokens)
        return indices.long()
