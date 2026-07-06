from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder
from starVLA.model.modules.latent_action.unit.model.gr00t_n1_tokenizer_unit_inference import GR00T_Tokenizer


class UniTVisualLatentActionEncoder(BaseLatentActionEncoder):
    """Frozen UniT/GR00T tokenizer wrapper for visual-only VQ targets."""

    def __init__(self, config) -> None:
        super().__init__()
        la_cfg = config.framework.get("latent_action", {})
        tokenizer_path = la_cfg.get("groot_tokenizer_path", None)
        if not tokenizer_path:
            raise ValueError("framework.latent_action.groot_tokenizer_path is required for backend='unit'.")
        if not Path(str(tokenizer_path)).exists():
            raise FileNotFoundError(f"UniT tokenizer checkpoint must be a local path: {tokenizer_path}")

        self.image_size = tuple(la_cfg.get("image_size", [224, 224]))
        dinov2_path_override = la_cfg.get("dinov2_path_override", None)
        if not dinov2_path_override:
            raise ValueError("framework.latent_action.dinov2_path_override is required for backend='unit'.")
        if not Path(str(dinov2_path_override)).exists():
            raise FileNotFoundError(f"UniT DINOv2 checkpoint must be a local path: {dinov2_path_override}")

        self.model = GR00T_Tokenizer.from_pretrained(
            str(tokenizer_path),
            dinov2_path_override=dinov2_path_override,
            tune_vision_model=False,
            tune_vision_m_former=False,
            tune_bridge_projector=False,
            tune_action_encoder=False,
            tune_fusion=False,
            tune_vq=False,
            tune_vision_decoder=False,
            tune_action_decoder_projector=False,
            tune_action_decoder_diffusion=False,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.model.vision_branch = torch.compile(self.model.vision_branch)
        self.model.fusion = torch.compile(self.model.fusion)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @staticmethod
    def imagenet_tensor_from_pil(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
        image = to_pil_preserve(image).convert("RGB").resize(image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _image_to_imagenet_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.imagenet_tensor_from_pil(image, self.image_size)

    def encode_tensors(self, obs_input: torch.Tensor, goal_input: torch.Tensor) -> torch.LongTensor:
        """Encode preprocessed observation/goal tensors shaped for UniT vision branch."""
        return self._encode_vq(obs_input, goal_input)

    def _is_multiview_pair(self, pair) -> bool:
        """Detect if a frame pair contains multi-view data (list of images per timestep)."""
        if len(pair) != 2:
            return False
        return isinstance(pair[0], (list, tuple))

    def _encode_vq(self, obs_input: torch.Tensor, goal_input: torch.Tensor) -> torch.LongTensor:
        """Run vision branch -> fusion -> VQ and return indices."""
        batch_size = obs_input.shape[0]
        self.model.eval()
        visual_tokens, _, _ = self.model.vision_branch(
            obs_input=obs_input,
            goal_input=goal_input,
            batch_size=batch_size,
        )
        dummy_action_tokens = torch.zeros_like(visual_tokens)
        pv = torch.ones((batch_size,), dtype=torch.long, device=visual_tokens.device)
        pa = torch.zeros((batch_size,), dtype=torch.long, device=visual_tokens.device)
        unit_tokens = self.model.fusion(
            visual_tokens=visual_tokens,
            action_tokens=dummy_action_tokens,
            pv=pv,
            pa=pa,
        )
        unit_tokens_down = self.model.vq_down_resampler(unit_tokens)
        _, vq_indices, _ = self.model.vq(unit_tokens_down)

        if vq_indices.ndim == 2:
            vq_indices = vq_indices.unsqueeze(-1)
        return vq_indices.long()

    @torch.no_grad()
    def encode(
        self,
        frame_pairs: Sequence[Sequence],
        instructions: Sequence[str] | None = None,
    ) -> torch.LongTensor:
        if not frame_pairs:
            return torch.empty((0, 0, 0), dtype=torch.long, device=self.device)

        multiview = self._is_multiview_pair(frame_pairs[0])
        if multiview:
            return self._encode_multiview(frame_pairs)
        return self._encode_single_view(frame_pairs)

    def _encode_single_view(self, frame_pairs: Sequence[Sequence[Image.Image]]) -> torch.LongTensor:
        obs_images = []
        goal_images = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each UniT frame pair to contain 2 frames, got {len(pair)}.")
            obs_images.append(self._image_to_imagenet_tensor(pair[0]))
            goal_images.append(self._image_to_imagenet_tensor(pair[1]))

        obs_input = torch.stack(obs_images, dim=0).unsqueeze(1).to(device=self.device, dtype=self.dtype)
        goal_input = torch.stack(goal_images, dim=0).unsqueeze(1).to(device=self.device, dtype=self.dtype)
        return self._encode_vq(obs_input, goal_input)

    def _encode_multiview(self, frame_pairs: Sequence[Sequence]) -> torch.LongTensor:
        """Encode multi-view frame pairs. Each pair is (obs_views, goal_views) where
        obs_views/goal_views are lists of V PIL images."""
        obs_batch = []
        goal_batch = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each UniT frame pair to contain 2 elements, got {len(pair)}.")
            obs_views, goal_views = pair
            if not isinstance(obs_views, (list, tuple)) or not isinstance(goal_views, (list, tuple)):
                raise ValueError("Multi-view frame pairs must contain lists of images per timestep.")
            if len(obs_views) != len(goal_views):
                raise ValueError(
                    f"obs and goal must have same number of views, got {len(obs_views)} vs {len(goal_views)}."
                )
            obs_tensors = [self._image_to_imagenet_tensor(img) for img in obs_views]
            goal_tensors = [self._image_to_imagenet_tensor(img) for img in goal_views]
            obs_batch.append(torch.stack(obs_tensors, dim=0))
            goal_batch.append(torch.stack(goal_tensors, dim=0))

        obs_input = torch.stack(obs_batch, dim=0).to(device=self.device, dtype=self.dtype)
        goal_input = torch.stack(goal_batch, dim=0).to(device=self.device, dtype=self.dtype)
        return self._encode_vq(obs_input, goal_input)
