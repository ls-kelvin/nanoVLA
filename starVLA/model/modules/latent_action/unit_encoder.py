from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.latent_action.interface import BaseLatentActionEncoder
from starVLA.model.modules.latent_action.unit.model.gr00t_n1_tokenizer_unit_inference import (
    GR00T_Tokenizer,
    GR00T_Tokenizer_Config,
)


def _resolve_sharla_native_unit_paths(
    tokenizer_path: Path,
    config_path: str | None,
) -> tuple[Path, Path]:
    """Resolve (yaml config, weights) for Sharla-native UniT checkpoints.

    Sharla ``UnitTokenizer`` saves ``config.yaml`` (``model.unit``) plus a flat
    ``unit.safetensors`` whose keys are prefixed with ``unit.``.
    """
    if tokenizer_path.is_file():
        if tokenizer_path.suffix != ".safetensors":
            raise ValueError(
                "UniT tokenizer file checkpoints must be .safetensors, "
                f"got: {tokenizer_path}"
            )
        ckpt_path = tokenizer_path
        yaml_path = Path(config_path).expanduser() if config_path else ckpt_path.parent / "config.yaml"
    elif tokenizer_path.is_dir():
        yaml_path = (
            Path(config_path).expanduser()
            if config_path
            else tokenizer_path / "config.yaml"
        )
        if (tokenizer_path / "unit.safetensors").is_file():
            ckpt_path = tokenizer_path / "unit.safetensors"
        elif (tokenizer_path / "config.json").is_file():
            raise ValueError(
                f"Expected HuggingFace UniT checkpoint directory, got Sharla-style dir "
                f"without unit.safetensors: {tokenizer_path}"
            )
        else:
            raise FileNotFoundError(
                f"UniT tokenizer directory must contain unit.safetensors or config.json: "
                f"{tokenizer_path}"
            )
    else:
        raise FileNotFoundError(f"UniT tokenizer checkpoint path does not exist: {tokenizer_path}")

    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"Sharla-native UniT checkpoint requires config.yaml at {yaml_path}. "
            "Set framework.latent_action.unit.config_path if it lives elsewhere."
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"UniT tokenizer weights not found: {ckpt_path}")
    return yaml_path, ckpt_path


def _is_sharla_native_unit_checkpoint(tokenizer_path: Path, config_path: str | None) -> bool:
    if config_path:
        return True
    if tokenizer_path.is_file() and tokenizer_path.suffix == ".safetensors":
        return True
    if tokenizer_path.is_dir() and (tokenizer_path / "config.yaml").is_file():
        return not (tokenizer_path / "config.json").is_file()
    return False


def _load_sharla_native_unit_tokenizer(
    tokenizer_path: Path,
    *,
    config_path: str | None,
    dinov2_path_override: str,
    strict_load: bool,
) -> GR00T_Tokenizer:
    yaml_path, ckpt_path = _resolve_sharla_native_unit_paths(tokenizer_path, config_path)
    raw_cfg = OmegaConf.load(yaml_path)
    if not hasattr(raw_cfg, "model") or not hasattr(raw_cfg.model, "unit"):
        raise ValueError(
            f"Sharla-native UniT config must define model.unit, got: {yaml_path}"
        )

    unit_cfg = OmegaConf.to_container(raw_cfg.model.unit, resolve=True)
    unit_cfg["backbone_cfg"]["dinov2_path"] = str(dinov2_path_override)
    model_config = GR00T_Tokenizer_Config(**unit_cfg)
    model = GR00T_Tokenizer(
        model_config,
        dinov2_path_override=dinov2_path_override,
        unified_embodiment_id=unit_cfg.get("unified_embodiment_id"),
    )
    model.set_trainable_parameters(
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

    state_dict = load_file(str(ckpt_path))
    unit_state = {
        key[len("unit.") :]: value
        for key, value in state_dict.items()
        if key.startswith("unit.")
    }
    if not unit_state:
        raise ValueError(
            f"No 'unit.' prefixed weights found in {ckpt_path}. "
            "Expected a Sharla UnitTokenizer safetensors checkpoint."
        )

    missing, unexpected = model.load_state_dict(unit_state, strict=strict_load)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load Sharla-native UniT weights from {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    print(f"Loaded Sharla-native UniT tokenizer from {ckpt_path} (config: {yaml_path})")
    return model


def _load_unit_tokenizer(
    tokenizer_path: str,
    *,
    config_path: str | None,
    dinov2_path_override: str,
    strict_load: bool,
) -> GR00T_Tokenizer:
    path = Path(str(tokenizer_path)).expanduser()
    if _is_sharla_native_unit_checkpoint(path, config_path):
        return _load_sharla_native_unit_tokenizer(
            path,
            config_path=config_path,
            dinov2_path_override=dinov2_path_override,
            strict_load=strict_load,
        )

    return GR00T_Tokenizer.from_pretrained(
        str(path),
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

        unit_cfg = la_cfg.get("unit", {})
        self.model = _load_unit_tokenizer(
            str(tokenizer_path),
            config_path=unit_cfg.get("config_path", None),
            dinov2_path_override=str(dinov2_path_override),
            strict_load=bool(unit_cfg.get("strict_load", True)),
        )
        self.model.eval()
        self.model.requires_grad_(False)
        
    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def query_num(self) -> int:
        """Number of UniT motion query tokens per frame pair (== VQ tokens)."""
        return int(self.model.query_num)

    @property
    def latent_dim(self) -> int:
        """Continuous latent dimension before quantization (VQ ``e_dim``)."""
        return int(self.model.config.vq_cfg["e_dim"])

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

    def _fuse_unit_tokens_down(self, obs_input: torch.Tensor, goal_input: torch.Tensor) -> torch.Tensor:
        """Run vision branch -> fusion -> vq_down_resampler; return continuous pre-VQ tokens.

        Returns tensor shaped ``[B, query_num, e_dim]`` (the ``before_quant`` embedding).
        """
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
        return self.model.vq_down_resampler(unit_tokens)

    def _encode_vq(self, obs_input: torch.Tensor, goal_input: torch.Tensor) -> torch.LongTensor:
        """Run vision branch -> fusion -> VQ and return indices."""
        unit_tokens_down = self._fuse_unit_tokens_down(obs_input, goal_input)
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

    def _build_single_view_tensors(
        self, frame_pairs: Sequence[Sequence[Image.Image]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_images = []
        goal_images = []
        for pair in frame_pairs:
            if len(pair) != 2:
                raise ValueError(f"Expected each UniT frame pair to contain 2 frames, got {len(pair)}.")
            obs_images.append(self._image_to_imagenet_tensor(pair[0]))
            goal_images.append(self._image_to_imagenet_tensor(pair[1]))

        obs_input = torch.stack(obs_images, dim=0).unsqueeze(1).to(device=self.device, dtype=self.dtype)
        goal_input = torch.stack(goal_images, dim=0).unsqueeze(1).to(device=self.device, dtype=self.dtype)
        return obs_input, goal_input

    def _build_multiview_tensors(self, frame_pairs: Sequence[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build multi-view frame-pair tensors. Each pair is (obs_views, goal_views) where
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
        return obs_input, goal_input

    def _prepare_pair_tensors(self, frame_pairs: Sequence[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        if self._is_multiview_pair(frame_pairs[0]):
            return self._build_multiview_tensors(frame_pairs)
        return self._build_single_view_tensors(frame_pairs)

    def _encode_single_view(self, frame_pairs: Sequence[Sequence[Image.Image]]) -> torch.LongTensor:
        obs_input, goal_input = self._build_single_view_tensors(frame_pairs)
        return self._encode_vq(obs_input, goal_input)

    def _encode_multiview(self, frame_pairs: Sequence[Sequence]) -> torch.LongTensor:
        obs_input, goal_input = self._build_multiview_tensors(frame_pairs)
        return self._encode_vq(obs_input, goal_input)

    @torch.no_grad()
    def encode_continuous(
        self,
        frame_pairs: Sequence[Sequence],
        instructions: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Encode frame pairs into continuous pre-quantization embeddings.

        Returns tensor shaped ``[num_pairs, query_num, e_dim]`` (the UniT
        ``before_quant`` / ``unit_tokens_down`` motion embedding).
        """
        if not frame_pairs:
            return torch.empty((0, 0, 0), dtype=self.dtype, device=self.device)
        obs_input, goal_input = self._prepare_pair_tensors(frame_pairs)
        return self._fuse_unit_tokens_down(obs_input, goal_input)
