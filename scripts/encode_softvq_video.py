#!/usr/bin/env python3
"""Encode a video into SoftVQ vision-only latent-action codes at a fixed stride."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_CKPT_PATH = (
    "/inspire/qb-ilm/project/qproject-fundationmodel/public/jjc/exp/tokenizer/"
    "0609_visonly_softvq_1token/checkpoints/partial_step_75000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--stride",
        default=16,
        type=int,
        help="Frame stride between encoded interval endpoints. Use 0 to encode each frame with itself as goal.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=DEFAULT_CKPT_PATH,
        type=Path,
        help=f"SoftVQ vision-only checkpoint path. Default: {DEFAULT_CKPT_PATH}",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        type=Path,
        help=(
            "Optional SoftVQ model config (.yaml) with a `model` section. "
            "When omitted, the built-in default architecture is used."
        ),
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the checkpoint state dict to match the model exactly. Default: false",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("decord", "pyav", "torchcodec", "torchvision_av"),
        help="Video decoder backend. Default: pyav",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Number of frame pairs encoded per forward pass.")
    parser.add_argument("--device", default="auto", help="Torch device. Default: auto")
    parser.add_argument(
        "--include-terminal-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the final video frame when stride offsets do not land on it. Default: true",
    )
    parser.add_argument(
        "--image-size",
        default=(256, 256),
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size passed to the SoftVQ tokenizer. Default: 256 256",
    )
    parser.add_argument(
        "--num-bridge-tokens",
        default=1,
        type=int,
        help="Expected number of SoftVQ delta tokens per interval. Default: 1",
    )
    parser.add_argument(
        "--num-codebooks",
        default=1,
        type=int,
        help="Expected number of SoftVQ codebooks. Default: 1",
    )
    parser.add_argument(
        "--codebook-size",
        default=64,
        type=int,
        help="Expected SoftVQ codebook size. Used for config/reporting. Default: 64",
    )
    parser.add_argument(
        "--strict-num-bridge-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require encoder output token count to equal --num-bridge-tokens. Default: true",
    )
    parser.add_argument(
        "--show-distribution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report the SoftVQ categorical distribution (max prob and entropy) per interval. Default: false",
    )
    parser.add_argument(
        "--token-format",
        default=None,
        help=(
            "Optional rendering for printed codes. Supports {i}, {codebook}, and {position}; "
            "for example '<softvq_cb{codebook}_{i}>'. Default: disabled"
        ),
    )
    return parser.parse_args()


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_config(
    ckpt_path: Path,
    config_path: Path | None,
    image_size: tuple[int, int],
    strict_load: bool,
    num_bridge_tokens: int,
    num_codebooks: int,
    codebook_size: int,
):
    from omegaconf import OmegaConf

    softvq_cfg: dict[str, Any] = {
        "ckpt_path": str(ckpt_path),
        "image_size": list(image_size),
        "strict_load": strict_load,
    }
    if config_path is not None:
        softvq_cfg["config_path"] = str(config_path)

    return OmegaConf.create(
        {
            "framework": {
                "latent_action": {
                    "backend": "softvq",
                    "softvq": softvq_cfg,
                    "num_bridge_tokens": num_bridge_tokens,
                    "num_codebooks": num_codebooks,
                    "codebook_size": codebook_size,
                }
            }
        }
    )


def make_offsets(num_frames: int, stride: int, include_terminal_frame: bool) -> list[int]:
    if stride < 0:
        raise ValueError(f"--stride must be non-negative, got {stride}.")
    if stride == 0:
        return list(range(num_frames))
    if num_frames < 2:
        raise ValueError(f"Video must contain at least 2 frames, got {num_frames}.")

    offsets = list(range(0, num_frames, stride))
    if include_terminal_frame and offsets[-1] != num_frames - 1:
        offsets.append(num_frames - 1)
    if len(offsets) < 2:
        raise ValueError(
            f"Only one sampled frame was produced from {num_frames} frames and stride={stride}; "
            "use a smaller stride or enable --include-terminal-frame."
        )
    return offsets


def make_pairs(offsets: list[int], stride: int) -> list[tuple[int, int]]:
    if stride == 0:
        return [(offset, offset) for offset in offsets]
    return list(zip(offsets[:-1], offsets[1:], strict=True))


def as_pil(frame: Any):
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def format_token_sequence(codes: list[list[int]], token_format: str | None) -> str:
    if token_format is None:
        return ""
    return "".join(
        token_format.format(i=int(code), codebook=codebook, position=position)
        for position, position_codes in enumerate(codes)
        for codebook, code in enumerate(position_codes)
    )


def normalize_weights(weights):
    if weights.ndim == 2:
        weights = weights.unsqueeze(1)
    if weights.ndim != 3:
        raise ValueError(f"Expected SoftVQ weights with 2 or 3 dims, got shape {tuple(weights.shape)}.")
    return weights


def encode_pairs(encoder, frames: Any, pairs: list[tuple[int, int]], batch_size: int):
    import torch

    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")

    encoded = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = [
                (as_pil(frames[left]), as_pil(frames[right]))
                for left, right in pairs[start : start + batch_size]
            ]
            encoded.append(normalize_weights(encoder.encode_distribution(batch_pairs).detach().cpu()))
    return torch.cat(encoded, dim=0)


def validate_shape(weights, expected_tokens: int, expected_codebooks: int, strict_num_bridge_tokens: bool) -> None:
    if expected_tokens <= 0:
        raise ValueError(f"--num-bridge-tokens must be positive, got {expected_tokens}.")
    if expected_codebooks <= 0:
        raise ValueError(f"--num-codebooks must be positive, got {expected_codebooks}.")

    actual_tokens = int(weights.shape[1])
    if strict_num_bridge_tokens and actual_tokens != expected_tokens:
        raise ValueError(
            f"SoftVQ tokenizer returned {actual_tokens} delta tokens, "
            f"but --num-bridge-tokens={expected_tokens}."
        )
    if not strict_num_bridge_tokens and actual_tokens < expected_tokens:
        raise ValueError(
            f"SoftVQ tokenizer returned {actual_tokens} delta tokens, "
            f"fewer than --num-bridge-tokens={expected_tokens}."
        )


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    ckpt_path = args.ckpt_path.expanduser().resolve()
    config_path = args.config_path.expanduser().resolve() if args.config_path is not None else None
    image_size = tuple(args.image_size)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"SoftVQ checkpoint not found: {ckpt_path}. Pass --ckpt-path if it is stored elsewhere."
        )
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(
            f"SoftVQ config not found: {config_path}. Pass --config-path with a valid .yaml or omit it."
        )

    import torch

    from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
    from starVLA.model.modules.latent_action import build_latent_action_encoder

    frames = get_all_frames(video_path.as_posix(), video_backend=args.video_backend)
    offsets = make_offsets(len(frames), args.stride, args.include_terminal_frame)
    pairs = make_pairs(offsets, args.stride)

    config = build_config(
        ckpt_path=ckpt_path,
        config_path=config_path,
        image_size=image_size,
        strict_load=args.strict_load,
        num_bridge_tokens=args.num_bridge_tokens,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
    )
    encoder = build_latent_action_encoder(config)
    encoder.to(resolve_device(args.device))
    encoder.eval()

    weights = encode_pairs(encoder, frames, pairs, args.batch_size)
    validate_shape(weights, args.num_bridge_tokens, args.num_codebooks, args.strict_num_bridge_tokens)
    weights = weights[:, : args.num_bridge_tokens]
    codes = weights.argmax(dim=-1).long().unsqueeze(-1)

    print(f"video: {video_path}")
    print(f"frames: {len(frames)}")
    print(f"stride: {args.stride}")
    print(f"sample_offsets: {offsets}")
    print(f"intervals: {len(pairs)}")
    print(f"codes_shape: {tuple(codes.shape)}")
    print(f"codes_per_interval: {codes.shape[1]}")
    print(f"codebooks: {codes.shape[2]}")
    print(f"codebook_size: {int(weights.shape[-1])}")
    print()

    for idx, ((left, right), code_tensor, weight_tensor) in enumerate(
        zip(pairs, codes, weights, strict=True), start=1
    ):
        code = [[int(value) for value in row] for row in code_tensor.tolist()]
        line = f"interval {idx:04d} | frames [{left}, {right}] | code {code}"
        tokens = format_token_sequence(code, args.token_format)
        if tokens:
            line = f"{line} | tokens {tokens}"
        if args.show_distribution:
            max_prob = weight_tensor.max(dim=-1).values
            entropy = -(weight_tensor * (weight_tensor + 1e-10).log()).sum(dim=-1)
            max_prob_str = ", ".join(f"{value:.4f}" for value in max_prob.tolist())
            entropy_str = ", ".join(f"{value:.4f}" for value in entropy.tolist())
            line = f"{line} | max_prob [{max_prob_str}] | entropy [{entropy_str}]"
        print(line)


if __name__ == "__main__":
    main()
