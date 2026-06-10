#!/usr/bin/env python3
"""Encode a video into UniT/GR00T visual latent-action VQ codes at a fixed stride."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_TOKENIZER_PATH = (
    "/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/models/"
    "UniT/VLA-UniT-3B-fulldata/tokenizer"
)
DEFAULT_DINOV2_PATH = "/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/models/dinov2-large"


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
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        type=Path,
        help=f"UniT/GR00T tokenizer checkpoint directory. Default: {DEFAULT_TOKENIZER_PATH}",
    )
    parser.add_argument(
        "--dinov2-path",
        default=DEFAULT_DINOV2_PATH,
        type=Path,
        help=f"Local DINOv2 checkpoint path used by the UniT tokenizer. Default: {DEFAULT_DINOV2_PATH}",
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
        default=(224, 224),
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size passed to the UniT tokenizer. Default: 224 224",
    )
    parser.add_argument(
        "--num-bridge-tokens",
        default=8,
        type=int,
        help="Expected number of UniT VQ tokens per interval. Default: 8",
    )
    parser.add_argument(
        "--num-codebooks",
        default=2,
        type=int,
        help="Expected number of UniT VQ codebooks. Default: 2",
    )
    parser.add_argument(
        "--codebook-size",
        default=128,
        type=int,
        help="Expected UniT VQ codebook size. Used for config/reporting. Default: 128",
    )
    parser.add_argument(
        "--strict-num-bridge-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require encoder output token count to equal --num-bridge-tokens. Default: true",
    )
    parser.add_argument(
        "--token-format",
        default=None,
        help=(
            "Optional rendering for printed codes. Supports {i}, {codebook}, and {position}; "
            "for example '<unit_cb{codebook}_{i}>'. Default: disabled"
        ),
    )
    return parser.parse_args()


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_config(
    tokenizer_path: Path,
    dinov2_path: Path,
    image_size: tuple[int, int],
    num_bridge_tokens: int,
    num_codebooks: int,
    codebook_size: int,
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "framework": {
                "latent_action": {
                    "backend": "unit",
                    "groot_tokenizer_path": str(tokenizer_path),
                    "dinov2_path_override": str(dinov2_path),
                    "image_size": list(image_size),
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


def normalize_codes(codes):
    if codes.ndim == 2:
        codes = codes.unsqueeze(-1)
    if codes.ndim != 3:
        raise ValueError(f"Expected UniT codes with 2 or 3 dims, got shape {tuple(codes.shape)}.")
    return codes


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
            encoded.append(normalize_codes(encoder.encode(batch_pairs).detach().cpu()))
    return torch.cat(encoded, dim=0)


def validate_shape(codes, expected_tokens: int, expected_codebooks: int, strict_num_bridge_tokens: bool) -> None:
    if expected_tokens <= 0:
        raise ValueError(f"--num-bridge-tokens must be positive, got {expected_tokens}.")
    if expected_codebooks <= 0:
        raise ValueError(f"--num-codebooks must be positive, got {expected_codebooks}.")

    actual_tokens = int(codes.shape[1])
    actual_codebooks = int(codes.shape[2])
    if strict_num_bridge_tokens and actual_tokens != expected_tokens:
        raise ValueError(
            f"UniT tokenizer returned {actual_tokens} VQ tokens, "
            f"but --num-bridge-tokens={expected_tokens}."
        )
    if not strict_num_bridge_tokens and actual_tokens < expected_tokens:
        raise ValueError(
            f"UniT tokenizer returned {actual_tokens} VQ tokens, "
            f"fewer than --num-bridge-tokens={expected_tokens}."
        )
    if actual_codebooks != expected_codebooks:
        raise ValueError(
            f"UniT tokenizer returned {actual_codebooks} codebooks, "
            f"but --num-codebooks={expected_codebooks}."
        )


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    dinov2_path = args.dinov2_path.expanduser().resolve()
    image_size = tuple(args.image_size)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"UniT tokenizer checkpoint not found: {tokenizer_path}. "
            "Pass --tokenizer-path if it is stored elsewhere."
        )
    if not dinov2_path.exists():
        raise FileNotFoundError(
            f"UniT DINOv2 checkpoint not found: {dinov2_path}. Pass --dinov2-path if it is stored elsewhere."
        )

    from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
    from starVLA.model.modules.latent_action import build_latent_action_encoder

    frames = get_all_frames(video_path.as_posix(), video_backend=args.video_backend)
    offsets = make_offsets(len(frames), args.stride, args.include_terminal_frame)
    pairs = make_pairs(offsets, args.stride)

    config = build_config(
        tokenizer_path=tokenizer_path,
        dinov2_path=dinov2_path,
        image_size=image_size,
        num_bridge_tokens=args.num_bridge_tokens,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
    )
    encoder = build_latent_action_encoder(config)
    encoder.to(resolve_device(args.device))
    encoder.eval()

    codes = encode_pairs(encoder, frames, pairs, args.batch_size)
    validate_shape(codes, args.num_bridge_tokens, args.num_codebooks, args.strict_num_bridge_tokens)
    codes = codes[:, : args.num_bridge_tokens]

    print(f"video: {video_path}")
    print(f"frames: {len(frames)}")
    print(f"stride: {args.stride}")
    print(f"sample_offsets: {offsets}")
    print(f"intervals: {len(pairs)}")
    print(f"codes_shape: {tuple(codes.shape)}")
    print(f"codes_per_interval: {codes.shape[1]}")
    print(f"codebooks: {codes.shape[2]}")
    print()

    for idx, ((left, right), code_tensor) in enumerate(zip(pairs, codes, strict=True), start=1):
        code = [[int(value) for value in row] for row in code_tensor.tolist()]
        line = f"interval {idx:04d} | frames [{left}, {right}] | code {code}"
        tokens = format_token_sequence(code, args.token_format)
        if tokens:
            line = f"{line} | tokens {tokens}"
        print(line)


if __name__ == "__main__":
    main()
