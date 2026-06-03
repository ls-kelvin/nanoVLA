#!/usr/bin/env python3
"""Encode a video into UniVLA latent-action codes at a fixed stride."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_CKPT_PATH = "./playground/Pretrained_models/UniVLA/lam-stage-2.ckpt"
DEFAULT_TOKEN_FORMAT = "<robot_action_{i}>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--stride", default=10, type=int, help="Frame stride between encoded interval endpoints.")
    parser.add_argument(
        "--ckpt-path",
        default=DEFAULT_CKPT_PATH,
        type=Path,
        help=f"UniVLA LAM checkpoint path. Default: {DEFAULT_CKPT_PATH}",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("decord", "pyav", "torchcodec", "torchvision_av"),
        help="Video decoder backend. Default: decord",
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
        "--token-format",
        default=DEFAULT_TOKEN_FORMAT,
        help=f"Token rendering for printed codes. Default: {DEFAULT_TOKEN_FORMAT}",
    )
    return parser.parse_args()


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_config(ckpt_path: Path, token_format: str):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "framework": {
                "latent_action": {
                    "backend": "univla",
                    "codebook_size": 16,
                    "token_format": token_format,
                    "univla": {
                        "ckpt_path": str(ckpt_path),
                    },
                }
            }
        }
    )


def make_offsets(num_frames: int, stride: int, include_terminal_frame: bool) -> list[int]:
    if stride <= 0:
        raise ValueError(f"--stride must be positive, got {stride}.")
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


def as_pil(frame: Any):
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def format_token_sequence(codes: list[int], token_format: str) -> str:
    return "".join(token_format.format(i=int(code)) for code in codes)


def encode_pairs(encoder, frames: Any, offsets: list[int], batch_size: int):
    import torch

    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}.")

    encoded = []
    pairs = list(zip(offsets[:-1], offsets[1:]))
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = [
                (as_pil(frames[left]), as_pil(frames[right]))
                for left, right in pairs[start : start + batch_size]
            ]
            encoded.append(encoder.encode(batch_pairs).detach().cpu())
    return torch.cat(encoded, dim=0)


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    ckpt_path = args.ckpt_path.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"UniVLA checkpoint not found: {ckpt_path}. Pass --ckpt-path if it is stored elsewhere."
        )

    from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
    from starVLA.model.modules.latent_action import build_latent_action_encoder

    frames = get_all_frames(video_path.as_posix(), video_backend=args.video_backend)
    offsets = make_offsets(len(frames), args.stride, args.include_terminal_frame)

    config = build_config(ckpt_path, args.token_format)
    encoder = build_latent_action_encoder(config)
    encoder.to(resolve_device(args.device))
    encoder.eval()

    codes = encode_pairs(encoder, frames, offsets, args.batch_size)

    print(f"video: {video_path}")
    print(f"frames: {len(frames)}")
    print(f"stride: {args.stride}")
    print(f"sample_offsets: {offsets}")
    print(f"intervals: {len(offsets) - 1}")
    print(f"codes_per_interval: {codes.shape[1] if codes.ndim > 1 else 1}")
    print()

    for idx, ((left, right), code_tensor) in enumerate(zip(zip(offsets[:-1], offsets[1:]), codes), start=1):
        code = [int(value) for value in code_tensor.reshape(-1).tolist()]
        print(
            f"interval {idx:04d} | frames [{left}, {right}] | "
            f"code {code} | tokens {format_token_sequence(code, args.token_format)}"
        )


if __name__ == "__main__":
    main()
