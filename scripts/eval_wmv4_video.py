"""Sample dataset windows and render WMv4 world-model video predictions.

Loads a QwenWMv4_LA checkpoint with the WAN branch (load_wan=True), draws val
windows from the training data mix, generates future frames with
``predict_video``, and writes per-sample mp4 (top: prediction, bottom: ground
truth) plus a PNG grid for quick inspection.

Example:
  source .venv/bin/activate && source scripts/wan_runtime_env.sh
  .venv/bin/python scripts/eval_wmv4_video.py \
      --checkpoint results/Checkpoints2/<run_id>/checkpoints/steps_90000_pytorch_model.pt \
      --num-samples 4 --output-dir results/wmv4_video_eval
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from starVLA.dataloader.hdf5_la_dataset import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to steps_*_pytorch_model.pt")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <run_dir>/video_eval/<checkpoint_stem>",
    )
    parser.add_argument("--data-mix", default=None, help="Override datasets.vla_data.data_mix")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="WAN DiT denoise steps (default: checkpoint config wan.num_inference_steps)",
    )
    parser.add_argument("--fps", type=int, default=2, help="Playback fps for the mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-tries",
        type=int,
        default=512,
        help="Sampling budget when skipping horizon-padded windows",
    )
    return parser.parse_args()


def sample_windows(dataset, num_samples, seed, max_tries):
    """Draw random val windows, skipping ones whose future horizon is padded."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    samples, indices = [], []
    for idx in order[:max_tries]:
        sample = dataset[int(idx)]
        if sample.get("la_padded"):
            continue
        samples.append(sample)
        indices.append(int(idx))
        if len(samples) >= num_samples:
            break
    if not samples:
        raise RuntimeError(f"No usable (non-padded) windows found within {max_tries} draws.")
    return samples, indices


def _put_label(frame_bgr, text):
    cv2.putText(frame_bgr, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def save_sample_outputs(pred, gt, offsets, out_prefix, fps):
    """pred/gt: (T, H, W, 3) uint8 RGB. Writes mp4 (pred over GT) + PNG grid."""
    num_frames = pred.shape[0]
    labels = [f"t+{offsets[t]}" if t < len(offsets) else f"frame{t}" for t in range(num_frames)]
    frames = []
    for t in range(num_frames):
        top = _put_label(pred[t][:, :, ::-1].copy(), f"pred {labels[t]}")
        bottom = _put_label(gt[t][:, :, ::-1].copy(), f"gt   {labels[t]}")
        frames.append(np.concatenate([top, bottom], axis=0))
    mp4_path = f"{out_prefix}.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    grid = np.concatenate(
        [np.concatenate(list(pred), axis=1), np.concatenate(list(gt), axis=1)], axis=0
    )
    imageio.imsave(f"{out_prefix}_grid.png", grid)
    return mp4_path


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint).resolve()
    run_dir = ckpt.parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "video_eval" / ckpt.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg, _ = read_mode_config(str(ckpt))
    cfg = dict_to_namespace(model_cfg)
    if args.data_mix:
        cfg.datasets.vla_data.data_mix = args.data_mix

    logger.info("Building val dataset (data_mix=%s)", cfg.datasets.vla_data.data_mix)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        balance_dataset_weights=cfg.datasets.vla_data.get("balance_dataset_weights", False),
        balance_trajectory_weights=cfg.datasets.vla_data.get("balance_trajectory_weights", False),
        seed=int(cfg.get("seed", 42)),
    )
    samples, indices = sample_windows(dataset, args.num_samples, args.seed, args.max_tries)
    logger.info("Sampled %d windows: %s", len(samples), indices)

    logger.info("Loading framework from %s (load_wan=True)", ckpt)
    model = baseframework.from_pretrained(str(ckpt), load_wan=True, load_latent_action_encoder=False)
    model = model.to(torch.bfloat16).to(args.device).eval()

    videos = model.predict_video(examples=samples, num_inference_steps=args.num_inference_steps)[
        "video"
    ]

    for i, (sample, index) in enumerate(zip(samples, indices)):
        gt = np.stack([np.asarray(frame.convert("RGB")) for frame in sample["wm_frames"]])
        pred = videos[i]
        if gt.shape != pred.shape:
            raise ValueError(f"GT/pred shape mismatch: gt={gt.shape}, pred={pred.shape}")
        offsets = sample.get("wm_frame_offsets", [])
        out_prefix = out_dir / f"sample_{i:02d}_idx{index}"
        mp4_path = save_sample_outputs(pred, gt, offsets, out_prefix, args.fps)
        lang = str(sample.get("lang", ""))
        robot_type = str(sample.get("robot_type", ""))
        with open(f"{out_prefix}_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"dataset_index: {index}\nrobot_type: {robot_type}\nlang: {lang}\n")
        logger.info("sample %d (idx=%d, %s): %s -> %s", i, index, robot_type, lang, mp4_path)

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
