"""Prewarm the exact WAN video windows without loading the VLM or DiT.

Example (the model path can instead be set in the YAML):
  .venv/bin/python scripts/cache_wan_vae.py --config-yaml CONFIG \
      --wan-model-path /path/to/Wan2.1-T2V-1.3B-Diffusers

Supports torchrun: shards complete batches across ranks. Existing entries are
reused; incomplete/corrupt batches are encoded at their original stream batch
size unless --encode-batch-size groups them into larger VAE calls. Cache keys
still match the training batch size. Larger calls can change BF16 rounding.
Run once per training batch size. Training also fills misses online.
"""

import argparse
import contextlib
import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import AutoencoderKLWan
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.hdf5_dataset import _collect_mixture_entries
from starVLA.dataloader.hdf5_la_dataset import make_HDF5LatentActionSingleDataset
from starVLA.dataloader.lerobot_la_datasets import _resolve_latent_action_horizon
from starVLA.dataloader.wan_frame_collator import WanFrameCollator
from starVLA.model.modules.wan.latent_cache import WanLatentCache


def identity(batch):
    return batch


class WanWindows(Dataset):
    def __init__(self, dataset, fallback_horizon):
        self.dataset = dataset
        self.robot_type = dataset.lerobot_info_meta.get("robot_type")
        self.horizon = _resolve_latent_action_horizon(
            dataset.data_cfg.latent_action, self.robot_type, fallback_horizon
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        trajectory, step = self.dataset.all_steps[index]
        sample = {}
        self.dataset._maybe_pack_wm_frames(sample, int(trajectory), int(step), self.robot_type, self.horizon)
        return sample


class ShardedBatches(Sampler):
    def __init__(self, size, batch_size, rank, world):
        self.size, self.batch_size, self.rank, self.world = size, batch_size, rank, world

    def __iter__(self):
        for start in range(self.rank * self.batch_size, self.size, self.world * self.batch_size):
            # Preserve VAE convolution batch size even for the final windows.
            yield [min(start + j, self.size - 1) for j in range(self.batch_size)]

    def __len__(self):
        batches = (self.size + self.batch_size - 1) // self.batch_size
        return len(range(self.rank, batches, self.world))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--wan-model-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--batch-size", type=int, help="Override both streams to match training CLI overrides")
    parser.add_argument("--latent-batch-size", type=int, help="Optional separate latent stream batch size")
    parser.add_argument("--encode-batch-size", type=int,
                        help="VAE compute batch only; cache keys still use training batch size")
    parser.add_argument("--dataset", help="Optional dataset-name substring for a small prewarm")
    parser.add_argument("--max-batches", type=int, help="Bound total processed batches per rank")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if any(size is not None and size < 1 for size in (args.batch_size, args.latent_batch_size, args.encode_batch_size)):
        parser.error("batch sizes must be positive")
    cfg = OmegaConf.load(args.config_yaml)
    wan = cfg.framework.wan
    model_path = args.wan_model_path or wan.get("wan_model_path")
    if not model_path:
        parser.error("Set --wan-model-path or framework.wan.wan_model_path")
    cache_cfg = dict(wan.get("vae_cache") or {})
    cache_cfg.update(enabled=True, write=True)
    if args.cache_dir:
        cache_cfg["dir"] = args.cache_dir
    cache = WanLatentCache.from_config(model_path, cache_cfg)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    from diffusers.utils import logging as diffusers_logging
    diffusers_logging.disable_progress_bar()
    diffusers_logging.set_verbosity_error()
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.cuda().eval().requires_grad_(False)
    data = cfg.datasets.vla_data
    if not data.get("wan", {}).get("enabled", False):
        parser.error("datasets.vla_data.wan.enabled must be true")
    action_batch = args.batch_size or int(data.per_device_batch_size)
    latent_batch = args.latent_batch_size or args.batch_size or int(data.get("latent_per_device_batch_size", data.per_device_batch_size))
    streams = [(data.data_mix, action_batch)]
    if cfg.trainer.get("use_dual_vla_dataloaders", False):
        streams.append((data.latent_data_mix, latent_batch))
    encode_batch_size = args.encode_batch_size or cache_cfg.get("encode_batch_size")
    if encode_batch_size is not None:
        encode_batch_size = int(encode_batch_size)
        if encode_batch_size < 1 or any(encode_batch_size % size for _, size in streams):
            parser.error("encode batch size must be a positive multiple of each training stream batch size")
    progress_group = None
    if world > 1:
        # Only small CPU progress counters are communicated, never VAE tensors.
        # Gloo prints native startup banners to stdout; errors remain on stderr.
        saved_stdout = os.dup(1)
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), 1)
                if torch.distributed.is_initialized():
                    progress_group = torch.distributed.new_group(backend="gloo")
                else:
                    torch.distributed.init_process_group("gloo")
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)

    def load_dataset(name, robot):
        # Dataset constructors print one banner each; keep the sole total bar.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            return make_HDF5LatentActionSingleDataset(
                data.data_root_dir, name, robot, data_cfg=data, cache_mode="load")

    plan = []
    if rank == 0:
        seen = set()
        consumed = [0] * world
        for mix, batch_size in streams:
            for name, _, robot in _collect_mixture_entries(DATASET_NAMED_MIXTURES[mix], None):
                if (name, batch_size) in seen or (args.dataset and args.dataset not in name):
                    continue
                seen.add((name, batch_size))
                if args.max_batches is not None and all(n >= args.max_batches for n in consumed):
                    break
                dataset = load_dataset(name, robot)
                counts = [len(ShardedBatches(len(dataset), batch_size, r, world)) for r in range(world)]
                del dataset
                if args.max_batches is not None:
                    counts = [min(n, args.max_batches - consumed[r]) for r, n in enumerate(counts)]
                consumed = [old + n for old, n in zip(consumed, counts)]
                plan.append((name, robot, batch_size, counts))
    if world > 1:
        shared = [plan]
        torch.distributed.broadcast_object_list(shared, src=0, group=progress_group)
        plan = shared[0]
    total = sum(sum(counts) for _, _, _, counts in plan)
    with tqdm(total=total, desc="VAE cache", unit="batch", disable=rank != 0,
              dynamic_ncols=True, mininterval=2) as progress:
        for name, robot, batch_size, counts in plan:
            dataset = load_dataset(name, robot)
            windows = WanWindows(dataset, int(cfg.framework.action_model.action_horizon))
            loader = DataLoader(windows, batch_sampler=ShardedBatches(len(windows), batch_size, rank, world),
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=WanFrameCollator(identity, wan.video_image_size, cache))
            iterator = iter(loader)
            compute_size = encode_batch_size or batch_size
            packs = compute_size // batch_size
            for step in range(0, max(counts), packs):
                completed = 0
                pending_pixels, pending_keys = [], []
                for _ in range(max(0, min(packs, counts[rank] - step))):
                    batch = next(iterator)
                    if "wm_cached_latents" not in batch[0]:
                        pending_pixels.append(batch[0]["wm_pixel_values"])
                        pending_keys.extend(batch[0]["wm_cache_keys"])
                    completed += 1
                if pending_pixels:
                    pixels = torch.cat(pending_pixels, dim=0)
                    valid_rows = len(pending_keys)
                    # Keep the chosen compute shape at dataset tails or mixed
                    # hits, while retaining keys made by the training collator.
                    if valid_rows < compute_size:
                        pixels = torch.cat((pixels, pixels[-1:].expand(compute_size - valid_rows, -1, -1, -1, -1)))
                    frames = pixels.float().div_(255.0).mul_(2.0).sub_(1.0)
                    video = frames.permute(0, 2, 1, 3, 4).to(device="cuda", dtype=vae.dtype)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        modes = vae.encode(video).latent_dist.mode()
                    cache.write_batch(pending_keys, modes[:valid_rows], encode_batch_size=compute_size)
                if world > 1:
                    count = torch.tensor(completed, dtype=torch.int64)
                    torch.distributed.all_reduce(count, group=progress_group)
                    completed = count.item()
                progress.update(completed)
            del iterator, loader, windows, dataset
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
