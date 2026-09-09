"""Pack WAN pixels in workers so training transfers one compact uint8 batch."""

import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.wan.latent_cache import WanLatentCache, video_cache_keys


class WanFrameCollator:
    def __init__(self, base_collate_fn, image_size, latent_cache=None):
        self.base_collate_fn = base_collate_fn
        self.image_size = tuple(int(v) for v in image_size)  # H, W
        self.latent_cache = latent_cache

    def __call__(self, batch):
        examples = self.base_collate_fn(batch)
        if not examples or not all("wm_frames" in example for example in examples):
            return examples
        # Keep the existing framework path for tensor/custom inputs. In
        # particular, do not quantize floating-point images to uint8.
        if any(
            not isinstance(frame, Image.Image)
            for example in examples for frame in example["wm_frames"]
        ):
            return examples
        lengths = {len(example["wm_frames"]) for example in examples}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError(f"wm_frames must have one nonzero length per batch, got {lengths}.")
        height, width = self.image_size
        pixels = np.empty((len(examples), next(iter(lengths)), 3, height, width), dtype=np.uint8)
        for row, example in enumerate(examples):
            for col, frame in enumerate(example["wm_frames"]):
                # Match QwenWMv4._frame_to_wan_tensor: convert RGB before
                # the optional PIL resize, retaining PIL's default filter.
                frame = frame.convert("RGB")
                if frame.size != (width, height):
                    frame = frame.resize((width, height))
                pixels[row, col] = np.asarray(frame).transpose(2, 0, 1)
        # A tensor uses shared-memory worker transport and DataLoader pinning;
        # Accelerate can move it with the rest of the batch. Avoid also sending
        # the original PIL payload, and avoid mutating dataset-owned samples.
        examples = [{k: v for k, v in example.items() if k != "wm_frames"} for example in examples]
        if self.latent_cache is not None:
            keys = video_cache_keys(pixels)
            meta = self.latent_cache.metadata
            shape = (meta["latent_channels"], 1 + (pixels.shape[1] - 1) // meta["temporal_stride"],
                     height // meta["spatial_stride"], width // meta["spatial_stride"])
            cached = self.latent_cache.read_batch(keys, shape)
            if cached is not None:
                examples[0]["wm_cached_latents"] = cached
                return examples
            examples[0]["wm_cache_keys"] = keys
        examples[0]["wm_pixel_values"] = torch.from_numpy(pixels)
        return examples


def build_wan_frame_collator(cfg, base_collate_fn):
    wan_data = cfg.datasets.vla_data.get("wan", {}) or {}
    if not wan_data.get("enabled", False) or not wan_data.get("precompute_inputs", False):
        return base_collate_fn
    wan_model = cfg.framework.get("wan", {}) or {}
    model_path = wan_model.get("wan_model_path")
    cache = (WanLatentCache.from_config(model_path, wan_model.get("vae_cache"))
             if model_path else None)
    return WanFrameCollator(base_collate_fn, wan_model.get("video_image_size", [256, 256]), cache)
