"""Content-addressed cache of the frozen WAN VAE's posterior modes.

Cache entries contain raw BF16 modes, not normalized targets or noisy latents.
The namespace fingerprints VAE weights/config and the encoding software. Keys
hash the exact RGB uint8 video fed to preprocessing, its shape, and encode batch
size expected by training. Offline prewarming can explicitly use a larger
compute batch (with potentially different BF16 rounding); its size is recorded
in each entry. No pickle objects are accepted by the reader.
"""

from functools import lru_cache
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
import tempfile

import torch

logger = logging.getLogger(__name__)
CACHE_VERSION = 1


@lru_cache(maxsize=8)
def _weight_fingerprint(files):
    digest = hashlib.sha256()
    for name, path, size, mtime in files:
        digest.update(name.encode())
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def vae_cache_namespace(model_path, dtype=torch.bfloat16):
    import diffusers

    root = Path(model_path).expanduser().resolve() / "vae"
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"No VAE weights found in {root}")
    paths = [root / "config.json", *weights]
    signature = tuple((p.name, str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    vae_config = json.loads((root / "config.json").read_text())
    meta = {
        "version": CACHE_VERSION,
        "weights_sha256": _weight_fingerprint(signature),
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        # Backend changes can select different BF16 convolution algorithms.
        # Also queried in DataLoader workers; this reads the library version
        # without creating a CUDA tensor or initializing a CUDA context.
        "cudnn": torch.backends.cudnn.version(),
        "dtype": str(dtype), "target": "posterior_mode_before_normalization",
        "preprocessing": "PIL_RGB_then_resize_default_CPU_to_tensor_fp32",
        "latent_channels": len(vae_config["latents_mean"]),
        "spatial_stride": vae_config.get("scale_factor_spatial", 8),
        "temporal_stride": vae_config.get("scale_factor_temporal", 4),
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(), meta


def video_cache_keys(pixels):
    """Hash CPU numpy uint8 [B,T,3,H,W] pixels; never download GPU frames."""
    header = str(tuple(pixels.shape)).encode()
    return [hashlib.sha256(header + video.tobytes()).hexdigest() for video in pixels]


class WanLatentCache:
    def __init__(self, root, namespace, metadata=None, write=True):
        self.root = Path(root).expanduser() / namespace
        self.metadata = metadata or {}
        self.write_enabled = bool(write)
        self._warned = False

    @classmethod
    def from_config(cls, model_path, config, dtype=torch.bfloat16):
        if not config or not config.get("enabled", False):
            return None
        namespace, meta = vae_cache_namespace(model_path, dtype)
        return cls(config.get("dir", ".cache/wan_vae"), namespace, meta,
                   write=config.get("write", True))

    def _path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("WAN cache keys must be SHA256 hex digests")
        return self.root / key[:2] / f"{key}.pt"

    def _warn(self, exc):
        if not self._warned:
            logger.warning("WAN VAE cache unavailable; using online encoding: %s", exc)
            self._warned = True

    def read_batch(self, keys, expected_shape):
        """All-hit batches skip encoding; mixed batches retain the original B.

        Encoding only misses could select a different convolution algorithm at
        a smaller batch size. Keep the full original batch on any cache miss.
        """
        paths = [self._path(key) for key in keys]
        if not paths or not all(path.is_file() for path in paths):
            return None
        try:
            values = []
            for path, key in zip(paths, keys):
                payload = torch.load(path, map_location="cpu", weights_only=True)
                value = payload["latent"]
                if (payload.get("key") != key or not isinstance(value, torch.Tensor)
                        or value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(expected_shape)):
                    raise ValueError(f"Invalid WAN VAE cache entry: {path}")
                values.append(value)
            return torch.stack(values)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError,
                IndexError, AttributeError, pickle.UnpicklingError) as exc:
            self._warn(exc)
            return None

    def write_batch(self, keys, raw_latents, encode_batch_size=None):
        if not self.write_enabled or keys is None:
            return
        if raw_latents.dtype != torch.bfloat16 or len(keys) != raw_latents.shape[0]:
            raise ValueError("WAN VAE cache expects one key per BF16 posterior mode")
        # One device-to-host transfer, rather than one synchronization per row.
        values = raw_latents.detach().cpu()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = self.root / "manifest.json"
            if not manifest.exists():
                self._atomic_save(manifest, self.metadata, is_json=True)
            for key, value in zip(keys, values):
                path = self._path(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                # clone: a tensor view would serialize the entire batch storage.
                self._atomic_save(path, {"key": key, "latent": value.clone(),
                                         "encode_batch_size": encode_batch_size or len(keys)})
        except (OSError, RuntimeError) as exc:
            self._warn(exc)

    @staticmethod
    def _atomic_save(path, payload, is_json=False):
        # Unique temporary names allow concurrent ranks to populate the cache.
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                if is_json:
                    handle.write(json.dumps(payload, sort_keys=True, indent=2).encode())
                else:
                    torch.save(payload, handle)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
