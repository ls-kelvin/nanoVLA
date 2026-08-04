"""Episode-level soft_kl distribution cache for Sharla.

Layout::

    {cache_root}/manifest.json
    {cache_root}/{dataset_name}/episode{id}.pt

Each ``.pt`` stores SoftVQ codebook weights (soft_kl teacher targets)::

    {"distribution": Tensor[T, Q, C]}

Cache building encodes each episode step with a full-stride window
``(f0, fmid, f1)``. Training reads ``num_pairs`` consecutive stride starts at
``base_index`` and flattens to ``[num_pairs * Q, C]``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

MANIFEST_NAME = "manifest.json"
TARGET_TYPE = "soft_distribution"
LAYOUT = "episode"
PAYLOAD_KEY = "distribution"


def target_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def episode_pt_path(root: Path, dataset: str, episode_id: int) -> Path:
    return root / dataset / f"episode{int(episode_id)}.pt"


def episode_cache_fingerprint(
    *,
    sharla_cfg: dict[str, Any],
    window_cfg: dict[str, Any],
    codebook_size: int,
    num_bridge_tokens: int,
) -> str:
    sharla = dict(sharla_cfg)
    sharla.pop("kl_eps", None)
    return target_fingerprint(
        {
            "layout": LAYOUT,
            "target_type": TARGET_TYPE,
            "sharla": sharla,
            "window": window_cfg,
            "codebook_size": int(codebook_size),
            "num_bridge_tokens": int(num_bridge_tokens),
        }
    )


def fingerprint_from_config(config: DictConfig) -> str:
    la = OmegaConf.to_container(config.framework.latent_action, resolve=True)
    data_la = OmegaConf.to_container(
        config.datasets.vla_data.get("latent_action") or {}, resolve=True
    )
    if not isinstance(la, dict) or not isinstance(data_la, dict):
        raise TypeError("latent_action configs must resolve to dicts")
    sharla = dict(la.get("sharla") or {})
    window = {
        "stride": data_la.get("stride"),
        "horizon": data_la.get("horizon"),
        "include_terminal_frame": data_la.get("include_terminal_frame", True),
        "image_size": data_la.get("image_size", la.get("image_size")),
        "video_keys": data_la.get("video_keys") or data_la.get("video_key"),
        "horizon_overrides": data_la.get("horizon_overrides"),
    }
    return episode_cache_fingerprint(
        sharla_cfg=sharla,
        window_cfg=window,
        codebook_size=int(la.get("codebook_size", 256)),
        num_bridge_tokens=int(la.get("num_bridge_tokens", 8)),
    )


class EpisodeSoftKLCache:
    """Read consecutive stride soft distributions from per-episode ``.pt`` files."""

    def __init__(
        self,
        root: str | Path,
        expected_fingerprint: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Episode soft_kl cache manifest is missing: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("layout") != LAYOUT:
            raise ValueError(
                f"Expected layout={LAYOUT!r}, got {self.manifest.get('layout')!r}"
            )
        if self.manifest.get("target_type") != TARGET_TYPE:
            raise ValueError(
                f"Expected target_type={TARGET_TYPE!r}, got {self.manifest.get('target_type')!r}"
            )
        if expected_fingerprint is not None and self.manifest.get("fingerprint") != expected_fingerprint:
            raise ValueError("Episode soft_kl cache fingerprint does not match the current config")
        self.datasets: dict[str, dict[str, Any]] = dict(self.manifest.get("datasets", {}))
        self._loaded_key: Optional[tuple[str, int]] = None
        self._loaded_payload: Optional[dict[str, torch.Tensor]] = None

    def _load_episode(self, dataset: str, episode_id: int) -> dict[str, torch.Tensor]:
        key = (dataset, int(episode_id))
        if key == self._loaded_key and self._loaded_payload is not None:
            return self._loaded_payload
        path = episode_pt_path(self.root, dataset, episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"Episode soft_kl cache file is missing: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and PAYLOAD_KEY in raw:
            payload = {PAYLOAD_KEY: raw[PAYLOAD_KEY]}
        elif isinstance(raw, torch.Tensor):
            payload = {PAYLOAD_KEY: raw}
        else:
            raise ValueError(f"Unsupported episode soft_kl cache payload at {path}")
        self._loaded_key = key
        self._loaded_payload = payload
        return payload

    def read_window(
        self,
        dataset: str,
        episode_id: int,
        base_index: int,
        *,
        num_pairs: int,
        stride: int,
    ) -> torch.Tensor:
        """Return soft weights ``[num_pairs * Q, C]`` for the training window."""
        payload = self._load_episode(dataset, episode_id)
        values = payload[PAYLOAD_KEY]
        last = values.shape[0] - 1
        indices = [min(int(base_index) + i * int(stride), last) for i in range(int(num_pairs))]
        selected = values[indices]
        return selected.reshape(-1, selected.shape[-1]).float()


def write_manifest(
    root: Path,
    *,
    fingerprint: str,
    datasets: dict[str, dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "layout": LAYOUT,
        "fingerprint": fingerprint,
        "target_type": TARGET_TYPE,
        "datasets": datasets,
    }
    if extra:
        payload.update(extra)
    path = root / MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _window_meta(la_cfg: dict[str, Any], robot_type: str | None) -> dict[str, int]:
    from starVLA.dataloader.lerobot_la_datasets import (
        _resolve_latent_action_horizon,
        _resolve_latent_action_stride,
    )

    stride = _resolve_latent_action_stride(la_cfg, robot_type)
    horizon = _resolve_latent_action_horizon(la_cfg, robot_type, fallback=int(la_cfg.get("horizon", 32)))
    include_terminal = bool(la_cfg.get("include_terminal_frame", True))
    offsets = list(range(0, horizon + 1, stride))
    if include_terminal and offsets[-1] != horizon:
        offsets.append(horizon)
    if len(offsets) < 2:
        raise ValueError(f"Latent window for robot_type={robot_type!r} needs at least two offsets")
    gaps = [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)]
    if any(gap != gaps[0] for gap in gaps):
        raise ValueError(
            "episode soft_kl cache requires a uniform stride between latent window offsets; "
            f"got offsets={offsets} for robot_type={robot_type!r}"
        )
    return {
        "stride": int(gaps[0]),
        "horizon": int(offsets[-1]),
        "num_pairs": int(len(gaps)),
    }


def _distributed_context() -> tuple[bool, int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return True, int(dist.get_rank()), int(dist.get_world_size())
    return False, 0, 1


def _save_episode_distribution(
    destination: Path,
    *,
    values: torch.Tensor,
    stride: int,
    dataset_name: str,
    episode_id: int,
    rank: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".pt.rank{rank}.tmp")
    torch.save(
        {
            PAYLOAD_KEY: values,
            "stride": stride,
            "dataset": dataset_name,
            "episode": int(episode_id),
            "target_type": TARGET_TYPE,
        },
        temporary,
    )
    temporary.replace(destination)


def build_episode_soft_kl_cache(
    config: DictConfig,
    *,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    use_bf16: bool = False,
    overwrite: bool = False,
    data_mixes: list[str] | None = None,
) -> Path:
    """Encode Sharla soft codebook weights per episode step and write ``.pt`` caches."""
    from starVLA.dataloader.hdf5_la_dataset import get_vla_dataset
    from starVLA.dataloader.sharla_la_cache_dataset import (
        SoftDistributionEpisodeAssembler,
        SharlaLatentActionCacheDataset,
        build_sharla_pair_entries,
        make_sharla_pair_dataloader,
    )
    from starVLA.model.modules.latent_action import build_latent_action_encoder
    from starVLA.model.modules.latent_action.sharla_encoder import SharlaLatentActionEncoder

    backend = str(config.framework.latent_action.get("backend", "")).lower()
    if backend != "sharla":
        raise ValueError(f"Episode soft_kl cache requires backend='sharla', got {backend!r}")
    loss_type = str(config.framework.latent_action.get("loss_type", "soft_kl")).lower()
    if loss_type not in {"soft_kl", "auto"}:
        raise ValueError(f"Episode soft_kl cache requires loss_type soft_kl/auto, got {loss_type!r}")

    enabled, rank, world_size = _distributed_context()
    is_main = rank == 0
    fingerprint = fingerprint_from_config(config)
    root = Path(cache_dir).expanduser()
    if is_main:
        root.mkdir(parents=True, exist_ok=True)
    if enabled:
        import torch.distributed as dist

        dist.barrier()

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch_device = torch.device("cuda", local_rank)
        torch.cuda.set_device(torch_device)

    data_cfg = OmegaConf.create(OmegaConf.to_container(config.datasets.vla_data, resolve=True))
    la_cfg = OmegaConf.to_container(data_cfg.get("latent_action") or {}, resolve=True)
    if not isinstance(la_cfg, dict):
        raise TypeError("datasets.vla_data.latent_action must resolve to a dict")

    mixes = list(data_mixes or [])
    if not mixes:
        latent_mix = data_cfg.get("latent_data_mix", None)
        if latent_mix:
            mixes.append(str(latent_mix))
        mixes.append(str(data_cfg.data_mix))
        # Preserve order, drop duplicates.
        mixes = list(dict.fromkeys(mixes))

    encoder = build_latent_action_encoder(config)
    if not isinstance(encoder, SharlaLatentActionEncoder):
        raise TypeError(f"Expected SharlaLatentActionEncoder, got {type(encoder)!r}")
    encoder.to(device=torch_device)
    encoder.eval()

    video_keys = list(la_cfg.get("video_keys") or [])
    if not video_keys:
        single = la_cfg.get("video_key", None)
        video_keys = [single] if single else []
    if len(video_keys) != 1:
        raise ValueError("Episode soft_kl cache currently supports exactly one latent video key")
    video_key = str(video_keys[0])
    if not video_key.startswith("video."):
        video_key = f"video.{video_key}"
    image_size = tuple(int(x) for x in (la_cfg.get("image_size") or [224, 224]))

    dataset_meta: dict[str, dict[str, Any]] = {}
    all_single_datasets = []
    # Group by stride so ALOHA (8) and ARX (6) mid-frame stacks never mix in one batch.
    pair_entries_by_stride: dict[int, list] = {}
    global_episode_index = 0

    if is_main:
        print(
            f"[soft_kl_cache] building world_size={world_size} device={torch_device} "
            f"batch_size={batch_size} num_workers={num_workers} use_bf16={use_bf16} mixes={mixes}",
            flush=True,
        )

    for mix_name in mixes:
        mix_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
        mix_cfg.data_mix = mix_name
        mixture = get_vla_dataset(
            mix_cfg,
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            seed=int(config.get("seed", 42)),
        )
        for single_ds in mixture.datasets:
            # Skip datasets already processed under another mix.
            if single_ds.dataset_name in dataset_meta:
                continue
            ds_idx = len(all_single_datasets)
            all_single_datasets.append(single_ds)

            robot_type = getattr(single_ds, "robot_type", None)
            robot_type = str(robot_type) if robot_type is not None else None
            meta = _window_meta(la_cfg, robot_type)
            dataset_meta[single_ds.dataset_name] = {
                **meta,
                "robot_type": robot_type,
                "query_count": int(encoder.query_num),
                "codebook_size": int(encoder.codebook_size),
            }
            stride = int(meta["stride"])

            mine_traj_ids: list[int] = []
            for trajectory_id in single_ds.trajectory_ids:
                mine = (global_episode_index % world_size) == rank
                global_episode_index += 1
                if not mine:
                    continue
                episode_id = int(trajectory_id)
                destination = episode_pt_path(root, single_ds.dataset_name, episode_id)
                if destination.is_file() and not overwrite:
                    continue
                mine_traj_ids.append(episode_id)

            if mine_traj_ids:
                pair_entries_by_stride.setdefault(stride, []).extend(
                    build_sharla_pair_entries(
                        single_ds,
                        ds_idx,
                        mine_traj_ids,
                        stride=stride,
                    )
                )

    assembler = SoftDistributionEpisodeAssembler()
    autocast_enabled = bool(use_bf16) and torch_device.type == "cuda"
    for stride, pair_entries in sorted(pair_entries_by_stride.items()):
        if not pair_entries:
            continue
        if is_main:
            print(
                f"[soft_kl_cache] encoding stride={stride} pairs={len(pair_entries)}",
                flush=True,
            )
        cache_dataset = SharlaLatentActionCacheDataset(
            single_datasets=all_single_datasets,
            entries=pair_entries,
            video_key=video_key,
            image_size=image_size,
        )
        loader = make_sharla_pair_dataloader(
            cache_dataset,
            batch_size=batch_size,
            num_workers=max(0, int(num_workers)),
            pin_memory=torch_device.type == "cuda",
        )
        progress = tqdm(loader, desc=f"encode[r{rank}/s{stride}]", disable=not is_main)
        for f0, f1, fmid, batch_meta in progress:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                encoded = encoder.encode_distribution_from_tensors(f0, f1, fmid=fmid)
            encoded_cpu = encoded.detach().cpu().float()
            for index, item in enumerate(batch_meta):
                finished = assembler.add(
                    item["traj_key"],
                    int(item["step"]),
                    int(item["ep_num_steps"]),
                    encoded_cpu[index],
                    dataset_name=str(item["dataset_name"]),
                    traj_id=int(item["traj_id"]),
                    stride=int(item["stride"]),
                )
                if finished is None:
                    continue
                _, values, ep_meta = finished
                destination = episode_pt_path(
                    root, ep_meta["dataset_name"], int(ep_meta["traj_id"])
                )
                _save_episode_distribution(
                    destination,
                    values=values,
                    stride=int(ep_meta["stride"]),
                    dataset_name=str(ep_meta["dataset_name"]),
                    episode_id=int(ep_meta["traj_id"]),
                    rank=rank,
                )

    if assembler.pending_traj_keys():
        leftover = assembler.pending_traj_keys()
        raise RuntimeError(
            f"Rank {rank} finished encoding with incomplete episodes: {leftover[:8]}"
        )

    if enabled:
        import torch.distributed as dist

        dist.barrier()

    if is_main:
        write_manifest(
            root,
            fingerprint=fingerprint,
            datasets=dataset_meta,
            extra={"version": 1, "data_mixes": mixes},
        )
        print(
            f"[soft_kl_cache] wrote root={root} datasets={len(dataset_meta)} "
            f"world_size={world_size}",
            flush=True,
        )

    if enabled:
        import torch.distributed as dist

        dist.barrier()
    return root
