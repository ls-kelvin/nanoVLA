"""Episode-level continuous embedding cache for Sharla (QwenWM_LA).

Layout::

    {cache_root}/manifest.json
    {cache_root}/latent_norm_stats.json
    {cache_root}/{dataset_name}/episode{id}.pt

Each ``.pt`` stores post-quantize / pre-``output_proj`` embeddings::

    {"embedding": Tensor[T, Q, D]}

Cache building encodes each episode step with a full-stride window
``(f0, fmid, f1)``. Training reads ``num_pairs`` consecutive stride starts at
``base_index`` and flattens to ``[num_pairs * Q, D]``.
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
NORM_STATS_NAME = "latent_norm_stats.json"
TARGET_TYPE = "continuous_embedding"
LAYOUT = "episode"
PAYLOAD_KEY = "embedding"


def target_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def episode_pt_path(root: Path, dataset: str, episode_id: int) -> Path:
    return root / dataset / f"episode{int(episode_id)}.pt"


def episode_cache_fingerprint(
    *,
    sharla_cfg: dict[str, Any],
    window_cfg: dict[str, Any],
    query_num: int,
    latent_dim: int,
) -> str:
    sharla = dict(sharla_cfg)
    # Norm-stats path is a consumer of the cache, not part of the embedding identity.
    sharla.pop("norm_stats_path", None)
    sharla.pop("kl_eps", None)
    return target_fingerprint(
        {
            "layout": LAYOUT,
            "target_type": TARGET_TYPE,
            "embedding_kind": "post_quant_pre_proj",
            "sharla": sharla,
            "window": window_cfg,
            "query_num": int(query_num),
            "latent_dim": int(latent_dim),
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
        query_num=int(la.get("query_num", 8)),
        latent_dim=int(la.get("latent_dim") or 64),
    )


class EpisodeEmbeddingCache:
    """Read consecutive stride embeddings from per-episode ``.pt`` files."""

    def __init__(
        self,
        root: str | Path,
        expected_fingerprint: Optional[str] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Episode embedding cache manifest is missing: {manifest_path}")
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
            raise ValueError("Episode embedding cache fingerprint does not match the current config")
        self.datasets: dict[str, dict[str, Any]] = dict(self.manifest.get("datasets", {}))
        self._loaded_key: Optional[tuple[str, int]] = None
        self._loaded_payload: Optional[dict[str, torch.Tensor]] = None

    def _load_episode(self, dataset: str, episode_id: int) -> dict[str, torch.Tensor]:
        key = (dataset, int(episode_id))
        if key == self._loaded_key and self._loaded_payload is not None:
            return self._loaded_payload
        path = episode_pt_path(self.root, dataset, episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"Episode embedding cache file is missing: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and PAYLOAD_KEY in raw:
            payload = {PAYLOAD_KEY: raw[PAYLOAD_KEY]}
        elif isinstance(raw, torch.Tensor):
            payload = {PAYLOAD_KEY: raw}
        else:
            raise ValueError(f"Unsupported episode embedding cache payload at {path}")
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
        """Return embeddings ``[num_pairs * Q, D]`` for the training window."""
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


def compute_and_save_embedding_norm_stats(
    cache_root: str | Path,
    *,
    eps: float = 1e-6,
) -> Path:
    """Scan episode embedding caches and write per-dim mean/std JSON.

    Statistics are computed over all ``[T, Q, D]`` tokens flattened to
    ``(N, D)``.  Writes ``{cache_root}/latent_norm_stats.json``.
    """
    root = Path(cache_root).expanduser()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Episode embedding cache manifest is missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("target_type") != TARGET_TYPE:
        raise ValueError(
            f"Norm-stats require target_type={TARGET_TYPE!r}, got {manifest.get('target_type')!r}"
        )

    count = 0
    sum_vec: torch.Tensor | None = None
    sumsq_vec: torch.Tensor | None = None
    datasets = dict(manifest.get("datasets", {}))
    for dataset_name in sorted(datasets.keys()):
        dataset_dir = root / dataset_name
        if not dataset_dir.is_dir():
            continue
        for path in sorted(dataset_dir.glob("episode*.pt")):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and PAYLOAD_KEY in raw:
                values = raw[PAYLOAD_KEY]
            elif isinstance(raw, torch.Tensor):
                values = raw
            else:
                raise ValueError(f"Unsupported episode embedding cache payload at {path}")
            flat = values.detach().float().reshape(-1, values.shape[-1])
            if sum_vec is None:
                sum_vec = flat.sum(dim=0)
                sumsq_vec = (flat * flat).sum(dim=0)
            else:
                sum_vec = sum_vec + flat.sum(dim=0)
                sumsq_vec = sumsq_vec + (flat * flat).sum(dim=0)
            count += int(flat.shape[0])

    if count <= 0 or sum_vec is None or sumsq_vec is None:
        raise RuntimeError(f"No embedding samples found under {root}")

    mean = sum_vec / float(count)
    var = (sumsq_vec / float(count)) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(eps)
    payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": int(count),
        "latent_dim": int(mean.numel()),
        "target_type": TARGET_TYPE,
        "fingerprint": manifest.get("fingerprint"),
    }
    out_path = root / NORM_STATS_NAME
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(out_path)
    return out_path


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
            "episode embedding cache requires a uniform stride between latent window offsets; "
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


def _save_episode_embedding(
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


def build_episode_embedding_cache(
    config: DictConfig,
    *,
    cache_dir: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    use_bf16: bool = False,
    overwrite: bool = False,
    data_mixes: list[str] | None = None,
    compute_norm_stats: bool = True,
) -> Path:
    """Encode Sharla post-quant embeddings per episode step and write ``.pt`` caches."""
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
        raise ValueError(f"Episode embedding cache requires backend='sharla', got {backend!r}")

    enabled, rank, world_size = _distributed_context()
    is_main = rank == 0
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
        mixes = list(dict.fromkeys(mixes))

    encoder = build_latent_action_encoder(config)
    if not isinstance(encoder, SharlaLatentActionEncoder):
        raise TypeError(f"Expected SharlaLatentActionEncoder, got {type(encoder)!r}")
    encoder.to(device=torch_device)
    encoder.eval()

    vision_encoder = encoder.model.vision_encoder
    needs_mid_frames = bool(getattr(vision_encoder, "num_mid_frames", 0)) or bool(
        getattr(vision_encoder, "all_frame_targets", False)
    )

    # Refresh fingerprint with the real encoder dims (config may leave latent_dim unset).
    la_container = OmegaConf.to_container(config.framework.latent_action, resolve=True)
    if not isinstance(la_container, dict):
        raise TypeError("framework.latent_action must resolve to a dict")
    la_container["query_num"] = int(encoder.query_num)
    la_container["latent_dim"] = int(encoder.latent_dim)
    fingerprint = episode_cache_fingerprint(
        sharla_cfg=dict(la_container.get("sharla") or {}),
        window_cfg={
            "stride": la_cfg.get("stride"),
            "horizon": la_cfg.get("horizon"),
            "include_terminal_frame": la_cfg.get("include_terminal_frame", True),
            "image_size": la_cfg.get("image_size", la_container.get("image_size")),
            "video_keys": la_cfg.get("video_keys") or la_cfg.get("video_key"),
            "horizon_overrides": la_cfg.get("horizon_overrides"),
        },
        query_num=int(encoder.query_num),
        latent_dim=int(encoder.latent_dim),
    )

    video_keys = list(la_cfg.get("video_keys") or [])
    if not video_keys:
        single = la_cfg.get("video_key", None)
        video_keys = [single] if single else []
    if len(video_keys) != 1:
        raise ValueError("Episode embedding cache currently supports exactly one latent video key")
    video_key = str(video_keys[0])
    if not video_key.startswith("video."):
        video_key = f"video.{video_key}"
    image_size = tuple(int(x) for x in (la_cfg.get("image_size") or [224, 224]))

    dataset_meta: dict[str, dict[str, Any]] = {}
    all_single_datasets = []
    pair_entries_by_stride: dict[int, list] = {}
    global_episode_index = 0

    if is_main:
        print(
            f"[embedding_cache] building world_size={world_size} device={torch_device} "
            f"batch_size={batch_size} num_workers={num_workers} use_bf16={use_bf16} mixes={mixes} "
            f"needs_mid_frames={needs_mid_frames}",
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
            if single_ds.dataset_name in dataset_meta:
                continue
            ds_idx = len(all_single_datasets)
            all_single_datasets.append(single_ds)

            robot_type = single_ds.lerobot_info_meta.get("robot_type", None)
            robot_type = str(robot_type) if robot_type is not None else None
            meta = _window_meta(la_cfg, robot_type)
            dataset_meta[single_ds.dataset_name] = {
                **meta,
                "robot_type": robot_type,
                "query_count": int(encoder.query_num),
                "latent_dim": int(encoder.latent_dim),
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
                        include_mid_frames=needs_mid_frames,
                    )
                )

    assembler = SoftDistributionEpisodeAssembler()
    autocast_enabled = bool(use_bf16) and torch_device.type == "cuda"
    for stride, pair_entries in sorted(pair_entries_by_stride.items()):
        if not pair_entries:
            continue
        if is_main:
            print(
                f"[embedding_cache] encoding stride={stride} pairs={len(pair_entries)}",
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
                encoded = encoder.encode_continuous_from_tensors(f0, f1, fmid=fmid)
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
                _save_episode_embedding(
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
            extra={"version": 1, "data_mixes": mixes, "embedding_kind": "post_quant_pre_proj"},
        )
        print(
            f"[embedding_cache] wrote root={root} datasets={len(dataset_meta)} "
            f"world_size={world_size}",
            flush=True,
        )
        if compute_norm_stats:
            stats_path = compute_and_save_embedding_norm_stats(root)
            print(f"[embedding_cache] wrote norm stats: {stats_path}", flush=True)

    if enabled:
        import torch.distributed as dist

        dist.barrier()
    return root
