import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py in {"lerobot_datasets", "lerobot_la_datasets"}:
        if dataset_py == "lerobot_la_datasets":
            from starVLA.dataloader.lerobot_la_datasets import collate_fn, get_vla_dataset
        else:
            from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader


def build_fixed_subset_indices(dataset_length: int, num_samples: int | None, seed: int) -> list[int]:
    """
    Build a deterministic subset of indices for validation.

    The same dataset length + seed will always produce the same index set.
    """
    if dataset_length <= 0:
        return []

    if num_samples is None or num_samples <= 0 or num_samples >= dataset_length:
        return list(range(dataset_length))

    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_length, size=int(num_samples), replace=False)
    return sorted(int(i) for i in indices.tolist())


def build_vla_eval_dataloader(cfg, num_samples: int | None = None, batch_size: int | None = None, seed: int | None = None):
    """
    Build a deterministic validation dataloader for VLA training.

    Validation samples are drawn once from the `mode="val"` dataset and then
    frozen via a deterministic subset so repeated evaluations always use the
    same examples when the seed is fixed.
    """
    dataset_py = getattr(cfg.datasets.vla_data, "dataset_py", "lerobot_datasets")
    if dataset_py == "lerobot_la_datasets":
        from starVLA.dataloader.lerobot_la_datasets import collate_fn, get_vla_dataset
    else:
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        mode="val",
        balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        seed=cfg.seed if seed is None else seed,
    )

    eval_num_samples = num_samples
    if eval_num_samples is None:
        eval_num_samples = getattr(cfg.trainer, "eval_num_samples", None)

    eval_batch_size = batch_size
    if eval_batch_size is None:
        eval_batch_size = getattr(vla_dataset_cfg, "eval_per_device_batch_size", None)
    if eval_batch_size is None:
        eval_batch_size = int(vla_dataset_cfg.per_device_batch_size)

    subset_indices = build_fixed_subset_indices(len(vla_dataset), eval_num_samples, cfg.seed if seed is None else seed)
    logger.info(
        "Building VLA validation dataloader with %d/%d samples, batch_size=%d, seed=%d",
        len(subset_indices),
        len(vla_dataset),
        int(eval_batch_size),
        cfg.seed if seed is None else seed,
    )
    eval_dataset = Subset(vla_dataset, subset_indices)

    return DataLoader(
        eval_dataset,
        batch_size=int(eval_batch_size),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )
