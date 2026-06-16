"""Quantitative action-space similarity metrics between embodiment groups."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ot
from scipy.spatial.distance import cdist

from visualization.action_space_viz import collect_mixed_actions, pca_preprocess

logger = logging.getLogger(__name__)


def split_features_by_group(features: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    unique_groups = list(dict.fromkeys(groups))
    group_array = np.asarray(groups)
    return {group: features[group_array == group] for group in unique_groups}


def subsample_features(features: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if features.shape[0] <= max_samples:
        return features
    indices = rng.choice(features.shape[0], size=max_samples, replace=False)
    return features[indices]


def wasserstein2_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_iter_max: int = 100000,
) -> float:
    """Exact 2-Wasserstein distance via POT (squared-Euclidean ground cost)."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2D, got shapes {x.shape} and {y.shape}")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("x and y must contain at least one sample.")

    n, m = x.shape[0], y.shape[0]
    weights_x = np.full(n, 1.0 / n, dtype=np.float64)
    weights_y = np.full(m, 1.0 / m, dtype=np.float64)
    cost = cdist(x, y, metric="sqeuclidean")
    w2_squared = float(ot.emd2(weights_x, weights_y, cost, numItermax=num_iter_max))
    return float(np.sqrt(max(w2_squared, 0.0)))


def pairwise_group_wasserstein2(
    features: np.ndarray,
    groups: list[str],
    *,
    max_samples_per_group: int = 512,
    seed: int = 42,
    num_iter_max: int = 100000,
) -> tuple[np.ndarray, list[str]]:
    features_by_group = split_features_by_group(features, groups)
    group_names = list(features_by_group.keys())
    rng = np.random.default_rng(seed)
    subsampled = {
        group: subsample_features(features_by_group[group], max_samples_per_group, rng)
        for group in group_names
    }

    n_groups = len(group_names)
    distance_matrix = np.zeros((n_groups, n_groups), dtype=np.float64)

    for i, group_i in enumerate(group_names):
        for j, group_j in enumerate(group_names):
            if j < i:
                distance_matrix[i, j] = distance_matrix[j, i]
                continue
            if i == j:
                distance_matrix[i, j] = 0.0
                continue

            distance = wasserstein2_distance(
                subsampled[group_i],
                subsampled[group_j],
                num_iter_max=num_iter_max,
            )
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            logger.info("Wasserstein-2: %s vs %s = %.6f", group_i, group_j, distance)

    return distance_matrix, group_names


def save_distance_matrix_csv(
    distance_matrix: np.ndarray,
    group_names: list[str],
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", *group_names])
        for group_name, row in zip(group_names, distance_matrix):
            writer.writerow([group_name, *[f"{value:.6f}" for value in row]])

    logger.info("Saved distance matrix to %s", output_path)


def plot_wasserstein_heatmap(
    distance_matrix: np.ndarray,
    group_names: list[str],
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(distance_matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(group_names)))
    ax.set_yticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticklabels(group_names)
    ax.set_title("Group Action Space Wasserstein-2")

    for row in range(len(group_names)):
        for col in range(len(group_names)):
            ax.text(
                col,
                row,
                f"{distance_matrix[row, col]:.3f}",
                ha="center",
                va="center",
                color="white" if distance_matrix[row, col] > distance_matrix.max() * 0.55 else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Wasserstein-2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)


def _parse_pca_components(cfg: dict[str, Any]) -> int | None:
    wasserstein_cfg = cfg.get("wasserstein")
    if isinstance(wasserstein_cfg, dict) and wasserstein_cfg.get("pca_components") is not None:
        return int(wasserstein_cfg["pca_components"])

    if cfg.get("pca_preprocess") is not None:
        return int(cfg["pca_preprocess"])

    pca_cfg = cfg.get("pca")
    if isinstance(pca_cfg, dict) and pca_cfg.get("n_components") is not None:
        return int(pca_cfg["n_components"])

    return None


def _parse_wasserstein_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    wasserstein_cfg = cfg.get("wasserstein")
    if not isinstance(wasserstein_cfg, dict):
        wasserstein_cfg = {}

    return {
        "max_samples_per_group": int(
            wasserstein_cfg.get(
                "max_samples_per_group",
                cfg.get("wasserstein_max_samples_per_group", 512),
            )
        ),
        "num_iter_max": int(
            wasserstein_cfg.get("num_iter_max", cfg.get("wasserstein_num_iter_max", 100000))
        ),
    }


def run_wasserstein_from_config(cfg: dict[str, Any]) -> tuple[Path, Path]:
    vectorize = str(cfg.get("vectorize", "flatten"))
    if vectorize != "flatten":
        raise ValueError(f"This metric script currently expects vectorize='flatten', got {vectorize!r}")

    seed = int(cfg.get("seed", 42))
    pca_components = _parse_pca_components(cfg)
    wasserstein_kwargs = _parse_wasserstein_cfg(cfg)

    target_horizon = cfg.get("target_horizon")
    if target_horizon is not None:
        target_horizon = int(target_horizon)

    features, groups, _ = collect_mixed_actions(
        data_root_dir=cfg["data_root_dir"],
        sources=list(cfg["sources"]),
        max_samples_per_source=int(cfg.get("max_samples_per_source", 500)),
        seed=seed,
        vectorize=vectorize,
        target_horizon=target_horizon,
    )

    if pca_components is not None:
        reduced_features, pca_reducer = pca_preprocess(features, pca_components, seed=seed)
        logger.info(
            "PCA for Wasserstein: %d -> %d dims, cumulative explained variance=%.3f",
            pca_reducer.n_features_in_,
            pca_reducer.n_components_,
            float(pca_reducer.explained_variance_ratio_.sum()),
        )
    else:
        reduced_features = features
        logger.info("Wasserstein on raw features: %d dims", features.shape[1])

    distance_matrix, group_names = pairwise_group_wasserstein2(
        reduced_features,
        groups,
        seed=seed,
        **wasserstein_kwargs,
    )

    wasserstein_cfg = cfg.get("wasserstein")
    if not isinstance(wasserstein_cfg, dict):
        wasserstein_cfg = {}

    output_matrix = Path(
        cfg.get("wasserstein_matrix_output", wasserstein_cfg.get("output_matrix", "results/viz/action_space_wasserstein.csv"))
    )
    output_heatmap = Path(
        cfg.get(
            "wasserstein_heatmap_output",
            wasserstein_cfg.get("output_heatmap", "results/viz/action_space_wasserstein.png"),
        )
    )

    save_distance_matrix_csv(distance_matrix, group_names, output_matrix)
    plot_wasserstein_heatmap(distance_matrix, group_names, output_heatmap)
    return output_matrix, output_heatmap
