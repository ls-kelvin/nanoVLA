"""Sample normalized action chunks from LeRobot datasets and plot mixed action space."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from visualization.data_config import RobotVizDataConfig, get_robot_type_config

logger = logging.getLogger(__name__)

VECTORIZE_MODES = ("flatten", "mean", "first_step")
PROJECTION_MODES = ("tsne", "pca")


def _filter_action_transforms(transforms: ComposedModalityTransform, action_keys: list[str]) -> ComposedModalityTransform:
    action_key_set = set(action_keys)
    filtered = [
        transform
        for transform in transforms.transforms
        if transform.apply_to and all(key in action_key_set for key in transform.apply_to)
    ]
    return ComposedModalityTransform(transforms=filtered)


def build_action_only_dataset(
    data_root_dir: Path | str,
    dataset_path: str,
    data_config: RobotVizDataConfig,
) -> tuple[LeRobotSingleDataset, RobotVizDataConfig]:
    modality_config = {"action": data_config.modality_config()["action"]}
    transforms = _filter_action_transforms(data_config.transform(), list(data_config.action_keys))
    embodiment_tag = data_config.embodiment_tag

    dataset = LeRobotSingleDataset(
        dataset_path=Path(data_root_dir) / dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
        data_cfg={"video_backend": "decord"},
    )
    return dataset, data_config


def resolve_source_data_config(source: dict[str, Any]) -> RobotVizDataConfig:
    data_config = get_robot_type_config(source["robot_type"])
    if source.get("horizon") is not None:
        data_config = data_config.with_horizon(int(source["horizon"]))
    return data_config


def resolve_source_horizon(source: dict[str, Any]) -> int:
    return resolve_source_data_config(source).horizon


def _pad_joint_chunk(chunk: np.ndarray, target_dim: int, key: str) -> np.ndarray:
    if chunk.ndim != 2:
        raise ValueError(f"{key} must be 2D (T, D), got shape {chunk.shape}")

    current_dim = chunk.shape[1]
    if current_dim == target_dim:
        return chunk.astype(np.float32)
    if current_dim == target_dim - 1:
        pad = np.zeros((chunk.shape[0], 1), dtype=np.float32)
        return np.concatenate([chunk.astype(np.float32), pad], axis=1)

    raise ValueError(
        f"Cannot pad {key} from dim={current_dim} to target_dim={target_dim}. "
        "Expected native dim to be target_dim or target_dim-1."
    )


def _pad_horizon_chunk(chunk: np.ndarray, target_horizon: int, key: str = "action") -> np.ndarray:
    if chunk.ndim != 2:
        raise ValueError(f"{key} must be 2D (T, D), got shape {chunk.shape}")

    current_horizon = chunk.shape[0]
    if current_horizon == target_horizon:
        return chunk.astype(np.float32)
    if current_horizon < target_horizon:
        pad = np.zeros((target_horizon - current_horizon, chunk.shape[1]), dtype=np.float32)
        return np.concatenate([chunk.astype(np.float32), pad], axis=0)

    raise ValueError(
        f"Cannot pad {key} horizon from T={current_horizon} to target_horizon={target_horizon}. "
        "Native horizon must be <= target_horizon."
    )


def _build_padded_action_chunk(
    data: dict[str, Any],
    data_config: RobotVizDataConfig,
    target_horizon: int,
) -> np.ndarray:
    left_joints = _pad_joint_chunk(
        np.asarray(data["action.left_joints"], dtype=np.float32),
        data_config.left_joints_target_dim,
        "action.left_joints",
    )
    right_joints = _pad_joint_chunk(
        np.asarray(data["action.right_joints"], dtype=np.float32),
        data_config.right_joints_target_dim,
        "action.right_joints",
    )
    left_gripper = np.asarray(data["action.left_gripper"], dtype=np.float32)
    right_gripper = np.asarray(data["action.right_gripper"], dtype=np.float32)
    action_chunk = np.concatenate([left_joints, left_gripper, right_joints, right_gripper], axis=1)
    return _pad_horizon_chunk(action_chunk, target_horizon)


def _vectorize_action_chunk(action_chunk: np.ndarray, vectorize: str) -> np.ndarray:
    if vectorize not in VECTORIZE_MODES:
        raise ValueError(f"vectorize must be one of {VECTORIZE_MODES}, got {vectorize!r}")

    if vectorize == "flatten":
        return action_chunk.reshape(-1)
    if vectorize == "mean":
        return action_chunk.mean(axis=0)
    return action_chunk[0]


def resolve_target_horizon(
    sources: list[dict[str, Any]],
    explicit_target_horizon: int | None,
) -> int:
    native_horizons = [resolve_source_horizon(source) for source in sources]
    max_native_horizon = max(native_horizons)

    if explicit_target_horizon is None:
        return max_native_horizon
    if explicit_target_horizon <= 0:
        raise ValueError(f"target_horizon must be positive, got {explicit_target_horizon}")
    if explicit_target_horizon < max_native_horizon:
        raise ValueError(
            "target_horizon is smaller than a source native horizon. "
            f"target_horizon={explicit_target_horizon}, max_native_horizon={max_native_horizon}"
        )
    return explicit_target_horizon


def sample_action_vectors(
    dataset: LeRobotSingleDataset,
    data_config: RobotVizDataConfig,
    max_samples: int,
    seed: int,
    vectorize: str,
    target_horizon: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    rng = np.random.default_rng(seed)
    all_steps = dataset.all_steps
    if not all_steps:
        raise ValueError(f"Dataset {dataset.dataset_path} has no steps.")

    sample_count = min(max_samples, len(all_steps))
    indices = rng.choice(len(all_steps), size=sample_count, replace=False)

    vectors: list[np.ndarray] = []
    action_shape: tuple[int, int] | None = None
    expected_dim = data_config.padded_action_dim

    for step_index in indices:
        trajectory_id, base_index = all_steps[int(step_index)]
        raw_data = dataset.get_step_data(trajectory_id, base_index)
        transformed = dataset.transforms(raw_data)
        action_chunk = _build_padded_action_chunk(transformed, data_config, target_horizon)

        if action_chunk.shape[1] != expected_dim:
            raise ValueError(
                f"Padded action dim mismatch for {dataset.dataset_path}: "
                f"expected {expected_dim}, got {action_chunk.shape[1]}"
            )

        if action_shape is None:
            action_shape = (action_chunk.shape[0], action_chunk.shape[1])
        elif (action_chunk.shape[0], action_chunk.shape[1]) != action_shape:
            raise ValueError(
                f"Inconsistent action chunk shape inside dataset {dataset.dataset_path}: "
                f"expected {action_shape}, got {action_chunk.shape[:2]}."
            )

        vectors.append(_vectorize_action_chunk(action_chunk, vectorize))

    assert action_shape is not None
    return np.stack(vectors, axis=0), action_shape


def collect_mixed_actions(
    data_root_dir: Path | str,
    sources: list[dict[str, Any]],
    max_samples_per_source: int,
    seed: int,
    vectorize: str,
    target_horizon: int | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if not sources:
        raise ValueError("sources must contain at least one entry.")

    resolved_target_horizon = resolve_target_horizon(sources, target_horizon)
    logger.info("Using target_horizon=%d for mixed action collection", resolved_target_horizon)

    all_vectors: list[np.ndarray] = []
    all_groups: list[str] = []
    source_labels: list[str] = []
    reference_feature_dim: int | None = None

    for source_index, source in enumerate(sources):
        path = source["path"]
        robot_type = source["robot_type"]
        group = source["group"]
        source_seed = seed + source_index
        data_config = resolve_source_data_config(source)
        native_horizon = data_config.horizon

        logger.info(
            "Sampling actions from path=%s robot_type=%s group=%s horizon=%d",
            path,
            robot_type,
            group,
            native_horizon,
        )
        dataset, data_config = build_action_only_dataset(data_root_dir, path, data_config)
        vectors, action_shape = sample_action_vectors(
            dataset,
            data_config,
            max_samples=max_samples_per_source,
            seed=source_seed,
            vectorize=vectorize,
            target_horizon=resolved_target_horizon,
        )

        horizon, padded_dim = action_shape
        feature_dim = vectors.shape[1]

        if horizon != resolved_target_horizon:
            raise ValueError(
                f"Padded horizon mismatch for path={path!r}: "
                f"expected {resolved_target_horizon}, got {horizon}"
            )

        if reference_feature_dim is None:
            reference_feature_dim = feature_dim
        elif feature_dim != reference_feature_dim:
            raise ValueError(
                "Feature dims differ after joint/horizon padding; check vectorize mode and robot configs. "
                f"reference={reference_feature_dim}, path={path!r} got={feature_dim}"
            )

        logger.info(
            "  sampled shape=%s native_horizon=%d padded_horizon=%d padded_dim=%d",
            vectors.shape,
            native_horizon,
            horizon,
            padded_dim,
        )

        all_vectors.append(vectors)
        all_groups.extend([group] * len(vectors))
        source_labels.extend([path] * len(vectors))

    return np.concatenate(all_vectors, axis=0), all_groups, source_labels


def tsne_2d(features: np.ndarray, seed: int, tsne_cfg: dict[str, Any] | None = None) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError(f"t-SNE requires at least 2 samples, got {features.shape[0]}")

    cfg = dict(tsne_cfg or {})
    requested_perplexity = float(cfg.get("perplexity", 30))
    max_perplexity = max(1.0, float(features.shape[0] - 1))
    perplexity = min(requested_perplexity, max_perplexity)

    learning_rate = cfg.get("learning_rate", "auto")
    if learning_rate != "auto":
        learning_rate = float(learning_rate)

    max_iter = int(cfg.get("max_iter", cfg.get("n_iter", 1000)))

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        learning_rate=learning_rate,
        init="pca",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def pca_preprocess(
    features: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, PCA]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError(f"PCA requires at least 2 samples, got {features.shape[0]}")
    if n_components <= 0:
        raise ValueError(f"pca n_components must be positive, got {n_components}")

    max_components = min(features.shape[0], features.shape[1])
    effective_components = min(n_components, max_components)
    if effective_components < n_components:
        logger.warning(
            "Clamping PCA n_components from %d to %d (n_samples=%d, n_features=%d)",
            n_components,
            effective_components,
            features.shape[0],
            features.shape[1],
        )

    reducer = PCA(n_components=effective_components, random_state=seed)
    reduced = reducer.fit_transform(features)
    return reduced, reducer


def pca_2d(features: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    reduced, reducer = pca_preprocess(features, n_components=2, seed=seed)
    return reduced, reducer.explained_variance_ratio_


def project_to_2d(
    features: np.ndarray,
    projection: str,
    seed: int,
    *,
    pca_preprocess_dim: int | None = None,
    tsne_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    projection = projection.lower()
    explained_variance = None
    pca_reducer: PCA | None = None

    if pca_preprocess_dim is not None:
        features, pca_reducer = pca_preprocess(features, pca_preprocess_dim, seed=seed)
        logger.info(
            "PCA preprocess: %d -> %d dims, cumulative explained variance=%.3f",
            pca_reducer.n_features_in_,
            pca_reducer.n_components_,
            float(pca_reducer.explained_variance_ratio_.sum()),
        )

    if projection == "pca":
        if features.shape[1] < 2:
            raise ValueError(
                f"PCA projection requires at least 2 feature dims after preprocess, got {features.shape[1]}"
            )
        if pca_reducer is not None:
            coords = features[:, :2]
            explained_variance = pca_reducer.explained_variance_ratio_[:2]
        else:
            coords, explained_variance = pca_2d(features, seed=seed)
    elif projection == "tsne":
        coords = tsne_2d(features, seed=seed, tsne_cfg=tsne_cfg)
    else:
        raise ValueError(f"projection must be one of {PROJECTION_MODES}, got {projection!r}")

    return coords, explained_variance


def _parse_pca_preprocess_dim(cfg: dict[str, Any]) -> int | None:
    if "pca_preprocess" in cfg and cfg["pca_preprocess"] is not None:
        return int(cfg["pca_preprocess"])

    pca_cfg = cfg.get("pca")
    if isinstance(pca_cfg, dict) and pca_cfg.get("n_components") is not None:
        return int(pca_cfg["n_components"])
    return None


def plot_scatter(
    coords: np.ndarray,
    groups: list[str],
    output_path: Path | str,
    *,
    projection: str = "tsne",
    explained_variance: np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection = projection.lower()
    if projection == "pca":
        title = "Mixed Action Space (PCA)"
        xlabel = "PC 1"
        ylabel = "PC 2"
        if explained_variance is not None:
            xlabel = f"PC 1 ({explained_variance[0] * 100:.1f}%)"
            ylabel = f"PC 2 ({explained_variance[1] * 100:.1f}%)"
    else:
        title = "Mixed Action Space (t-SNE)"
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"

    unique_groups = list(dict.fromkeys(groups))
    cmap = plt.get_cmap("tab10")
    color_map = {group: cmap(index % 10) for index, group in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for group in unique_groups:
        mask = np.array([label == group for label in groups])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.65,
            label=f"{group} (n={mask.sum()})",
            color=color_map[group],
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    counts = Counter(groups)
    logger.info("Saved scatter plot to %s", output_path)
    for group, count in counts.items():
        logger.info("  group=%s count=%d", group, count)
    if explained_variance is not None:
        logger.info(
            "  PCA explained variance ratio: PC1=%.3f PC2=%.3f",
            explained_variance[0],
            explained_variance[1],
        )


def run_from_config(cfg: dict[str, Any]) -> Path:
    projection = str(cfg.get("projection", "tsne")).lower()
    if projection not in PROJECTION_MODES:
        raise ValueError(f"projection must be one of {PROJECTION_MODES}, got {projection!r}")

    seed = int(cfg.get("seed", 42))
    target_horizon = cfg.get("target_horizon")
    if target_horizon is not None:
        target_horizon = int(target_horizon)

    features, groups, _ = collect_mixed_actions(
        data_root_dir=cfg["data_root_dir"],
        sources=list(cfg["sources"]),
        max_samples_per_source=int(cfg.get("max_samples_per_source", 500)),
        seed=seed,
        vectorize=str(cfg.get("vectorize", "flatten")),
        target_horizon=target_horizon,
    )

    coords, explained_variance = project_to_2d(
        features,
        projection=projection,
        seed=seed,
        pca_preprocess_dim=_parse_pca_preprocess_dim(cfg),
        tsne_cfg=cfg.get("tsne"),
    )

    output_path = Path(cfg["output"])
    plot_scatter(coords, groups, output_path, projection=projection, explained_variance=explained_variance)
    return output_path
