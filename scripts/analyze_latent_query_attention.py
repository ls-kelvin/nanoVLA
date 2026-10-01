#!/usr/bin/env python3
"""Analyze action-expert attention paid to latent action (learnable) query tokens
during action generation, on offline HDF5 data.

For each sample we run the standard inference path (``predict_action`` -> Euler
sampling with the joint foresight suffix ``[state | learnable(N) | action(T)]``)
and capture, via non-invasive forward pre-hooks on every
``Qwen2ExpertDecoderLayer``, the exact sdpa attention weights of the noisy-action
query rows over all keys. Weights are reduced in-hook to:

- group mass: attention mass on [prefix | state | learnable | action] key groups
  (per head and head-averaged),
- a2l matrix: full action-query x learnable-key sub-matrix (head-averaged),
- prefix profile: binned attention curve over prefix tokens.

No model/training code is modified; the hook recomputes Q/K exactly as
``Qwen2ExpertDecoderLayer.forward`` does for the sdpa backend (both target
checkpoints use ``action_expert_attention: sdpa``).

Example (single combo):
  source .venv/bin/activate
  python scripts/analyze_latent_query_attention.py \
    --output_dir results/Checkpoints2/0927_hdf5_aloha_clean_eef_action_hdf5_arx_clean_random_eef_latent_qwenwmv32_4b_la_sharla_a2a_a4l16 \
    --ckpt steps_80000 --data_mix hdf5_aloha_clean_eef \
    --num_samples 64 --batch_size 4 --out_dir results/attn_analysis

Smoke test:
  python scripts/analyze_latent_query_attention.py --output_dir ... \
    --ckpt steps_80000 --data_mix hdf5_aloha_clean_eef \
    --num_samples 4 --batch_size 2 --out_dir /tmp/attn_smoke

Compare mode (after running all combos):
  python scripts/analyze_latent_query_attention.py --compare \
    --out_dir results/attn_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Training run dir containing checkpoints/, config.full.yaml.")
    parser.add_argument("--ckpt", type=str, default=None, help="e.g. steps_80000.")
    parser.add_argument("--data_mix", type=str, default=None,
                        help="Registered data mix, e.g. hdf5_aloha_clean_eef.")
    parser.add_argument("--hdf5_root", type=Path, default=None,
                        help="Override ROBOTWIN2_HDF5_ROOT (must be set before starVLA imports).")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=Path, default=Path("results/attn_analysis"))
    parser.add_argument("--n_bins", type=int, default=128, help="Bins for the prefix attention profile.")
    parser.add_argument("--compare", action="store_true",
                        help="Compare mode: scan --out_dir for */attn_stats.npz and plot cross-run figures.")
    return parser.parse_args()


_ARGS = _parse_args()
if _ARGS.hdf5_root is not None:
    os.environ["ROBOTWIN2_HDF5_ROOT"] = str(_ARGS.hdf5_root)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

GROUP_NAMES = ["prefix", "state", "learnable", "action"]

OLD_HOME_PREFIX = "/mnt/netdata/Team/Personal/zzt/"
NEW_HOME_PREFIX = "/mnt/netdata/Team/Personal/zetong.zhou/"


def _patch_config_paths(node: Any) -> Any:
    """Recursively rewrite relocated absolute paths inside a config container."""
    if isinstance(node, str):
        if OLD_HOME_PREFIX in node:
            patched = node.replace(OLD_HOME_PREFIX, NEW_HOME_PREFIX)
            if not os.path.exists(node) and os.path.exists(patched):
                return patched
        return node
    if isinstance(node, dict):
        return {key: _patch_config_paths(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_patch_config_paths(value) for value in node]
    return node


def _load_run_config(output_dir: Path, hdf5_root: Optional[Path]):
    from omegaconf import OmegaConf

    from starVLA.model.framework.share_tools import apply_config_compat

    for name in ("config.full.yaml", "config.yaml"):
        config_path = output_dir / name
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            container = _patch_config_paths(OmegaConf.to_container(cfg, resolve=True))
            cfg = OmegaConf.create(container)
            break
    else:
        raise FileNotFoundError(f"No config.full.yaml/config.yaml under {output_dir}")

    if hdf5_root is not None:
        cfg.datasets.vla_data.data_root_dir = str(hdf5_root)
    return apply_config_compat(cfg)


def _task_of(dataset_name: str) -> str:
    """dataset_name is '<task>/<embodiment>/<domain>' for the hdf5 layout."""
    return str(dataset_name).split("/")[0]


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------
class AttnCollector:
    """Forward pre-hooks on every expert layer; recomputes sdpa attention weights
    for the noisy-action query rows and reduces them in-hook."""

    def __init__(self, model, n_bins: int = 128):
        from starVLA.model.modules.action_model.dual_stream_expert.qwen2_expert import AdaRMSNorm

        self._AdaRMSNorm = AdaRMSNorm
        action_model = model.action_model
        self.num_lt = int(action_model.num_learnable_tokens)
        self.action_len = int(action_model.n_action_steps)
        self.suffix_len = 1 + self.num_lt + self.action_len
        self.n_bins = int(n_bins)
        self.layers = action_model.qwenvl_with_expert.qwen_expert.layers
        self.num_layers = len(self.layers)
        self.handles = []
        self.active = False
        self._reset_batch()

    def _reset_batch(self):
        # per layer: list over Euler steps of reduced tensors
        self.buf = [[] for _ in range(self.num_layers)]

    def register(self):
        for layer_idx, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_pre_hook(self._make_hook(layer_idx), with_kwargs=True)
            )

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @staticmethod
    def _bin_profile(prof, n_bins: int):
        """prof: [B, P] -> [B, n_bins] segment-mean."""
        import torch

        bsize, prefix_len = prof.shape
        edges = torch.linspace(0, prefix_len, n_bins + 1, device=prof.device).long()
        out = prof.new_zeros(bsize, n_bins)
        for i in range(n_bins):
            s, e = int(edges[i]), int(edges[i + 1])
            if e > s:
                out[:, i] = prof[:, s:e].mean(dim=1)
        return out

    def _make_hook(self, layer_idx: int):
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        num_lt = self.num_lt
        suffix_len = self.suffix_len
        AdaRMSNorm = self._AdaRMSNorm

        def hook(module, args, kwargs):
            if not self.active:
                return
            if kwargs.get("attention_implementation", "sdpa") != "sdpa":
                return
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None or not args:
                return
            hidden_states = args[0]
            bsize, seq_len, _ = hidden_states.shape
            if seq_len != suffix_len:
                return  # not the joint foresight suffix

            ada_cond = kwargs.get("ada_cond")
            prefix_key = kwargs["prefix_key"].float()
            cos, sin = kwargs["position_embeddings"]

            # Reproduce Q/K exactly as Qwen2ExpertDecoderLayer.forward (sdpa path).
            hs = hidden_states.float()
            norm = module.input_layernorm
            if isinstance(norm, AdaRMSNorm):
                hs = norm(hs, ada_cond.float() if ada_cond is not None else None)
            else:
                hs = norm(hs)
            q = module.q_proj(hs).view(
                bsize, seq_len, module.num_attention_heads, module.head_dim
            ).transpose(1, 2)
            k = module.k_proj(hs).view(
                bsize, seq_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
            k_full = torch.cat([prefix_key, k], dim=2)
            prefix_len = prefix_key.shape[2]
            kv_len = prefix_len + seq_len
            if module.num_attention_heads != module.num_key_value_heads:
                n_rep = module.num_attention_heads // module.num_key_value_heads
                k_full = (
                    k_full[:, :, None, :, :]
                    .expand(bsize, module.num_key_value_heads, n_rep, kv_len, module.head_dim)
                    .reshape(bsize, module.num_attention_heads, kv_len, module.head_dim)
                )

            # Noisy-action query rows only: suffix positions [1+num_lt : 1+num_lt+action_len].
            q_act = q[:, :, 1 + num_lt :, :].float()
            scores = torch.matmul(q_act, k_full.transpose(-1, -2)) * module.scaling
            mask_act = attention_mask[:, 1 + num_lt :, :]  # [B, La, K], bool (True=attend)
            scores = scores.masked_fill(~mask_act[:, None, :, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)  # [B, H, La, K]

            s0 = prefix_len
            l0 = prefix_len + 1
            a0 = prefix_len + 1 + num_lt
            group = torch.stack(
                [
                    weights[..., :s0].sum(-1),
                    weights[..., s0:l0].sum(-1),
                    weights[..., l0:a0].sum(-1),
                    weights[..., a0:].sum(-1),
                ],
                dim=-1,
            )  # [B, H, La, 4]
            group_head = group.mean(dim=2)  # [B, H, 4]
            a2l = weights[..., l0:a0].mean(dim=1)  # [B, La, num_lt]
            prof = weights[..., :s0].mean(dim=(1, 2))  # [B, P]
            prof_bin = self._bin_profile(prof, self.n_bins)  # [B, n_bins]

            self.buf[layer_idx].append(
                {
                    "group_head": group_head.detach().cpu(),
                    "a2l": a2l.detach().cpu(),
                    "prefix_prof": prof_bin.detach().cpu(),
                }
            )

        return hook

    def finish_batch(self) -> dict[str, Any]:
        """Stack buffers -> [B, steps, layers, ...] numpy arrays."""
        import torch

        num_steps = len(self.buf[0])
        for layer_buf in self.buf:
            if len(layer_buf) != num_steps:
                raise RuntimeError("Unequal Euler step counts across expert layers.")
        group_head = torch.stack(
            [torch.stack([s["group_head"] for s in layer_buf], dim=1) for layer_buf in self.buf],
            dim=2,
        )  # [B, steps, layers, H, 4]
        a2l = torch.stack(
            [torch.stack([s["a2l"] for s in layer_buf], dim=1) for layer_buf in self.buf],
            dim=2,
        )  # [B, steps, layers, La, num_lt]
        prof = torch.stack(
            [torch.stack([s["prefix_prof"] for s in layer_buf], dim=1) for layer_buf in self.buf],
            dim=2,
        )  # [B, steps, layers, n_bins]
        return {
            "group_mass_head": group_head.numpy(),
            "a2l": a2l.numpy(),
            "prefix_prof": prof.numpy(),
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_combo(out_dir: Path, stats: dict[str, Any], meta: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = stats["group_mass"].astype(np.float64)  # [N, S, L, 4]
    a2l = stats["a2l"].astype(np.float64)  # [N, S, L, A, Nlt]
    prof = stats["prefix_prof"].astype(np.float64)  # [N, S, L, Bins]
    tasks = np.array([_task_of(n) for n in stats["dataset_name"]])
    title = f"{meta['run_tag']} | {meta['data_mix']}"

    # 1. Overall group mass (stacked bar).
    fig, ax = plt.subplots(figsize=(5, 4))
    mean_mass = group.mean(axis=(0, 1, 2))  # [4]
    bottom = 0.0
    for name, value in zip(GROUP_NAMES, mean_mass):
        ax.bar([meta["data_mix"]], [value], bottom=bottom, label=name)
        bottom += value
    ax.set_ylabel("attention mass")
    ax.set_title(f"Action-query attention mass\n{title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_group_mass_overall.png", dpi=150)
    plt.close(fig)

    # 2. Learnable mass heatmap over (layer, step).
    learnable = group[..., 2].mean(axis=0)  # [S, L]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(learnable.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Euler step")
    ax.set_ylabel("expert layer")
    ax.set_title(f"Learnable-query attention mass\n{title}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_learnable_layer_step.png", dpi=150)
    plt.close(fig)

    # 3. Representative action->learnable 32x32 heatmaps (first samples, last layer,
    #    mean over steps).
    num_show = min(4, a2l.shape[0])
    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mat = a2l[i, :, -1].mean(axis=0)  # [A, Nlt] at last layer, mean over steps
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(f"{tasks[i]}", fontsize=8)
        ax.set_xlabel("learnable key")
        ax.set_ylabel("action query")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Action->learnable attention (last layer, step-mean)\n{title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_a2l_heatmaps.png", dpi=150)
    plt.close(fig)

    # 4. Per-task learnable mass.
    uniq_tasks = sorted(set(tasks.tolist()))
    task_mass = [group[tasks == t][..., 2].mean() for t in uniq_tasks]
    fig, ax = plt.subplots(figsize=(max(8, len(uniq_tasks) * 0.4), 4))
    ax.bar(range(len(uniq_tasks)), task_mass)
    ax.set_xticks(range(len(uniq_tasks)))
    ax.set_xticklabels(uniq_tasks, rotation=90, fontsize=6)
    ax.set_ylabel("learnable attention mass")
    ax.set_title(f"Per-task learnable-query attention mass\n{title}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_task_learnable.png", dpi=150)
    plt.close(fig)

    # 5. Mean prefix attention profile.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(prof.mean(axis=(0, 1, 2)))
    ax.set_xlabel(f"prefix position (binned x{prof.shape[-1]})")
    ax.set_ylabel("attention mass")
    ax.set_title(f"Action-query attention over prefix positions\n{title}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_prefix_profile.png", dpi=150)
    plt.close(fig)


def _plot_compare(out_dir: Path, runs: dict[str, dict[str, Any]]) -> None:
    """runs: {combo_key: (meta, stats)}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    combos = sorted(runs.keys())
    # 1. Group mass per combo (grouped bars).
    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.2), 5))
    width = 0.2
    x = np.arange(len(combos))
    for gi, gname in enumerate(GROUP_NAMES):
        values = [runs[c][1]["group_mass"].astype(np.float64)[..., gi].mean() for c in combos]
        ax.bar(x + gi * width, values, width, label=gname)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(combos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("attention mass")
    ax.set_title("Action-query attention mass per key group (all combos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(compare_dir / "compare_group_mass.png", dpi=150)
    plt.close(fig)

    # 2. Learnable mass layer x step heatmaps, one subplot per combo.
    ncol = min(4, len(combos))
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), squeeze=False)
    for ax, combo in zip(axes.ravel(), combos):
        group = runs[combo][1]["group_mass"].astype(np.float64)
        learnable = group[..., 2].mean(axis=0)  # [S, L]
        im = ax.imshow(learnable.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(combo, fontsize=7)
        ax.set_xlabel("Euler step", fontsize=7)
        ax.set_ylabel("layer", fontsize=7)
        fig.colorbar(im, ax=ax)
    for ax in axes.ravel()[len(combos):]:
        ax.axis("off")
    fig.suptitle("Learnable-query attention mass (layer x Euler step)", fontsize=11)
    fig.tight_layout()
    fig.savefig(compare_dir / "compare_learnable_layer_step.png", dpi=150)
    plt.close(fig)

    # 3. Per-task learnable mass comparison across runs sharing the same mix.
    by_mix: dict[str, list[str]] = {}
    for combo in combos:
        by_mix.setdefault(runs[combo][0]["data_mix"], []).append(combo)
    for mix, mix_combos in by_mix.items():
        task_sets = []
        for combo in mix_combos:
            stats = runs[combo][1]
            task_sets.append(sorted({_task_of(n) for n in stats["dataset_name"]}))
        common = sorted(set.intersection(*(set(t) for t in task_sets))) if task_sets else []
        if not common:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(common) * 0.5), 4))
        width = 0.8 / len(mix_combos)
        x = np.arange(len(common))
        for ci, combo in enumerate(mix_combos):
            meta, stats = runs[combo]
            group = stats["group_mass"].astype(np.float64)
            tasks = np.array([_task_of(n) for n in stats["dataset_name"]])
            values = [group[tasks == t][..., 2].mean() for t in common]
            ax.bar(x + ci * width, values, width, label=meta["run_tag"])
        ax.set_xticks(x + width * (len(mix_combos) - 1) / 2)
        ax.set_xticklabels(common, rotation=90, fontsize=6)
        ax.set_ylabel("learnable attention mass")
        ax.set_title(f"Per-task learnable-query attention mass | {mix}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        safe_mix = mix.replace("/", "_")
        fig.savefig(compare_dir / f"compare_task_learnable__{safe_mix}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_single(args: argparse.Namespace) -> None:
    import torch

    from accelerate import PartialState

    from starVLA.dataloader import build_vla_eval_dataloader
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.train_starvla import _make_eval_dataloader_cfg
    from starVLA.training.trainer_utils.trainer_tools import enable_torch_load_legacy_pickle

    enable_torch_load_legacy_pickle()
    PartialState()  # accelerate's get_logger requires an initialized state

    output_dir = args.output_dir.resolve()
    run_tag = f"{output_dir.name.split('_')[0]}_{args.ckpt}"
    combo_key = f"{run_tag}__{args.data_mix}"
    out_dir = args.out_dir / combo_key
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = _load_run_config(output_dir, args.hdf5_root)

    ckpt_path = output_dir / "checkpoints" / f"{args.ckpt}_pytorch_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Run dir: {output_dir}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Eval mix: {args.data_mix}, num_samples={args.num_samples}, "
          f"batch_size={args.batch_size}, seed={cfg.seed}")

    print("Building eval dataloader ...")
    eval_cfg = _make_eval_dataloader_cfg(cfg, data_mix=args.data_mix)
    eval_dataloader = build_vla_eval_dataloader(
        cfg=eval_cfg,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=cfg.seed,
    )

    print("Building model framework ...")
    model = build_framework(cfg)
    model.to(device)
    model.eval()

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=False)
    del state_dict
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"Checkpoint loaded (strict=False): {len(missing)} missing, {len(unexpected)} unexpected keys")

    collector = AttnCollector(model, n_bins=args.n_bins)
    collector.register()
    print(f"Hooks on {collector.num_layers} expert layers; "
          f"num_learnable={collector.num_lt}, action_len={collector.action_len}, "
          f"suffix_len={collector.suffix_len}")

    torch.manual_seed(int(cfg.seed))

    all_group_head, all_a2l, all_prof = [], [], []
    all_names, all_langs = [], []
    try:
        for batch_idx, examples in enumerate(eval_dataloader):
            collector.active = True
            collector._reset_batch()
            _ = model.predict_action(examples=examples)
            collector.active = False
            batch = collector.finish_batch()
            all_group_head.append(batch["group_mass_head"])
            all_a2l.append(batch["a2l"])
            all_prof.append(batch["prefix_prof"])
            all_names.extend([str(e.get("dataset_name", "unknown")) for e in examples])
            all_langs.extend([str(e.get("lang", "")) for e in examples])
            print(f"batch {batch_idx}: captured {batch['group_mass_head'].shape[0]} samples x "
                  f"{batch['group_mass_head'].shape[1]} steps")
    finally:
        collector.active = False
        collector.remove()

    group_mass_head = np.concatenate(all_group_head, axis=0)  # [N, S, L, H, 4]
    a2l = np.concatenate(all_a2l, axis=0)  # [N, S, L, A, Nlt]
    prof = np.concatenate(all_prof, axis=0)  # [N, S, L, Bins]
    group_mass = group_mass_head.mean(axis=3)  # [N, S, L, 4]

    # Sanity: the four groups must sum to ~1 per query row.
    mass_sum = group_mass.sum(axis=-1)
    print(f"group mass sum: min={mass_sum.min():.4f} max={mass_sum.max():.4f} "
          f"mean={mass_sum.mean():.4f} (expect ~1.0)")

    meta = {
        "run_dir": str(output_dir),
        "run_tag": run_tag,
        "ckpt": args.ckpt,
        "data_mix": args.data_mix,
        "num_samples": int(group_mass.shape[0]),
        "num_steps": int(group_mass.shape[1]),
        "num_layers": int(group_mass.shape[2]),
        "num_heads": int(group_mass_head.shape[3]),
        "num_learnable_tokens": collector.num_lt,
        "action_len": collector.action_len,
        "n_bins": int(args.n_bins),
        "group_names": GROUP_NAMES,
    }
    np.savez_compressed(
        out_dir / "attn_stats.npz",
        group_mass=group_mass.astype(np.float32),
        group_mass_head=group_mass_head.astype(np.float16),
        a2l=a2l.astype(np.float16),
        prefix_prof=prof.astype(np.float16),
        dataset_name=np.array(all_names),
        lang=np.array(all_langs),
        meta=json.dumps(meta),
    )
    print(f"Saved stats to {out_dir / 'attn_stats.npz'}")

    stats = {
        "group_mass": group_mass,
        "a2l": a2l,
        "prefix_prof": prof,
        "dataset_name": all_names,
    }
    _plot_combo(out_dir, stats, meta)
    print(f"Saved figures to {out_dir}")


def _run_compare(args: argparse.Namespace) -> None:
    runs: dict[str, tuple[dict, dict]] = {}
    for npz_path in sorted(args.out_dir.glob("*/attn_stats.npz")):
        combo_key = npz_path.parent.name
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        stats = {
            "group_mass": data["group_mass"],
            "a2l": data["a2l"],
            "prefix_prof": data["prefix_prof"],
            "dataset_name": data["dataset_name"].tolist(),
        }
        runs[combo_key] = (meta, stats)
        print(f"Loaded {combo_key}: {meta['num_samples']} samples")
    if not runs:
        raise FileNotFoundError(f"No attn_stats.npz found under {args.out_dir}")
    _plot_compare(args.out_dir, runs)
    print(f"Saved comparison figures to {args.out_dir / 'compare'}")


def main() -> None:
    args = _ARGS
    if args.compare:
        _run_compare(args)
        return
    if args.output_dir is None or args.ckpt is None or args.data_mix is None:
        raise SystemExit("--output_dir, --ckpt and --data_mix are required (or use --compare).")
    _run_single(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
