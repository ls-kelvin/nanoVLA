#!/usr/bin/env bash
# Thin launcher for soft_kl episode cache. Options live in the YAML soft_kl_cache section.
#
#   bash scripts/cache_sharla_soft_kl_episodes.sh [config.yaml] [--overwrite]
#
# Multi-node: reuse the same MLP_* platform variables as the training launchers.
# Single-node fallback uses soft_kl_cache.num_gpus (or --standalone).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source "${repo_root}/.venv/bin/activate"

CONFIG_YAML="${1:-examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_sharla.yaml}"
if [[ $# -ge 1 && "${1}" != --* ]]; then
  shift
fi
if [[ "${CONFIG_YAML}" != /* ]]; then
  CONFIG_YAML="${repo_root}/${CONFIG_YAML}"
fi

if [[ -d "${repo_root}/third_party/sharla" ]]; then
  export PYTHONPATH="${repo_root}/third_party/sharla${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_NUM_GPUS="$(python - <<PY
from omegaconf import OmegaConf
print(int(OmegaConf.load("${CONFIG_YAML}").soft_kl_cache.get("num_gpus", 1)))
PY
)"

PET_NPROC_PER_NODE=${MLP_WORKER_GPU:-${YAML_NUM_GPUS}}
PET_NNODES=${MLP_WORKER_NUM:-1}
PET_NODE_RANK=${MLP_ROLE_INDEX:-0}
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

entry=(
  "${repo_root}/scripts/cache_sharla_soft_kl_episodes.py"
  --config-yaml "${CONFIG_YAML}"
  "$@"
)

if [[ "${TOTAL_PROCESSES}" -le 1 ]]; then
  exec python "${entry[@]}"
fi

if [[ "${PET_NNODES}" -le 1 ]]; then
  exec python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${PET_NPROC_PER_NODE}" \
    "${entry[@]}"
fi

: "${MLP_WORKER_0_HOST:?MLP_WORKER_0_HOST is required for multi-node soft_kl cache}"
: "${MLP_WORKER_0_PORT:?MLP_WORKER_0_PORT is required for multi-node soft_kl cache}"

exec python -m torch.distributed.run \
  --nnodes="${PET_NNODES}" \
  --node_rank="${PET_NODE_RANK}" \
  --nproc_per_node="${PET_NPROC_PER_NODE}" \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT}" \
  "${entry[@]}"
