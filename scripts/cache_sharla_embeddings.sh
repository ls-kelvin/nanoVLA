#!/usr/bin/env bash
# Thin launcher for Sharla continuous embedding episode cache.
# Options live in the YAML embedding_cache section; CLI can override paths.
#
#   bash scripts/cache_sharla_embeddings.sh [config.yaml] [--overwrite] \
#     [--output-dir DIR] [--data-root-dir DIR]
#
# Multi-node env matches training launchers (WORLD_SIZE / RANK / MASTER_*).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source "${repo_root}/.venv/bin/activate"

CONFIG_YAML="${1:-examples/Robotwin/new_train/starvla_qwenwm_hdf5_aloha_clean_random_la_sharla.yaml}"
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
print(int(OmegaConf.load("${CONFIG_YAML}").embedding_cache.get("num_gpus", 8)))
PY
)"

PET_NPROC_PER_NODE=${MLP_WORKER_GPU:-${YAML_NUM_GPUS}}
PET_NNODES=${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}
PET_NODE_RANK=${RANK:-${MLP_ROLE_INDEX:-0}}
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

entry=(
  "${repo_root}/scripts/cache_sharla_embeddings.py"
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

MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:?"MASTER_ADDR or MLP_WORKER_0_HOST required for multi-node cache"}}
MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:?"MASTER_PORT or MLP_WORKER_0_PORT required for multi-node cache"}}

exec python -m torch.distributed.run \
  --nnodes="${PET_NNODES}" \
  --node_rank="${PET_NODE_RANK}" \
  --nproc_per_node="${PET_NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${entry[@]}"
