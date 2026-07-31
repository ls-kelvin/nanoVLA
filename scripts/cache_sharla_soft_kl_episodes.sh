#!/usr/bin/env bash
# Thin launcher for soft_kl episode cache. Options live in the YAML soft_kl_cache section.
#
#   bash scripts/cache_sharla_soft_kl_episodes.sh [config.yaml] [--overwrite]
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

NUM_GPUS="$(python - <<PY
from omegaconf import OmegaConf
print(int(OmegaConf.load("${CONFIG_YAML}").soft_kl_cache.get("num_gpus", 1)))
PY
)"

entry=(
  "${repo_root}/scripts/cache_sharla_soft_kl_episodes.py"
  --config-yaml "${CONFIG_YAML}"
  "$@"
)

if [[ "${NUM_GPUS}" -le 1 ]]; then
  exec python "${entry[@]}"
fi

exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  "${entry[@]}"
