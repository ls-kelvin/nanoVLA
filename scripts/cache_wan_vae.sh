#!/usr/bin/env bash
set -euo pipefail

# ==================== 在这里修改配置 ====================
GPUS="0,1"                       # 使用哪些 GPU；单卡写 "0"
ENCODE_BATCH_SIZE=256              # 每张 GPU 的 VAE 编码 batch
TRAIN_BATCH_SIZE=8                # 必须与训练 batch 一致，用于生成缓存键
NUM_WORKERS=16                     # 每张 GPU 的数据读取进程数

CONFIG="examples/Robotwin/new_train/starvla_qwenwmv4_hdf5_aloha_clean_random_la_sharla_foresight_soft_kl.yaml"
WAN_MODEL="/mnt/netdata/Team/Personal/yiyang/ckpts/Wan2.1-T2V-1.3B-Diffusers"
CACHE_DIR=".cache/wan_vae"
# ======================================================

# 运行：bash scripts/cache_wan_vae.sh
# 小范围缓存：bash scripts/cache_wan_vae.sh --dataset adjust_bottle --max-batches 10
# 默认缓存配置中的两个数据流，复用已有缓存，只显示一个总进度条。
# 编码 batch 需为训练 batch 的整数倍；增大编码 batch 可能改变 BF16 舍入。

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source "${repo_root}/scripts/wan_runtime_env.sh"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OMP_NUM_THREADS=4
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

IFS=',' read -r -a gpu_list <<< "${GPUS}"

exec "${repo_root}/.venv/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${#gpu_list[@]}" \
  scripts/cache_wan_vae.py \
  --config-yaml "${CONFIG}" \
  --wan-model-path "${WAN_MODEL}" \
  --cache-dir "${CACHE_DIR}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
