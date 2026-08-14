#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

export NCCL_IB_DISABLE=0

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${MLP_WORKER_GPU:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${WORLD_SIZE:-1}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${RANK:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

export SWANLAB_MODE=offline

base_vlm=/mnt/netdata/Team/Personal/zzt/models/RynnBrain-4B
config_yaml=./examples/Robotwin/new_train/starvla_qwenwmv3_hdf5_aloha_clean_random_la_sharla_foresight_soft_kl_eef_rot6d.yaml
run_root_dir=./results/Checkpoints2
batch_size=8
hdf5_root=/mnt/netdata/Team/Personal/jjc/data/RoboTwin2.0/dataset
data_mix=hdf5_aloha_clean_eef_rot6d
latent_data_mix=hdf5_arx_clean_random_eef_rot6d
run_id=0814_${data_mix}_action_${latent_data_mix}_latent_qwenwmv3_4b_la_sharla_a2a_foresight_soft_kl
SHARLA_CONFIG_PATH=/mnt/netdata/Team/Personal/zzt/models/sharla_a2a/config.yaml
SHARLA_CKPT_PATH=/mnt/netdata/Team/Personal/zzt/models/sharla_a2a/partial_step_30000.pt
EPISODE_CACHE_DIR=.cache/latent_cache/sharla_soft_kl

# bash scripts/cache_sharla_soft_kl_episodes.sh "${config_yaml}" --output-dir "${EPISODE_CACHE_DIR}" --data-root-dir "${hdf5_root}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR:-127.0.0.1} \
  --main_process_port ${MASTER_PORT:-29500} \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenWMv3_LA \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.loss_type soft_kl \
  --framework.latent_action.sharla.config_path "${SHARLA_CONFIG_PATH}" \
  --framework.latent_action.sharla.ckpt_path "${SHARLA_CKPT_PATH}" \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
  --datasets.vla_data.latent_action.episode_cache_dir "${EPISODE_CACHE_DIR}" \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
  --trainer.eval_batch_size "${batch_size}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
