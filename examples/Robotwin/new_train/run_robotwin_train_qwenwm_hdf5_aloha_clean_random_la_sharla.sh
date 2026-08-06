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

base_vlm=/mnt/netdata/Team/Personal/zzt/models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenwm_hdf5_aloha_clean_random_la_sharla.yaml
run_root_dir=./results/Checkpoints2
batch_size=8
hdf5_root=/mnt/netdata/Team/Personal/jjc/data/RoboTwin2.0/dataset
data_mix=hdf5_aloha_clean_eef
latent_data_mix=hdf5_arx_clean_random_eef
run_id=0806_${data_mix}_action_${latent_data_mix}_latent_qwenwm_la_sharla_a2a
SHARLA_CONFIG_PATH=/mnt/netdata/Team/Personal/zzt/models/sharla_a2a/config.yaml
SHARLA_CKPT_PATH=/mnt/netdata/Team/Personal/zzt/models/sharla_a2a/partial_step_30000.pt
EMBEDDING_CACHE_DIR=.cache/latent_cache/sharla_a2a_embedding

bash scripts/cache_sharla_embeddings.sh "${config_yaml}" --output-dir "${EMBEDDING_CACHE_DIR}" --data-root-dir "${hdf5_root}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR:-127.0.0.1} \
  --main_process_port ${MASTER_PORT:-29500} \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name QwenWM_LA \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.sharla.config_path "${SHARLA_CONFIG_PATH}" \
  --framework.latent_action.sharla.ckpt_path "${SHARLA_CKPT_PATH}" \
  --framework.latent_action.sharla.norm_stats_path "${EMBEDDING_CACHE_DIR}/latent_norm_stats.json" \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
  --datasets.vla_data.latent_action.embedding_cache_dir "${EMBEDDING_CACHE_DIR}" \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
  --trainer.eval_batch_size "${batch_size}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
