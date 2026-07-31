#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

export NCCL_IB_DISABLE=0

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${MLP_WORKER_GPU:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${MLP_WORKER_NUM:-2}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${MLP_ROLE_INDEX:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

# export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
# export SWANLAB_MODE=offline
export SWANLAB_API_KEY=r0jz2sjqk2ALFvcHjBVQF

num_processes=${NUM_PROCESSES:-8}
base_vlm=/mnt/netdata/Team/Personal/zzt/models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_sharla.yaml
run_root_dir=./results/Checkpoints2
batch_size=2
hdf5_root=/mnt/netdata/Team/Personal/jjc/data/RoboTwin2.0/dataset
data_mix=hdf5_aloha_clean_eef
latent_data_mix=hdf5_arx_clean_random_eef
run_id=0722_${data_mix}_action_${latent_data_mix}_latent_qwenpiv4_la_sharla_lang_lh
SHARLA_CONFIG_PATH=/mnt/netdata/Team/Personal/jjc/exp/sharla/0722_ebd_dino_softvq_langalign_largeglobal_hyperadjusted/config.yaml
SHARLA_CKPT_PATH=/mnt/netdata/Team/Personal/jjc/exp/sharla/0722_ebd_dino_softvq_langalign_largeglobal_hyperadjusted/checkpoints/partial_step_30000.pt

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# accelerate launch \
#   --num_processes 2 \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   starVLA/training/train_starvla.py \
#   --config_yaml "${config_yaml}" \
#   --framework.qwenvl.base_vlm "${base_vlm}" \
#   --framework.latent_action.sharla.config_path "${SHARLA_CONFIG_PATH}" \
#   --framework.latent_action.sharla.ckpt_path "${SHARLA_CKPT_PATH}" \
#   --datasets.vla_data.data_root_dir "${hdf5_root}" \
#   --datasets.vla_data.data_mix "${data_mix}" \
#   --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
#   --datasets.vla_data.per_device_batch_size "${batch_size}" \
#   --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
#   --trainer.eval_batch_size "${batch_size}" \
#   --run_root_dir "${run_root_dir}" \
#   --run_id "${run_id}"

accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MLP_WORKER_0_HOST} \
  --main_process_port ${MLP_WORKER_0_PORT} \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.sharla.config_path "${SHARLA_CONFIG_PATH}" \
  --framework.latent_action.sharla.ckpt_path "${SHARLA_CKPT_PATH}" \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
  --datasets.vla_data.history_frame.enabled true \
  --trainer.eval_batch_size "${batch_size}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
