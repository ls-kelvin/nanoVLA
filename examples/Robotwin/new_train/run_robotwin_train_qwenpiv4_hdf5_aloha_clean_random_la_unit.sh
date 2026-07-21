#!/usr/bin/env bash

set -euo pipefail

export NCCL_IB_DISABLE=0

source .venv/bin/activate

# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1

# Detailed per-submodule timing (data / forward / backward). Metrics are logged to
# wandb + results/.../metrics.jsonl; setup breakdown is saved to profile_setup.json.

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${MLP_WORKER_GPU:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${MLP_WORKER_NUM:-2}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${MLP_ROLE_INDEX:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

###########################################################################################
# QwenPI_v4_LA + UniT on RoboTwin2.0 raw HDF5 aloha-agilex EEF data.
# Action loss: clean batch. Latent-action loss: clean + randomized batch.
###########################################################################################

# export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
# export SWANLAB_MODE=offline
export SWANLAB_API_KEY=r0jz2sjqk2ALFvcHjBVQF


Framework_name=QwenPI_v4_LA
freeze_module_list=''
base_vlm=/mnt/netdata/Team/Personal/zzt/models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_unit.yaml
run_root_dir=./results/Checkpoints2
data_mix=hdf5_aloha_clean_eef
latent_data_mix=hdf5_arx_clean_random_eef
run_id=0718_${data_mix}_action_${latent_data_mix}_latent_qwenpiv4_la_unit
batch_size=4
hdf5_root=/mnt/netdata/Team/Personal/jjc/data/RoboTwin2.0/dataset
unit_tokenizer_ckpt=/mnt/netdata/Team/Personal/zzt/models/unit_stage2/unit.safetensors

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes 8 \
#   starVLA/training/train_starvla.py \
#   --config_yaml "${config_yaml}" \
#   --framework.name "${Framework_name}" \
#   --framework.qwenvl.base_vlm "${base_vlm}" \
#   --framework.latent_action.enabled true \
#   --framework.latent_action.detach_vl_embs_for_action_head false \
#   --framework.latent_action.groot_tokenizer_path "${unit_tokenizer_ckpt}" \
#   --framework.latent_action.action_train_robot_types "[aloha-agilex]" \
#   --datasets.vla_data.dataset_py hdf5_la_dataset \
#   --datasets.vla_data.data_root_dir "${hdf5_root}" \
#   --datasets.vla_data.data_mix "${data_mix}" \
#   --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
#   --datasets.vla_data.hdf5_action_type eef \
#   --datasets.vla_data.latent_action.enabled true \
#   --datasets.vla_data.per_device_batch_size "${batch_size}" \
#   --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
#   --trainer.use_dual_vla_dataloaders true \
#   --trainer.freeze_modules "${freeze_module_list}" \
#   --trainer.max_train_steps 100000 \
#   --trainer.num_warmup_steps 5000 \
#   --trainer.save_interval 10000 \
#   --trainer.logging_frequency 50 \
#   --trainer.eval_interval 1000 \
#   --trainer.eval_num_samples 512 \
#   --trainer.eval_batch_size "${batch_size}" \
#   --trainer.gradient_accumulation_steps 1 \
#   --run_root_dir "${run_root_dir}" \
#   --run_id "${run_id}" \
#   --wandb_project starVLA_Robotwin

accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MLP_WORKER_0_HOST} \
  --main_process_port ${MLP_WORKER_0_PORT} \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.enabled true \
  --framework.latent_action.detach_vl_embs_for_action_head false \
  --framework.latent_action.groot_tokenizer_path "${unit_tokenizer_ckpt}" \
  --framework.latent_action.action_train_robot_types "[aloha-agilex]" \
  --datasets.vla_data.dataset_py hdf5_la_dataset \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
  --datasets.vla_data.hdf5_action_type eef \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
  --datasets.vla_data.history_frame.enabled true \
  --trainer.use_dual_vla_dataloaders true \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 90000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --trainer.eval_num_samples 1024 \
  --trainer.eval_batch_size "${batch_size}" \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin
