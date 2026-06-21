#!/usr/bin/env bash

set -euo pipefail

export NCCL_IB_DISABLE=0

source .venv/bin/activate

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${PET_NNODES:-2}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${PET_NODE_RANK:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

echo "=================================================="
echo "Distributed Training Config:"
echo "  Nodes: ${PET_NNODES}"
echo "  GPU per Node: ${PET_NPROC_PER_NODE}"
echo "  Total Processes (World Size): ${TOTAL_PROCESSES}"
echo "  Current Node Rank: ${PET_NODE_RANK}"
echo "  Master Addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=================================================="

###########################################################################################
# QwenPI_v4_LA + UniT on RoboTwin2.0 raw HDF5 aloha-agilex clean EEF data.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export SWANLAB_MODE=offline
# export WANDB_MODE=offline

Framework_name=QwenPI_v4_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_la_unit.yaml
run_root_dir=./results/Checkpoints2
data_mix=hdf5_aloha_clean_eef
run_id=0620_${data_mix}_eef_qwenpiv4_la_unit
batch_size=8
hdf5_root=/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/dataset
unit_tokenizer_ckpt=/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/starVLA/third_party/UniT/outputs/robotwin_tokenizer/checkpoint-80000

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file starVLA/config/deepseeds/deepspeed_zero1.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.enabled true \
  --framework.latent_action.detach_vl_embs_for_action_head true \
  --framework.latent_action.groot_tokenizer_path "${unit_tokenizer_ckpt}" \
  --framework.latent_action.action_train_robot_types "[aloha-agilex]" \
  --datasets.vla_data.dataset_py hdf5_la_dataset \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.hdf5_action_type eef \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 100000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size "${batch_size}" \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin
