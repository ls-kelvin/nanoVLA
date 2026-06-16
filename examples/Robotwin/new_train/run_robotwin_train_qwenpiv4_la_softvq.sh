#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

###########################################################################################
# QwenPI_v4_LA + 0614 softla JointTokenizer SoftVQ KL latent-action loss on RoboTwin.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export WANDB_MODE=offline

Framework_name=QwenPI_v4_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_la_softvq_robotwin.yaml
run_root_dir=./results/Checkpoints2
data_mix=robotwin32_aloha_hanging_mug
run_id=0615_${data_mix}_qwenpiv4_la_softvq
batch_size=16

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.latent_action.enabled true \
  --framework.latent_action.backend softvq \
  --framework.latent_action.loss_type soft_kl \
  --framework.latent_action.train_latent true \
  --framework.latent_action.train_action true \
  --framework.latent_action.action_train_robot_types "[aloha-agilex]" \
  --datasets.vla_data.dataset_py lerobot_la_datasets \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.latent_action.stride 16 \
  --datasets.vla_data.latent_action.horizon 16 \
  --datasets.vla_data.latent_action.image_size "[224,224]" \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 50000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size "${batch_size}" \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin
