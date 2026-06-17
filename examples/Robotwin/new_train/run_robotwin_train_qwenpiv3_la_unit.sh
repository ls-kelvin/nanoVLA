#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

###########################################################################################
# QwenPI_v3_LA + SoftVQ KL latent-action loss on RoboTwin hanging_mug.
# Action data: hanging_mug/aloha-agilex/clean.
# Latent-action data: hanging_mug/aloha-agilex/randomized.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export WANDB_MODE=offline

Framework_name=QwenPI_v3_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv3_la_unit.yaml
run_root_dir=./results/Checkpoints2
data_mix=robotwin48_aloha_hanging_mug_clean
latent_data_mix=robotwin48_aloha_hanging_mug_randomized
run_id=0616_hanging_mug_clean_action_randomized_latent_qwenpiv3_la_unit
batch_size=8

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
  --datasets.vla_data.dataset_py lerobot_la_datasets \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.latent_per_device_batch_size "${batch_size}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.latent_data_mix "${latent_data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 50000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size "${batch_size}" \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin
