#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# QwenMetaQuery_LA + UniVLA AR latent-action loss on RoboTwin.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export WANDB_MODE=offline

Framework_name=QwenMetaQuery_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_metaquery_la_univla_robotwin.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin32_cross_place_phone_stand
run_id=0610_${data_mix}_qwen_metaquery_la_univla
batch_size=32
univla_ckpt=./playground/Pretrained_models/UniVLA/lam-stage-2.ckpt

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.latent_action.enabled true \
  --framework.latent_action.backend univla \
  --framework.latent_action.loss_type ar_lm \
  --framework.latent_action.univla.ckpt_path ${univla_ckpt} \
  --datasets.vla_data.dataset_py lerobot_la_datasets \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.norm_stats_path results/Checkpoints/0603_robotwin32_aloha_place_phone_stand_qwen_mantis/dataset_statistics.json \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size ${batch_size} \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin
