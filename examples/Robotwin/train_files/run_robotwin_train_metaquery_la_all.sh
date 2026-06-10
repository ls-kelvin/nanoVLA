#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# QwenMetaQuery_LA on RoboTwin.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export WANDB_MODE=offline

Framework_name=QwenMetaQuery_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_metaquery_la_robotwin.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin32_non_franka_all_aloha_3
run_id=0609_${data_mix}_qwen_metaquery_la
batch_size=32

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
  --framework.latent_action.backend unit \
  --datasets.vla_data.dataset_py lerobot_la_datasets \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.norm_stats_path results/Checkpoints/0601_robotwin_all_50_qwen3piv3_all/dataset_statistics.json \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size ${batch_size} \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin
