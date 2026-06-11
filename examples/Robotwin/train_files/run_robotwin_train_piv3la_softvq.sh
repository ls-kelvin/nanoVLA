#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# QwenPI_v3_LA + vision-only SoftVQ KL latent-action loss on RoboTwin.
# Action branch (LayerwiseFM DiT) and SoftVQ latent branch are independent.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export WANDB_MODE=offline

Framework_name=QwenPI_v3_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/train_files/qwenpiv3_la_softvq.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin_no_piper_place_phone_stand
run_id=0611_${data_mix}_qwen3piv3_la_softvq
batch_size=16

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
  --framework.latent_action.backend softvq \
  --framework.latent_action.train_latent true \
  --framework.latent_action.softvq.ckpt_path "/inspire/qb-ilm/project/qproject-fundationmodel/public/jjc/exp/tokenizer/0610_z_joint_softvq_1token_64x32_delta_kl001->003/checkpoints/partial_step_6000.pt" \
  --framework.latent_action.train_action true \
  --framework.latent_action.num_bridge_tokens 1 \
  --framework.latent_action.num_codebooks 1 \
  --framework.latent_action.codebook_size 64 \
  --framework.latent_action.action_train_robot_types "[aloha-agilex]" \
  --datasets.vla_data.dataset_py lerobot_la_datasets \
  --datasets.vla_data.latent_action.enabled true \
  --datasets.vla_data.latent_action.image_size "[256,256]" \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size ${batch_size} \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin
