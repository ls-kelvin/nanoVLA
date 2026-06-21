#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

###########################################################################################
# QwenPI_v4_LA + UniT on RoboTwin2.0 raw HDF5 aloha-agilex EEF data.
# Action loss: clean batch. Latent-action loss: clean + randomized batch.
###########################################################################################

export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export SWANLAB_MODE=offline
# export WANDB_MODE=offline

Framework_name=QwenPI_v4_LA
freeze_module_list=''
base_vlm=playground/Pretrained_models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_unit.yaml
run_root_dir=./results/Checkpoints2
data_mix=hdf5_aloha_clean_eef
latent_data_mix=hdf5_aloha_clean_random_eef
run_id=0620_${data_mix}_action_clean_random_latent_qwenpiv4_la_unit
batch_size=16
hdf5_root=/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/dataset
unit_tokenizer_ckpt=/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/starVLA/third_party/UniT/outputs/robotwin_tokenizer/checkpoint-80000

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
  --trainer.use_dual_vla_dataloaders true \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 100000 \
  --trainer.num_warmup_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --trainer.eval_num_samples 512 \
  --trainer.eval_batch_size "${batch_size}" \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin
