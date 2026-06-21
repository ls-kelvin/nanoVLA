#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

###########################################################################################
# QwenPI_v4 on RoboTwin2.0 raw HDF5 aloha-agilex clean EEF data.
###########################################################################################

export SWANLAB_MODE=offline
# export WANDB_MODE=offline

Framework_name=QwenPI_v4
freeze_module_list=''
base_vlm=playground/Pretrained_models/RynnBrain-2B
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean.yaml
run_root_dir=./results/Checkpoints2
data_mix=hdf5_aloha_clean_eef
run_id=0619_${data_mix}_eef_qwenpiv4
batch_size=16
hdf5_root=/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/dataset

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
  --datasets.vla_data.dataset_py hdf5_dataset \
  --datasets.vla_data.data_root_dir "${hdf5_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.hdf5_action_type eef \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
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
