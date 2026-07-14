#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

: "${SHARLA_CONFIG_PATH:?Set SHARLA_CONFIG_PATH to the checkpoint-matched config.yaml}"
: "${SHARLA_CKPT_PATH:?Set SHARLA_CKPT_PATH to partial_step_100000.pt}"

export NCCL_IB_DISABLE=0
export TORCH_HOME="/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/.cache/torch"
export SWANLAB_MODE=offline

num_processes=${NUM_PROCESSES:-8}
base_vlm=${BASE_VLM:-playground/Pretrained_models/RynnBrain-2B}
config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv4_hdf5_aloha_clean_random_la_sharla.yaml
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints2}
run_id=${RUN_ID:-qwenpiv4_hdf5_aloha_clean_random_la_sharla}
batch_size=${BATCH_SIZE:-16}
hdf5_root=${HDF5_ROOT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/data/RoboTwin2.0/dataset}
data_mix=${DATA_MIX:-hdf5_aloha_clean_eef}
latent_data_mix=${LATENT_DATA_MIX:-hdf5_aloha_clean_random_eef}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --num_processes "${num_processes}" \
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
  --trainer.eval_batch_size "${batch_size}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
