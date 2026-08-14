#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export SWANLAB_MODE=offline
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda-12.9/bin/ptxas}

config_yaml=./examples/Robotwin/new_train/starvla_qwenpiv5_hdf5_aloha_clean_eef_rot6d.yaml
run_id=qwenpiv5_hdf5_aloha_clean_eef_rot6d
mkdir -p "./results/Checkpoints2/${run_id}"
cp "$0" "./results/Checkpoints2/${run_id}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --run_id "${run_id}"
