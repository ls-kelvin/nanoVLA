###########################################################################################
# QwenMetaQuery (Mantis "loss为action" metaquery architecture) on RoboTwin.
# Training settings kept consistent with Mantis (ZeRO-1, lr 5e-5, warmup 1000, ...).
# === Please modify the following paths according to your environment ===
Framework_name=QwenMetaQuery
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen/Qwen3-VL-2B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_metaquery_robotwin.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin32_aloha_place_phone_stand
run_id=0603_${data_mix}_qwen_mantis
batch_size=16                 # Mantis per_device_train_batch_size
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
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
