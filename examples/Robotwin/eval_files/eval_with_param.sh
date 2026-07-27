#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Subgoal codes are auto-dumped to {LOG_DIR}/dumped_codes/{task_name}/ep{NN}/
# (LOG_DIR is the per-run eval log dir built by start_eval.sh). To disable,
# `export STARVLA_DUMP_CODES_DIR=` (empty). To override the location,
# `export STARVLA_DUMP_CODES_DIR=/some/path` before invoking this script.
# Decode later with:
#   .venv/bin/python examples/Robotwin/eval_files/decode_dumped_codes.py \
#       --dump_dir <printed dump dir> \
#       --ckpt_path <same ckpt as below> --output_dir <out> \
#       --decoder_type ref --ref_decoder_config ... --ref_decoder_ckpt_path ...



TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle_horizontally
    shake_bottle
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

# Sort TASKS by descending step_limit (from RoboTwin's _eval_step_limit.yml) so that
# start_eval.sh's greedy "whichever slot is free grabs the next task" scheduler behaves
# like LPT (Longest-Processing-Time-first) list scheduling: dispatching the longest tasks
# first keeps the per-GPU total task time balanced across the run. Tasks missing from the
# file fall back to 1000, matching RoboTwin's own default (see envs/_base_task.py).
STEP_LIMIT_FILE="${ROBOTWIN_PATH:-/mnt/netdata/Team/Personal/zzt/codebase/RoboTwin}/task_config/_eval_step_limit.yml"
DEFAULT_STEP_LIMIT=1000

declare -A STEP_LIMITS
if [[ -f "${STEP_LIMIT_FILE}" ]]; then
    while IFS=':' read -r step_limit_key step_limit_value; do
        step_limit_key="$(echo "${step_limit_key}" | xargs)"
        step_limit_value="$(echo "${step_limit_value}" | xargs)"
        if [[ -n "${step_limit_key}" ]]; then
            STEP_LIMITS["${step_limit_key}"]="${step_limit_value}"
        fi
    done < "${STEP_LIMIT_FILE}"
else
    echo "[WARN] step limit file not found: ${STEP_LIMIT_FILE}; leaving task order unchanged" >&2
fi

sort_tasks_by_step_limit_desc() {
    local task
    for task in "${TASKS[@]}"; do
        printf '%s\t%s\n' "${STEP_LIMITS[${task}]:-${DEFAULT_STEP_LIMIT}}" "${task}"
    done | sort -t $'\t' -k1,1nr -k2,2 | cut -f2
}

if [[ -f "${STEP_LIMIT_FILE}" ]]; then
    mapfile -t TASKS < <(sort_tasks_by_step_limit_desc)
fi

echo "[INFO] Starting RoboTwin evaluation from ${REPO_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; tasks=${#TASKS[@]}"
echo "[INFO] Tasks sorted by step_limit desc (LPT scheduling for balanced per-GPU load):"
for task in "${TASKS[@]}"; do
    echo "         ${task}: ${STEP_LIMITS[${task}]:-${DEFAULT_STEP_LIMIT}}"
done

exec bash "${SCRIPT_DIR}/start_eval.sh" \
    -m demo_clean -n 0724_qwenpiv4_la_sharla_lang_align2aloha -p 5542 -j 2 \
    -c /mnt/netdata/Team/Personal/zzt/codebase/starVLA/results/Checkpoints2/0724_hdf5_aloha_clean_eef_action_hdf5_arx_clean_random_eef_latent_qwenpiv4_la_sharla_lang_align2aloha/checkpoints/steps_90000_pytorch_model.pt \
    "${TASKS[@]}"