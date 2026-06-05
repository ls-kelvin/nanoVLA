set -e

export CUDA_VISIBLE_DEVICES=0

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
    # adjust_bottle
    # beat_block_hammer
    # blocks_ranking_rgb
    # blocks_ranking_size
    # click_alarmclock
    # click_bell
    # dump_bin_bigbin
    # grab_roller
    # handover_block
    # handover_mic
    # hanging_mug
    # lift_pot
    # move_can_pot
    # move_pillbottle_pad
    # move_playingcard_away
    # move_stapler_pad
    # open_laptop
    # open_microwave
    # pick_diverse_bottles
    # pick_dual_bottles
    # place_a2b_left
    # place_a2b_right
    # place_bread_basket
    # place_bread_skillet
    # place_burger_fries
    # place_can_basket
    # place_cans_plasticbox
    # place_container_plate
    # place_dual_shoes
    # place_empty_cup
    # place_fan
    # place_mouse_pad
    # place_object_basket
    # place_object_scale
    # place_object_stand
    place_phone_stand
    # place_shoe
    # press_stapler
    # put_bottles_dustbin
    # put_object_cabinet
    # rotate_qrcode
    # scan_object
    # shake_bottle_horizontally
    # shake_bottle
    # stack_blocks_three
    # stack_blocks_two
    # stack_bowls_three
    # stack_bowls_two
    # stamp_seal
    # turn_switch
)

bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n 0603_robotwin_place_phone_stand_qwen3piv3_place_phone_stand_la-joint -p 5542 -j 1 \
    -c /inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/starVLA/results/Checkpoints/0603_robotwin_cross_place_phone_stand_qwen3piv3_place-phone-stand_la-joint/checkpoints/steps_15000_pytorch_model.pt \
    "${TASKS[@]}"

# bash examples/Robotwin/eval_files/start_eval.sh \
#     -m demo_clean -n 0603_robotwin_place_phone_stand_qwen3piv3_place_phone_stand -p 5543 -j 1 \
#     -c /inspire/qb-ilm/project/qproject-fundationmodel/public/zzt/starVLA/results/Checkpoints/0603_robotwin_place_phone_stand_qwen3piv3_place_phone_stand/checkpoints/steps_30000_pytorch_model.pt \
#     "${TASKS[@]}"