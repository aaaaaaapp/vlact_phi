#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUN_ONE="$ROOT/scripts/physical_tokens/run_one_last_four_rollout.sh"
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
ACTION="$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
JOINT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
# All are expert-stable seeds already present in prior fixed-seed manifests.
# The first six are stable pure-success discovery seeds; the last four are the
# previously repeated pure-failure panel.
SEEDS='[100004,100007,100008,100009,100010,100011,100005,100017,100018,100022]'

cd "$ROOT"
env GPU_ID=6 PORT=6850 TASK=click_alarmclock TAG=p64_click10_pure CKPT="$PURE" SEEDS="$SEEDS" N=10 \
  bash "$RUN_ONE" >logs/p64_click10_pure_gpu6.log 2>&1

env GPU_ID=6 PORT=6851 TASK=click_alarmclock TAG=p64_click10_actiononly CKPT="$ACTION" SEEDS="$SEEDS" N=10 \
  bash "$RUN_ONE" >logs/p64_click10_actiononly_gpu6.log 2>&1 &
action_pid=$!
env GPU_ID=7 PORT=6852 TASK=click_alarmclock TAG=p64_click10_joint_hard CKPT="$JOINT" SEEDS="$SEEDS" N=10 \
  bash "$RUN_ONE" >logs/p64_click10_joint_gpu7.log 2>&1 &
joint_pid=$!
wait "$action_pid" "$joint_pid"
