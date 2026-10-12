#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SEEDS='[100100001,100100002,100100003,100100004,100100005,100100006,100100007,100100008,100100009,100100010]'

run_model() {
  local gpu="$1" port="$2" tag="$3" ckpt="$4"
  echo "[$(date -Is)] START $tag"
  GPU_ID="$gpu" PORT="$port" TASK=click_bell TAG="$tag" CKPT="$ckpt" SEEDS="$SEEDS" N=10 \
    bash "$ROOT/scripts/physical_tokens/run_one_last_four_rollout.sh"
  echo "[$(date -Is)] DONE $tag"
}

run_gpu0() {
  run_model 0 7020 actiononly "$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
  run_model 0 7020 physicsonly "$ROOT/results/Checkpoints/qwenpi_v4_p64_physicsonly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
  run_model 0 7020 m8_predicted "$ROOT/results/Checkpoints/qwenpi_v4_p64_m8_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
}

run_gpu1() {
  run_model 1 7021 joint_m32 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
  run_model 1 7021 no_token "$ROOT/results/Checkpoints/qwenpi_v4_clean_no_token_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
}

run_gpu0 > "$ROOT/logs/click_bell_compare_m8_gpu0.log" 2>&1 & p0=$!
run_gpu1 > "$ROOT/logs/click_bell_compare_m8_gpu1.log" 2>&1 & p1=$!
wait "$p0" "$p1"
