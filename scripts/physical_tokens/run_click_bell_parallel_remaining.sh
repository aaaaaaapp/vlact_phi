#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SEEDS='[100100001,100100002,100100003,100100004,100100005,100100006,100100007,100100008,100100009,100100010]'

run_one() {
  local gpu="$1" port="$2" tag="$3" ckpt="$4"
  GPU_ID="$gpu" PORT="$port" TASK=click_bell TAG="$tag" CKPT="$ckpt" SEEDS="$SEEDS" N=10 \
    bash "$ROOT/scripts/physical_tokens/run_one_last_four_rollout.sh" \
    > "$ROOT/logs/click_bell_${tag}_driver.log" 2>&1
}

run_one 2 7030 m8_predicted "$ROOT/results/Checkpoints/qwenpi_v4_p64_m8_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt" &
run_one 3 7031 gate_tanh05 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh0p05_derived/checkpoints/steps_20000_pytorch_model.pt" &
run_one 4 7032 gate_tanh10 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh0p10_derived/checkpoints/steps_20000_pytorch_model.pt" &
run_one 5 7033 gate_tanh20 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh0p20_derived/checkpoints/steps_20000_pytorch_model.pt" &
run_one 6 7034 gate_tanh50 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh0p50_derived/checkpoints/steps_20000_pytorch_model.pt" &
run_one 7 7035 gate_sigmoid20 "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_sigmoid02_steps20000/checkpoints/steps_20000_pytorch_model.pt" &
wait
