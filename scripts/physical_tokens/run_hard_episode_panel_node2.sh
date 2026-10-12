#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
ACTION="$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
JOINT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
M8="$ROOT/results/Checkpoints/qwenpi_v4_p64_m8_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"

TASKS=(click_alarmclock click_alarmclock click_alarmclock click_alarmclock click_bell hanging_mug hanging_mug press_stapler)
SEEDS=(100005 100017 100018 100022 100008 100002 100003 100053)
LOGROOT="$ROOT/logs/hard_episode_panel_node2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGROOT"

run_one() {
  local model="$1" ckpt="$2" i="$3"
  local task="${TASKS[$i]}" seed="${SEEDS[$i]}" gpu="$i" port="$((7100 + i))"
  local log="$LOGROOT/${model}_${task}_${seed}.log"
  echo "[$(date -Is)] START $model $task seed=$seed gpu=$gpu" >> "$LOGROOT/queue.log"
  GPU_ID="$gpu" PORT="$port" TASK="$task" SEED="$seed" TAG="hard_${model}_${task}_${seed}" CKPT="$ckpt" \
    bash "$RUNNER" > "$log" 2>&1
  echo "[$(date -Is)] DONE $model $task seed=$seed" >> "$LOGROOT/queue.log"
}

run_batch() {
  local model="$1" ckpt="$2"; local pids=()
  for i in "${!TASKS[@]}"; do run_one "$model" "$ckpt" "$i" & pids+=("$!"); done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return "$rc"
}

echo "[$(date -Is)] LOGROOT=$LOGROOT" | tee "$LOGROOT/queue.log"
run_batch pure "$PURE" || true
run_batch actiononly "$ACTION" || true
run_batch joint_m32 "$JOINT" || true
run_batch m8 "$M8" || true
echo "[$(date -Is)] ALL_DONE" >> "$LOGROOT/queue.log"
