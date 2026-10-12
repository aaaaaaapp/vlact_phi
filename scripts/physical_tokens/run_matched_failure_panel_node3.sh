#!/usr/bin/env bash
set -euo pipefail

# Long-running matched evaluation queue for the pure-Pi failure panel.
# Repeats each fixed scene three times per policy to estimate rollout variance.
# Run on node3; each worker owns one GPU and one policy server at a time.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
LOG="${LOGDIR:-$ROOT/logs/matched_failure_panel_node3_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG"

TASKS=(click_alarmclock click_alarmclock click_alarmclock click_alarmclock click_bell hanging_mug hanging_mug press_stapler)
SEEDS=(100005 100017 100018 100022 100008 100002 100003 100053)
GPUS=(0 1 2 3 4 5 6 7)
PORTS=(7000 7001 7002 7003 7004 7005 7006 7007)
ACTION_CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
JOINT_CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"

run_one() {
  local i="$1" model="$2" ckpt="$3"
  local task="${TASKS[$i]}" seed="${SEEDS[$i]}" gpu="${GPUS[$i]}" port="${PORTS[$i]}"
  local tag="panel_${model}_${task}_${seed}_rep${REPEAT_ID:-1}"
  local out="$LOG/${model}_${task}_${seed}_rep${REPEAT_ID:-1}.log"
  echo "[$(date -Is)] start $model $task seed=$seed gpu=$gpu" | tee -a "$LOG/queue.log"
  GPU_ID="$gpu" PORT="$port" TASK="$task" SEED="$seed" TAG="$tag" CKPT="$ckpt" \
    bash "$RUNNER" >"$out" 2>&1
  echo "[$(date -Is)] done $model $task seed=$seed" | tee -a "$LOG/queue.log"
}

run_batch() {
  local model="$1" ckpt="$2"
  local pids=()
  for i in "${!TASKS[@]}"; do
    run_one "$i" "$model" "$ckpt" & pids+=("$!")
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return "$rc"
}

echo "[$(date -Is)] panel log=$LOG" | tee "$LOG/queue.log"
for repeat_id in 2 3; do
  export REPEAT_ID="$repeat_id"
  run_batch action "$ACTION_CKPT" || true
  run_batch joint "$JOINT_CKPT" || true
done
echo "[$(date -Is)] all queued batches finished" | tee -a "$LOG/queue.log"
