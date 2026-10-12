#!/usr/bin/env bash
set -euo pipefail

# Repeat pure Pi on the same expert-stable randomized scenes used by the
# action-only/joint panel. Wait for the currently running matched panel to end.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
LOG="${LOGDIR:-$ROOT/logs/matched_pure_pi_repeats_node3_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG"
PURE_CKPT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
TASKS=(click_alarmclock click_alarmclock click_alarmclock click_alarmclock click_bell hanging_mug hanging_mug press_stapler)
SEEDS=(100005 100017 100018 100022 100008 100002 100003 100053)
GPUS=(0 1 2 3 4 5 6 7)
PORTS=(7100 7101 7102 7103 7104 7105 7106 7107)

echo "[$(date -Is)] waiting for current fixed-seed evals to finish" | tee "$LOG/queue.log"
while pgrep -f '[e]val_policy_fixed_seeds.py' >/dev/null; do sleep 60; done
echo "[$(date -Is)] starting pure Pi repeats; log=$LOG" | tee -a "$LOG/queue.log"

run_one() {
  local i="$1" rep="$2"
  local task="${TASKS[$i]}" seed="${SEEDS[$i]}" gpu="${GPUS[$i]}" port="${PORTS[$i]}"
  local tag="panel_pure_${task}_${seed}_rep${rep}"
  local out="$LOG/${task}_${seed}_rep${rep}.log"
  echo "[$(date -Is)] start pure $task seed=$seed rep=$rep gpu=$gpu" | tee -a "$LOG/queue.log"
  GPU_ID="$gpu" PORT="$port" TASK="$task" SEED="$seed" TAG="$tag" CKPT="$PURE_CKPT" \
    bash "$RUNNER" >"$out" 2>&1
  echo "[$(date -Is)] done pure $task seed=$seed rep=$rep" | tee -a "$LOG/queue.log"
}

for rep in 1 2 3; do
  pids=()
  for i in "${!TASKS[@]}"; do run_one "$i" "$rep" & pids+=("$!"); done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  echo "[$(date -Is)] repeat $rep finished rc=$rc" | tee -a "$LOG/queue.log"
done
echo "[$(date -Is)] all pure Pi repeats finished" | tee -a "$LOG/queue.log"
