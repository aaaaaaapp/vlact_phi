#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the final 20k checkpoint of the current distillation run, then run
# the fixed 8-episode panel followed by the 4-task demo_clean panel.
#
# Usage:
#   bash run_distill_medium_eval_after_train_node3.sh
#   DRY_RUN=1 bash run_distill_medium_eval_after_train_node3.sh

ROOT="/mnt/jfzn/liujn/code/VLAct-PhysToken"
RUN_ID="${P64_POLICY_RUN_ID:-qwenpi_v4_p64_m32_distill_medium_gate_tanh02_seed42_steps20000}"
CKPT="$ROOT/results/Checkpoints/$RUN_ID/checkpoints/steps_20000_pytorch_model.pt"
TRAIN_LOG="$ROOT/logs/distill_medium_train_node3_retry3.log"
ROBOTWIN_PATH="/mnt/jfzn/liujn/code/RoboTwin-main"
ROBOTWIN_PYTHON="/mnt/jfzn/liujn/robotwin_env/bin/python"
STARVLA_PATH="/mnt/jfzn/liujn/code/Starvla/Starvla"
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"

POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/logs/distill_medium_eval_after_train_node3_$STAMP"
LOCK="$ROOT/logs/.distill_medium_eval_after_train_node3.lock"

HARD_TASKS=(
  click_alarmclock click_alarmclock click_alarmclock click_alarmclock
  click_bell hanging_mug hanging_mug press_stapler
)
HARD_SEEDS=(100005 100017 100018 100022 100008 100002 100003 100053)

FOUR_TASKS=(
  handover_block handover_block handover_block handover_block handover_block
  handover_block handover_block handover_block handover_block handover_block
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
)
FOUR_SEEDS=(
  100100000 100100005 100100006 100100009 100100015 100100023 100100027 100100028 100100036 100100039
  100150000 100150001 100150004 100150005 100150006 100150008 100150009 100150010 100150011 100150013
  100100001 100100002 100100003 100100004 100100005 100100006 100100007 100100008 100100009 100100010
  100100002 100100003 100100005 100100006 100100007 100100009 100100010 100100013 100100015 100100017
)

log() {
  mkdir -p "$OUT"
  echo "[$(date '+%F %T%z')] $*" | tee -a "$OUT/queue.log"
}

preflight() {
  [[ "$(hostname)" == *node003* ]] || log "WARNING: expected node003, got $(hostname)"
  [[ -x "$ROBOTWIN_PYTHON" ]] || { echo "missing executable: $ROBOTWIN_PYTHON" >&2; return 1; }
  [[ -d "$ROBOTWIN_PATH" ]] || { echo "missing RoboTwin path: $ROBOTWIN_PATH" >&2; return 1; }
  [[ -f "$RUNNER" ]] || { echo "missing runner: $RUNNER" >&2; return 1; }
  [[ -x /usr/bin/flock ]] || { echo "missing /usr/bin/flock" >&2; return 1; }
  ((${#HARD_TASKS[@]} == 8 && ${#HARD_SEEDS[@]} == 8)) || { echo "hard panel length mismatch" >&2; return 1; }
  ((${#FOUR_TASKS[@]} == 40 && ${#FOUR_SEEDS[@]} == 40)) || { echo "four-task panel length mismatch" >&2; return 1; }
}

training_is_running() {
  pgrep -f "train_starvla\\.py.*${RUN_ID}" >/dev/null 2>&1
}

wait_for_final_checkpoint() {
  log "waiting for final checkpoint: $CKPT"
  while [[ ! -s "$CKPT" ]]; do
    sleep "$POLL_SECONDS"
  done

  # A save can be visible before it has finished writing. Require two stable
  # size samples, then wait for the matching training process to exit.
  local size1 size2
  size1="$(stat -c %s "$CKPT")"
  sleep 30
  size2="$(stat -c %s "$CKPT")"
  while [[ "$size1" != "$size2" ]]; do
    log "checkpoint still being written: $size1 -> $size2 bytes"
    size1="$size2"
    sleep 30
    size2="$(stat -c %s "$CKPT")"
  done
  log "final checkpoint is stable: $size2 bytes"

  if [[ "$WAIT_FOR_TRAINING" == 1 ]]; then
    while training_is_running; do
      log "training process still running; waiting before evaluation"
      sleep "$POLL_SECONDS"
    done
  fi
}

run_one() {
  local phase="$1" gpu="$2" port="$3" task="$4" seed="$5" config="$6"
  local tag="${RUN_ID}_${phase}_${task}_${seed}"
  local log_file="$OUT/${phase}_${task}_${seed}.log"
  log "START phase=$phase task=$task seed=$seed gpu=$gpu port=$port"
  if GPU_ID="$gpu" PORT="$port" TASK="$task" TASK_CONFIG="$config" SEED="$seed" TAG="$tag" CKPT="$CKPT" \
      ROBOTWIN_PATH="$ROBOTWIN_PATH" ROBOTWIN_PYTHON="$ROBOTWIN_PYTHON" \
      STARVLA_PATH="$STARVLA_PATH" bash "$RUNNER" >"$log_file" 2>&1; then
    log "DONE phase=$phase task=$task seed=$seed"
    return 0
  fi
  log "FAIL phase=$phase task=$task seed=$seed (see $log_file)"
  return 1
}

run_hard_panel() {
  if [[ ! -d "$ROBOTWIN_PATH/assets/background_texture/unseen" ]] ||
     [[ -z "$(find "$ROBOTWIN_PATH/assets/background_texture/unseen" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]]; then
    log "HARD_PANEL_SKIPPED missing randomized-eval assets: $ROBOTWIN_PATH/assets/background_texture/unseen"
    return 2
  fi
  local -a pids=()
  local status=0 i
  for i in "${!HARD_TASKS[@]}"; do
    run_one hard "$i" "$((7100 + i))" "${HARD_TASKS[$i]}" "${HARD_SEEDS[$i]}" demo_randomized &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || status=1; done
  log "HARD_PANEL_DONE status=$status"
  return "$status"
}

run_four_task_panel() {
  local status=0 base end j gpu
  for ((base = 0; base < ${#FOUR_TASKS[@]}; base += 8)); do
    local -a pids=()
    end=$((base + 8))
    ((end > ${#FOUR_TASKS[@]})) && end="${#FOUR_TASKS[@]}"
    for ((j = base; j < end; j++)); do
      gpu=$((j - base))
      run_one four_tasks "$gpu" "$((7300 + gpu))" "${FOUR_TASKS[$j]}" "${FOUR_SEEDS[$j]}" demo_clean &
      pids+=("$!")
    done
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || status=1; done
    log "FOUR_TASK_BATCH_DONE base=$base end=$end status=$status"
  done
  return "$status"
}

main() {
  mkdir -p "$ROOT/logs"
  exec 9>"$LOCK"
  flock -n 9 || { echo "another watcher/evaluation is already running: $LOCK" >&2; exit 2; }
  preflight
  log "watcher started run_id=$RUN_ID ckpt=$CKPT"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY_RUN: preflight passed; would wait for training, then run 8 hard episodes and 40 four-task episodes"
    exit 0
  fi
  wait_for_final_checkpoint
  [[ -s "$CKPT" ]] || { log "checkpoint disappeared"; exit 1; }
  local status=0
  log "starting hard 8-episode panel"
  run_hard_panel || { status=1; log "hard panel incomplete; continuing to four-task panel"; }
  log "starting four-task demo_clean panel"
  run_four_task_panel || status=1
  log "ALL_DONE status=$status output=$OUT"
  exit "$status"
}

main "$@"
