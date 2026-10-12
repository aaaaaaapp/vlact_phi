#!/usr/bin/env bash
set -u

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
RUN_ID=qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000
CKPT="$ROOT/results/Checkpoints/$RUN_ID/checkpoints/steps_20000_pytorch_model.pt"
TRAIN_PATTERN="train_starvla.py.*$RUN_ID"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/logs/posttrain_eval_${RUN_ID}_${STAMP}"
mkdir -p "$OUT"

TASKS=(click_alarmclock click_alarmclock click_alarmclock click_alarmclock click_bell hanging_mug hanging_mug press_stapler)
SEEDS=(100005 100017 100018 100022 100008 100002 100003 100053)

# Existing four-task 10-seed panel. Baselines for these seeds already exist;
# this script evaluates only the newly trained checkpoint.
EXTRA_TASKS=(
  handover_block handover_block handover_block handover_block handover_block
  handover_block handover_block handover_block handover_block handover_block
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
)
EXTRA_SEEDS=(
  100100000 100100005 100100006 100100009 100100015 100100023 100100027 100100028 100100036 100100039
  100150000 100150001 100150004 100150005 100150006 100150008 100150009 100150010 100150011 100150013
  100100001 100100002 100100003 100100004 100100005 100100006 100100007 100100008 100100009 100100010
  100100002 100100003 100100005 100100006 100100007 100100009 100100010 100100013 100100015 100100017
)

log() { echo "[$(date '+%F %T%z')] $*" | tee -a "$OUT/queue.log"; }

log "waiting for checkpoint: $CKPT"
while [ ! -s "$CKPT" ]; do
  sleep 30
done
log "checkpoint detected"

# Do not compete with the training workers while they are still finalizing.
for _ in $(seq 1 120); do
  if ! pgrep -af train_starvla.py | grep -F "$RUN_ID" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if pgrep -af train_starvla.py | grep -F "$RUN_ID" >/dev/null 2>&1; then
  log "WARNING: training process still present after wait; proceeding with saved 20k checkpoint"
else
  log "training process finished"
fi
sleep 20

run_one() {
  local i="$1" task="$2" seed="$3"
  local tag="posttrain_${RUN_ID}_${task}_${seed}"
  log "START $task seed=$seed gpu=$i"
  if GPU_ID="$i" PORT="$((7200+i))" TASK="$task" SEED="$seed" TAG="$tag" CKPT="$CKPT" \
      bash "$ROOT/scripts/physical_tokens/run_policy_single_seed.sh" >"$OUT/${task}_${seed}.log" 2>&1; then
    log "DONE $task seed=$seed"
  else
    log "FAIL $task seed=$seed (see $OUT/${task}_${seed}.log)"
  fi
}

pids=()
for i in $(seq 0 7); do
  run_one "$i" "${TASKS[$i]}" "${SEEDS[$i]}" &
  pids+=("$!")
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done

log "HARD_PANEL_DONE status=$status; starting extra four-task panel"
for ((base=0; base<${#EXTRA_TASKS[@]}; base+=8)); do
  pids=()
  end=$((base+8))
  [ "$end" -gt "${#EXTRA_TASKS[@]}" ] && end="${#EXTRA_TASKS[@]}"
  for ((j=base; j<end; j++)); do
    gpu=$((j-base))
    run_one "$gpu" "${EXTRA_TASKS[$j]}" "${EXTRA_SEEDS[$j]}" &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  log "EXTRA_BATCH_DONE base=$base end=$end status=$status"
done
log "ALL_DONE status=$status output=$OUT"
exit "$status"
