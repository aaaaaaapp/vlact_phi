#!/usr/bin/env bash
set -u
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/logs/tanh02_four_tasks_demo_clean_node4_$STAMP"
mkdir -p "$OUT"

TASKS=(
  handover_block handover_block handover_block handover_block handover_block
  handover_block handover_block handover_block handover_block handover_block
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer beat_block_hammer
  click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell click_bell
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
  place_phone_stand place_phone_stand place_phone_stand place_phone_stand place_phone_stand
)
SEEDS=(
  100100000 100100005 100100006 100100009 100100015 100100023 100100027 100100028 100100036 100100039
  100150000 100150001 100150004 100150005 100150006 100150008 100150009 100150010 100150011 100150013
  100100001 100100002 100100003 100100004 100100005 100100006 100100007 100100008 100100009 100100010
  100100002 100100003 100100005 100100006 100100007 100100009 100100010 100100013 100100015 100100017
)
log() { echo "[$(date '+%F %T%z')] $*" | tee -a "$OUT/queue.log"; }
while [ ! -s "$CKPT" ]; do log "waiting for checkpoint $CKPT"; sleep 30; done

run_one() {
  local j="$1" task="$2" seed="$3"
  local tag="tanh02_demo_clean_${task}_${seed}"
  log "START $task seed=$seed gpu=$j"
  if GPU_ID="$j" PORT="$((7300+j))" TASK="$task" TASK_CONFIG=demo_clean SEED="$seed" TAG="$tag" CKPT="$CKPT" \
      bash "$ROOT/scripts/physical_tokens/run_policy_single_seed.sh" >"$OUT/${task}_${seed}.log" 2>&1; then
    log "DONE $task seed=$seed"
  else
    log "FAIL $task seed=$seed (see $OUT/${task}_${seed}.log)"
  fi
}

status=0
for ((base=0; base<${#TASKS[@]}; base+=8)); do
  pids=()
  end=$((base+8)); [ "$end" -gt "${#TASKS[@]}" ] && end="${#TASKS[@]}"
  for ((j=base; j<end; j++)); do
    gpu=$((j-base))
    run_one "$gpu" "${TASKS[$j]}" "${SEEDS[$j]}" & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  log "BATCH_DONE base=$base end=$end status=$status"
done
log "ALL_DONE status=$status output=$OUT"
exit "$status"
