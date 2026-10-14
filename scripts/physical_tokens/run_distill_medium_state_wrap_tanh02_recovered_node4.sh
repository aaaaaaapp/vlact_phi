#!/usr/bin/env bash
set -Eeuo pipefail

# Paired node004 evaluation of the node003 distill-medium checkpoint.
# The only model change is tanh-bounding of residual gates at inference.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
ROBOTWIN_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_distill_medium_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
RUN_TAG=distill_medium_state_wrap_tanh02_recovered_rt_node4_20261013
RUN_DIR="$ROOT/logs/$RUN_TAG"
mkdir -p "$RUN_DIR"

[[ "$(hostname)" == *node004* ]] || { echo "This run requires node004" >&2; exit 2; }
[[ -s "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }
[[ -d "$ROBOTWIN_ROOT/assets/background_texture/unseen" ]] || { echo "Missing unseen background assets" >&2; exit 2; }

export ROBOTWIN_PATH="$ROBOTWIN_ROOT"
export ROBOTWIN_PYTHON=/mnt/jfzn/liujn/robotwin_env/bin/python
export STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export PI_GATE_BOUND_MODE=tanh PI_GATE_ALPHA_MAX=0.2
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"

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

log() { echo "[$(date -Is)] $*" | tee -a "$RUN_DIR/progress.log"; }

run_one() {
  local phase="$1" gpu="$2" task="$3" seed="$4" task_config="$5"
  local tag="${RUN_TAG}_${phase}_${task}_${seed}"
  local job_log="$RUN_DIR/${phase}_${task}_${seed}.runner.log"
  log "START phase=$phase task=$task seed=$seed gpu=$gpu config=$task_config"
  if GPU_ID="$gpu" PORT="$((7900 + gpu))" TASK="$task" SEED="$seed" \
      TASK_CONFIG="$task_config" TAG="$tag" CKPT="$CKPT" \
      bash "$RUNNER" >"$job_log" 2>&1; then
    log "DONE phase=$phase task=$task seed=$seed gpu=$gpu"
  else
    log "ERROR phase=$phase task=$task seed=$seed gpu=$gpu log=$job_log"
  fi
}

log "BEGIN ckpt=$CKPT state=1 decode=wrap gate=tanh alpha_max=0.2 simulator=recovered_rt"
pids=()
for ((j=0; j<8; j++)); do
  run_one hard "$j" "${HARD_TASKS[j]}" "${HARD_SEEDS[j]}" demo_randomized &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
log "HARD_BATCH_DONE"

for ((base=0; base<40; base+=8)); do
  pids=()
  end=$((base+8))
  ((end>40)) && end=40
  for ((j=base; j<end; j++)); do
    run_one four "$((j-base))" "${FOUR_TASKS[j]}" "${FOUR_SEEDS[j]}" demo_clean &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  log "FOUR_BATCH_DONE base=$base end=$end"
done
log "ALL_DONE"
