#!/usr/bin/env bash
set -Eeuo pipefail

# One independent port per seed prevents slow websocket teardown from
# invalidating the next batch. Run only after the hard-episode batch ends.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
ROBOTWIN_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_distill_medium_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
EXPECTED_NODE=${EXPECTED_NODE:?EXPECTED_NODE is required}
BASE_PORT=${BASE_PORT:?BASE_PORT is required}
RUN_DIR="$ROOT/logs/$RUN_TAG"
mkdir -p "$RUN_DIR"

[[ "$(hostname)" == *"$EXPECTED_NODE"* ]] || { echo "Expected $EXPECTED_NODE, got $(hostname)" >&2; exit 2; }
[[ -s "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 2; }

export ROBOTWIN_PATH="$ROBOTWIN_ROOT"
export ROBOTWIN_PYTHON=/mnt/jfzn/liujn/robotwin_env/bin/python
export STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"

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

log() { echo "[$(date -Is)] $*" | tee -a "$RUN_DIR/progress.log"; }

run_one() {
  local index="$1" gpu="$2"
  local task="${TASKS[index]}" seed="${SEEDS[index]}"
  local port=$((BASE_PORT + index))
  local tag="${RUN_TAG}_${task}_${seed}"
  local runner_log="$RUN_DIR/${task}_${seed}.runner.log"
  log "START index=$index task=$task seed=$seed gpu=$gpu port=$port"
  if GPU_ID="$gpu" PORT="$port" TASK="$task" SEED="$seed" \
      TASK_CONFIG=demo_clean TAG="$tag" CKPT="$CKPT" \
      bash "$RUNNER" >"$runner_log" 2>&1; then
    log "DONE index=$index task=$task seed=$seed"
  else
    log "ERROR index=$index task=$task seed=$seed log=$runner_log"
  fi
}

log "BEGIN ckpt=$CKPT state=1 decode=wrap gate=${PI_GATE_BOUND_MODE:-none} alpha_max=${PI_GATE_ALPHA_MAX:-none} simulator=recovered_rt"
for ((base=0; base<40; base+=8)); do
  pids=()
  end=$((base+8))
  for ((j=base; j<end; j++)); do
    run_one "$j" "$((j-base))" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  log "BATCH_DONE base=$base end=$end"
done
log "ALL_DONE"
