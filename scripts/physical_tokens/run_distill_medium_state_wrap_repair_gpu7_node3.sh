#!/usr/bin/env bash
set -Eeuo pipefail

# The original eight-GPU run's port 7897 was occupied by another service.
# The hard seed has completed separately. Run only five missing four-task
# assignments, each on a unique port so server teardown cannot race reuse.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
ROBOTWIN_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_distill_medium_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
RUNNER="$ROOT/scripts/physical_tokens/run_policy_single_seed.sh"
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
RUN_TAG=distill_medium_state_wrap_recovered_rt_node3_20261013
RUN_DIR="$ROOT/logs/${RUN_TAG}_gpu7_repair"
mkdir -p "$RUN_DIR"

[[ "$(hostname)" == *node003* ]] || { echo "This run requires node003" >&2; exit 2; }
export ROBOTWIN_PATH="$ROBOTWIN_ROOT"
export ROBOTWIN_PYTHON=/mnt/jfzn/liujn/robotwin_env/bin/python
export STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"

run_one() {
  local phase="$1" task="$2" seed="$3" task_config="$4" port="$5"
  local tag="${RUN_TAG}_${phase}_${task}_${seed}"
  local log_file="$RUN_DIR/${phase}_${task}_${seed}.runner.log"
  echo "[$(date -Is)] START $phase $task $seed" | tee -a "$RUN_DIR/progress.log"
  if GPU_ID=7 PORT="$port" TASK="$task" SEED="$seed" TASK_CONFIG="$task_config" \
      TAG="$tag" CKPT="$CKPT" bash "$RUNNER" >"$log_file" 2>&1; then
    echo "[$(date -Is)] DONE $phase $task $seed" | tee -a "$RUN_DIR/progress.log"
  else
    echo "[$(date -Is)] ERROR $phase $task $seed log=$log_file" | tee -a "$RUN_DIR/progress.log"
  fi
}

run_one four handover_block 100100028 demo_clean 7918
run_one four beat_block_hammer 100150008 demo_clean 7919
run_one four click_bell 100100004 demo_clean 7920
run_one four place_phone_stand 100100003 demo_clean 7921
run_one four place_phone_stand 100100017 demo_clean 7922
echo "[$(date -Is)] ALL_DONE" | tee -a "$RUN_DIR/progress.log"
