#!/usr/bin/env bash
set -euo pipefail

# Run on node003. Paired with the recovered legacy RoboTwin + RT Pure PI 8/8.
# ACTION_UNNORM_MODE defaults to wrap, but can be set to stats for paired tests.
VLROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
ROBOTWIN_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
CKPT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
POLICY_PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
ROBOTWIN_PY=/mnt/jfzn/liujn/robotwin_env/bin/python
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-7864}
ACTION_UNNORM_MODE=${ACTION_UNNORM_MODE:-wrap}
if [[ "$ACTION_UNNORM_MODE" != wrap && "$ACTION_UNNORM_MODE" != stats ]]; then
  echo "Invalid ACTION_UNNORM_MODE: $ACTION_UNNORM_MODE" >&2
  exit 2
fi
RUN_TAG=${RUN_TAG:-pure_no_state_wrap_rt_recovered_node3_20261013}
RUN_DIR="$VLROOT/logs/$RUN_TAG"
mkdir -p "$RUN_DIR"

cd "$VLROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$VLROOT:${PYTHONPATH:-}" ROBOTWIN_ACTION_UNNORM_MODE="$ACTION_UNNORM_MODE" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$POLICY_PY" "$VLROOT/deployment/model_server/server_policy_starvla_compat.py" \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  grep -q 'server running' "$RUN_DIR/server.log" && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -60 "$RUN_DIR/server.log"; exit 1; }
  sleep 2
done
grep -q 'server running' "$RUN_DIR/server.log"

for SEED in ${SEED_LIST:-100100009 100100023 100100028 100100000 100100015 100100027 100100036 100100039}; do
  TAG="${RUN_TAG}_seed_${SEED}"
  echo "Starting seed $SEED at $(date -Is)" | tee -a "$RUN_DIR/progress.log"
  cd "$ROBOTWIN_ROOT"
  if SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl \
    VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json" \
    __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json" \
    ROBOTWIN_FIXED_TEST_SEEDS="[$SEED]" ROBOTWIN_SKIP_UNSTABLE_SEEDS=1 \
    ROBOTWIN_TEST_NUM=1 ROBOTWIN_PATH="$ROBOTWIN_ROOT" ROBOTWIN_PYTHON="$ROBOTWIN_PY" \
    ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment \
    ROBOTWIN_SEND_STATE=0 ROBOTWIN_ACTION_UNNORM_MODE="$ACTION_UNNORM_MODE" \
    STARVLA_PATH="$VLROOT" \
    bash "$VLROOT/examples/Robotwin/eval_files/eval.sh" \
      handover_block demo_clean "$TAG" 0 "$GPU_ID" "$CKPT" "$PORT" 127.0.0.1 \
    >"$RUN_DIR/seed_${SEED}.log" 2>&1; then
    echo "Finished seed $SEED at $(date -Is)" | tee -a "$RUN_DIR/progress.log"
  else
    echo "ERROR seed $SEED at $(date -Is)" | tee -a "$RUN_DIR/progress.log"
  fi
done
echo "Run directory: $RUN_DIR"
