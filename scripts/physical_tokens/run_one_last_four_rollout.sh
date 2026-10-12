#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
STAR=/mnt/jfzn/liujn/code/Starvla/Starvla
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
RPATH=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval
SAPIEN_VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
GPU=${GPU_ID:?GPU_ID required}
PORT=${PORT:?PORT required}
TASK=${TASK:?TASK required}
TAG=${TAG:?TAG required}
CKPT=${CKPT:?CKPT required}
SEEDS=${SEEDS:?SEEDS required}
N=${N:?N required}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}

logdir="$ROOT/logs/last_four_${TASK}_${TAG}_$(date +%Y%m%d_%H%M%S)_gpu${GPU}"
mkdir -p "$logdir"
slog="$logdir/server.log"
elog="$logdir/eval.log"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$STAR:${PYTHONPATH:-}" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$PY" deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$CKPT" --port "$PORT" --use_bf16 $SERVER_EXTRA_ARGS >"$slog" 2>&1 &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  grep -q 'server running' "$slog" && break
  kill -0 "$spid" 2>/dev/null || { tail -60 "$slog"; exit 1; }
  sleep 2
done
grep -q 'server running' "$slog"
cd "$RPATH"
SAPIEN_DISABLE_RAYTRACING=1 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl \
VK_ICD_FILENAMES="$SAPIEN_VULKAN_DIR/nvidia_icd.json" \
__EGL_VENDOR_LIBRARY_FILENAMES="$SAPIEN_VULKAN_DIR/10_nvidia.json" \
ROBOTWIN_FIXED_TEST_SEEDS="$SEEDS" ROBOTWIN_SKIP_UNSTABLE_SEEDS=1 ROBOTWIN_TEST_NUM="$N" ROBOTWIN_PATH="$RPATH" ROBOTWIN_PYTHON="$RPY" \
  ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment STARVLA_PATH="$STAR" \
  bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$TASK" demo_clean "$TAG" 0 "$GPU" "$CKPT" "$PORT" 127.0.0.1 >"$elog" 2>&1
