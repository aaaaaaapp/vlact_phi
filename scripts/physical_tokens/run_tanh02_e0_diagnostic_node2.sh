#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
CKPT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
GPU=0
PORT=8200
TAG=tanh02_e0_click_failure3_node2_20261029
OUT="$ROOT/logs/$TAG"
SEEDS='[100002,100004,100008]'
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library

[[ "$(hostname)" == *node002* ]] || { echo "Run only on node002" >&2; exit 2; }
[[ -s "$CKPT" ]] || { echo "Missing checkpoint" >&2; exit 2; }
if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then echo "Port $PORT occupied" >&2; exit 2; fi
mkdir -p "$OUT"
export ROBOTWIN_PATH="$SIM" ROBOTWIN_PYTHON="$RPY" STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$RPY"):$PATH" NINJA_PYTHON="$RPY" CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.0

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  deployment/model_server/server_policy_starvla_compat.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 --disable-discrete-conditioning \
  >"$OUT/server.log" 2>&1 &
spid=$!
cleanup() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  if grep -q 'server running' "$OUT/server.log"; then break; fi
  kill -0 "$spid" 2>/dev/null || { tail -n 40 "$OUT/server.log"; exit 1; }
  sleep 2
done
grep -q 'server running' "$OUT/server.log"

cd "$SIM"
ROBOTWIN_FIXED_TEST_SEEDS="$SEEDS" ROBOTWIN_TEST_NUM=3 ROBOTWIN_POLICY_PORT="$PORT" \
  ROBOTWIN_UNNORM_KEY=new_embodiment \
  bash "$ROOT/examples/Robotwin/eval_files/eval.sh" click_alarmclock demo_clean "$TAG" 0 "$GPU" "$CKPT" "$PORT" 127.0.0.1 \
  >"$OUT/eval.log" 2>&1
echo "$OUT"
