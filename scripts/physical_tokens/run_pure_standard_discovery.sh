#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
STAR=/mnt/jfzn/liujn/code/Starvla/Starvla
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
RPATH=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval
TASK=${TASK:?TASK is required}
MODE=${MODE:-demo_randomized}
GPU=${GPU_ID:?GPU_ID is required}
PORT=${PORT:?PORT is required}
N=${N:-20}
TAG=${TAG:-pure_standard_${TASK}}
EVAL_SEED=${EVAL_SEED:-0}
CKPT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
LOGDIR="$ROOT/logs/standard_discovery_${TASK}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$STAR" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$PY" deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$CKPT" --port "$PORT" --use_bf16 >"$LOGDIR/server.log" 2>&1 &
SPID=$!; trap 'kill "$SPID" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do grep -q 'server running' "$LOGDIR/server.log" && break; kill -0 "$SPID" 2>/dev/null || exit 1; sleep 2; done
grep -q 'server running' "$LOGDIR/server.log"
cd "$RPATH"
ROBOTWIN_TEST_NUM="$N" ROBOTWIN_PATH="$RPATH" ROBOTWIN_PYTHON="$RPY" ROBOTWIN_POLICY_PORT="$PORT" \
  ROBOTWIN_UNNORM_KEY=new_embodiment STARVLA_PATH="$STAR" \
  bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$TASK" "$MODE" "$TAG" "$EVAL_SEED" "$GPU" "$CKPT" "$PORT" 127.0.0.1 >"$LOGDIR/eval.log" 2>&1
echo "$LOGDIR"
