#!/usr/bin/env bash
set -euo pipefail

# Discovery only: estimate the task's pure-Pi success floor on a wider, fixed
# seed panel.  Do not use this panel alone for the final method claim.
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
STAR=/mnt/jfzn/liujn/code/Starvla/Starvla
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
RPATH=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval
TASK=click_alarmclock
SEEDS=${SEEDS:-'[100000,100001,100002,100003,100004,100005,100006,100007,100008,100009]'}
GPU=${GPU_ID:-0}
PORT=${PORT:-6799}
CKPT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
TAG=${TAG:-pure_pi_seed_discovery_10}
LOGDIR="$ROOT/logs/pure_seed_discovery_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$STAR" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$PY" deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  >"$LOGDIR/server.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  grep -q 'server running' "$LOGDIR/server.log" && break
  kill -0 "$SPID" 2>/dev/null || { tail -80 "$LOGDIR/server.log"; exit 1; }
  sleep 2
done
grep -q 'server running' "$LOGDIR/server.log"
cd "$RPATH"
ROBOTWIN_FIXED_TEST_SEEDS="$SEEDS" ROBOTWIN_TEST_NUM=10 ROBOTWIN_PATH="$RPATH" ROBOTWIN_PYTHON="$RPY" \
  ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment STARVLA_PATH="$STAR" \
  bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$TASK" demo_clean "$TAG" 0 "$GPU" "$CKPT" "$PORT" 127.0.0.1 \
  >"$LOGDIR/eval.log" 2>&1
echo "result log: $LOGDIR/eval.log"
