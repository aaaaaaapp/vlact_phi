#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
STAR=/mnt/jfzn/liujn/code/Starvla/Starvla
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
RPATH=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval
GPU=${GPU_ID:-2}
PORT=${PORT:-6810}
SEEDS='[100000,100001,100002,100003,100004,100005,100006,100007,100008,100009]'
TASKS=(click_alarmclock hanging_mug)

PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
ACTION="$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
JOINT="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"

run_one() {
  local task="$1" tag="$2" ckpt="$3"
  local logdir="$ROOT/logs/p64_rollout_${task}_${tag}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$logdir"
  local slog="$logdir/server.log" elog="$logdir/eval.log"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$STAR" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
    "$PY" deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$ckpt" --port "$PORT" --use_bf16 >"$slog" 2>&1 &
  local spid=$!
  cleanup_server() { kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; }
  trap cleanup_server RETURN
  for _ in $(seq 1 300); do
    grep -q 'server running' "$slog" && break
    kill -0 "$spid" 2>/dev/null || { tail -60 "$slog"; return 1; }
    sleep 2
  done
  grep -q 'server running' "$slog"
  cd "$RPATH"
    ROBOTWIN_FIXED_TEST_SEEDS="$SEEDS" ROBOTWIN_TEST_NUM=10 ROBOTWIN_PATH="$RPATH" ROBOTWIN_PYTHON="$RPY" \
    ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment STARVLA_PATH="$STAR" \
    bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$task" demo_clean "$tag" 0 "$GPU" "$ckpt" "$PORT" 127.0.0.1 >"$elog" 2>&1
  cleanup_server
  trap - RETURN
  PORT=$((PORT + 1))
}

for task in "${TASKS[@]}"; do
  run_one "$task" p64_fixed_pure_pi "$PURE"
  run_one "$task" p64_fixed_actiononly "$ACTION"
  run_one "$task" p64_fixed_joint_hard "$JOINT"
done
