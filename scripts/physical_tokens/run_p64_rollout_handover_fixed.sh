#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
STAR=/mnt/jfzn/liujn/code/Starvla/Starvla
PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
RPY=/mnt/jfzn/liujn/robotwin_env/bin/python
RPATH=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval
TASK=handover_block
SEEDS='[100000,100002,100003]'
GPU=${GPU_ID:-0}
PORT=${PORT:-6799}
run_one() {
  local tag="$1" ckpt="$2" extra="${3:-}"
  local logdir="$ROOT/logs/p64_handover_rollout_${tag}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$logdir"
  local slog="$logdir/server.log" elog="$logdir/eval.log"
  echo "[$(date -Is)] starting $tag" | tee -a "$ROOT/logs/p64_rollout_master.log"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$STAR" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "$PY" deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$ckpt" --port "$PORT" --use_bf16 $extra > "$slog" 2>&1 &
  local spid=$!
  trap 'kill "$spid" 2>/dev/null || true' RETURN
  for i in $(seq 1 300); do grep -q 'server running' "$slog" && break; kill -0 "$spid" 2>/dev/null || { tail -40 "$slog"; return 1; }; sleep 2; done
  grep -q 'server running' "$slog"
  cd "$RPATH"
  ROBOTWIN_FIXED_TEST_SEEDS="$SEEDS" ROBOTWIN_TEST_NUM=3 ROBOTWIN_PATH="$RPATH" ROBOTWIN_PYTHON="$RPY" ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment STARVLA_PATH="$STAR" bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$TASK" demo_clean "$tag" 0 "$GPU" "$ckpt" "$PORT" 127.0.0.1 > "$elog" 2>&1
  kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true
  echo "[$(date -Is)] finished $tag" | tee -a "$ROOT/logs/p64_rollout_master.log"
  PORT=$((PORT+1))
}
run_one p64_handover_pure_pi /mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
run_one p64_handover_actiononly "$ROOT/results/Checkpoints/qwenpi_v4_p64_actiononly_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
run_one p64_handover_joint_hard "$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
