#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
SERVER_PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
TANH="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
RUN_TAG=${RUN_TAG:-tanh02_pure_full8_node2_$(date +%Y%m%d_%H%M%S)}
OUT="$ROOT/logs/$RUN_TAG"
COUNT=${COUNT:-100}
CHUNK=${CHUNK:-20}
GPU_PURE=${GPU_PURE:-0}
GPU_TANH=${GPU_TANH:-7}
BASE_PORT=${BASE_PORT:-8100}
SEED_MANIFEST=${SEED_MANIFEST:-}
TASKS=(click_alarmclock beat_block_hammer open_microwave shake_bottle move_playingcard_away pick_dual_bottles handover_block place_object_basket)

[[ "$(hostname)" == *node002* ]] || { echo "Run only on node002" >&2; exit 2; }
[[ -s "$PURE" && -s "$TANH" && -d "$SIM/assets/background_texture/unseen" ]] || { echo "Missing checkpoint or simulator assets" >&2; exit 2; }
[[ "$COUNT" =~ ^[1-9][0-9]*$ && "$CHUNK" =~ ^[1-9][0-9]*$ ]] || { echo "COUNT/CHUNK must be positive integers" >&2; exit 2; }
mkdir -p "$OUT"
export ROBOTWIN_PATH="$SIM" ROBOTWIN_PYTHON="$PY" STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$PY"):$PATH" NINJA_PYTHON="$PY" CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.0

log() { echo "[$(date -Is)] $*" | tee -a "$OUT/progress.log"; }
run_batch() {
  local task="$1" mode="$2" ckpt="$3" gpu="$4" port="$5" start="$6" seeds="$7" n="$8"
  local tag="${RUN_TAG}_${task}_${mode}_${start}"
  local dir="$OUT/$task/$mode/$start"
  mkdir -p "$dir"
  if [[ -s "$dir/result_path.txt" ]]; then
    local prior
    prior=$(<"$dir/result_path.txt")
    if [[ -s "$prior/_fixed_seeds.json" ]]; then log "SKIP completed $task $mode start=$start"; return 0; fi
  fi
  if nc -z 127.0.0.1 "$port" 2>/dev/null; then log "ERROR port $port occupied"; return 2; fi
  log "START $task $mode start=$start n=$n gpu=$gpu port=$port"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT:$ROOT" "$SERVER_PY" \
    deployment/model_server/server_policy_starvla_compat.py --ckpt_path "$ckpt" --port "$port" --use_bf16 \
    >"$dir/server.log" 2>&1 &
  local spid=$!
  local rc=0
  for _ in $(seq 1 300); do
    if grep -q 'server running' "$dir/server.log"; then break; fi
    if ! kill -0 "$spid" 2>/dev/null; then rc=1; break; fi
    sleep 2
  done
  if ! grep -q 'server running' "$dir/server.log"; then rc=1; fi
  if ((rc == 0)); then
    cd "$SIM"
    ROBOTWIN_FIXED_TEST_SEEDS="$seeds" ROBOTWIN_TEST_NUM="$n" ROBOTWIN_POLICY_PORT="$port" \
      ROBOTWIN_UNNORM_KEY=new_embodiment \
      bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$task" demo_clean "$tag" 0 "$gpu" "$ckpt" "$port" 127.0.0.1 \
      >"$dir/eval.log" 2>&1 || rc=$?
  fi
  kill "$spid" 2>/dev/null || true
  wait "$spid" 2>/dev/null || true
  if ((rc == 0)); then
    local result
    result=$(grep -o 'wrote .*_fixed_seeds.json' "$dir/eval.log" | tail -n 1 | sed 's/^wrote //; s#/_fixed_seeds.json$##')
    if [[ -n "$result" && -s "$result/_fixed_seeds.json" ]]; then
      echo "$result" >"$dir/result_path.txt"
      log "DONE $task $mode start=$start result=$result"
    else
      rc=1
    fi
  fi
  if ((rc != 0)); then log "ERROR $task $mode start=$start rc=$rc logs=$dir"; fi
  return "$rc"
}

log "BEGIN count=$COUNT chunk=$CHUNK GPUs=$GPU_PURE,$GPU_TANH pure=$PURE tanh=$TANH sim=$SIM"
for task in "${TASKS[@]}"; do
  discovery="$OUT/$task/discovery.json"
  if [[ -n "$SEED_MANIFEST" && ! -s "$discovery" ]]; then
    mkdir -p "$OUT/$task"
    "$PY" -c 'import json,sys; from pathlib import Path; source=json.load(open(sys.argv[1])); task=sys.argv[2]; count=int(sys.argv[4]); seeds=source["tasks"][task]["seeds"]; assert source["complete"] and source["task_config"] == "demo_clean" and len(seeds) == count; Path(sys.argv[3]).write_text(json.dumps({"task":task,"task_config":"demo_clean","seeds":seeds,"rejected":[],"candidates_checked":len(seeds)},indent=2)+"\n")' "$SEED_MANIFEST" "$task" "$discovery" "$COUNT"
    log "REUSE_SEEDS $task source=$SEED_MANIFEST"
  fi
  if [[ ! -s "$discovery" ]]; then
    mkdir -p "$OUT/$task"
    log "DISCOVER $task count=$COUNT"
    cd "$SIM"
    ROBOTWIN_DISCOVERY_OUTPUT="$discovery" ROBOTWIN_DISCOVERY_COUNT="$COUNT" \
      PYTHONPATH="$SIM:$ROOT/examples/Robotwin/eval_files:$ROOT" \
      "$PY" "$ROOT/scripts/physical_tokens/discover_robotwin_stable_seeds.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --policy_ckpt_path "$PURE" \
        --overrides --task_name "$task" --task_config demo_clean --ckpt_setting "${RUN_TAG}_discovery" \
        --seed 0 --policy_name model2robotwin_interface --test_num "$COUNT" \
        >"$OUT/$task/discovery.log" 2>&1 || { log "ERROR discovery $task"; exit 1; }
  fi
  if [[ "$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["seeds"]))' "$discovery")" != "$COUNT" ]]; then
    log "ERROR discovery has wrong seed count for $task"; exit 1
  fi
  for ((start=0; start<COUNT; start+=CHUNK)); do
    read -r seeds n < <("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["seeds"][int(sys.argv[2]):int(sys.argv[2])+int(sys.argv[3])]; print(json.dumps(a,separators=(",",":")),len(a))' "$discovery" "$start" "$CHUNK")
    pids=()
    run_batch "$task" pure "$PURE" "$GPU_PURE" "$BASE_PORT" "$start" "$seeds" "$n" & pids+=("$!")
    run_batch "$task" tanh "$TANH" "$GPU_TANH" "$((BASE_PORT+1))" "$start" "$seeds" "$n" & pids+=("$!")
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    ((status == 0)) || { log "ERROR batch task=$task start=$start"; exit 1; }
  done
  log "TASK_DONE $task"
  "$PY" "$ROOT/scripts/physical_tokens/summarize_tanh02_pure_full8.py" "$OUT" >"$OUT/summary_progress.log"
done
log "ALL_DONE"
