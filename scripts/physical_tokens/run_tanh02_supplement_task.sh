#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
SERVER_PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
TANH="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
SUP="$ROOT/logs/tanh02_pure_full8_supplement_20261030"
ORIG="$ROOT/logs/tanh02_pure_full8_node2_20261029"
TRACK="$ROOT/scripts/physical_tokens/supplement_tanh02_full8.py"
TASK=${TASK:?TASK required}
GPU_PURE=${GPU_PURE:?GPU_PURE required}
GPU_TANH=${GPU_TANH:?GPU_TANH required}
BASE_PORT=${BASE_PORT:?BASE_PORT required}
HOST_TAG=${HOST_TAG:-$(hostname)}
mkdir -p "$SUP/$TASK"

export ROBOTWIN_PATH="$SIM" ROBOTWIN_PYTHON="$PY" STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json" __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$PY"):$PATH" NINJA_PYTHON="$PY" CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0
log() { echo "[$(date -Is)] [$HOST_TAG $TASK] $*" | tee -a "$SUP/progress.log"; }

spid_pure= spid_tanh=
cleanup() {
  for pid in "$spid_pure" "$spid_tanh"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
  spid_pure= spid_tanh=
}
trap cleanup EXIT

for ((pool_index=0; ; pool_index++)); do
  status=$("$PY" "$TRACK" status "$TASK")
  needed=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["needed"])' "$status")
  if ((needed == 0)); then log "DONE paired=100"; break; fi
  pool="$SUP/$TASK/pool_${pool_index}.json"
  if [[ ! -s "$pool" ]]; then
    start_seed=$("$PY" -c 'import json,glob,sys; d=json.load(open(sys.argv[1])); seeds=d["seeds"][:]; [seeds.extend(json.load(open(p))["seeds"]+ [x["seed"] for x in json.load(open(p))["rejected"]]) for p in glob.glob(sys.argv[2])]; print(max(seeds)+1)' "$ORIG/$TASK/discovery.json" "$SUP/$TASK/pool_*.json")
    count=$((needed * 3 + 10))
    ((count < 20)) && count=20
    log "DISCOVER pool=$pool_index start=$start_seed target=$count"
    cd "$SIM"
    CUDA_VISIBLE_DEVICES="$GPU_PURE" ROBOTWIN_DISCOVERY_START_SEED="$start_seed" \
      ROBOTWIN_DISCOVERY_OUTPUT="$pool" ROBOTWIN_DISCOVERY_COUNT="$count" ROBOTWIN_DISCOVERY_MAX_CANDIDATES="$((count * 5 + 100))" \
      PYTHONPATH="$SIM:$ROOT/examples/Robotwin/eval_files:$ROOT" \
      "$PY" "$ROOT/scripts/physical_tokens/discover_robotwin_stable_seeds.py" \
        --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --policy_ckpt_path "$PURE" \
        --overrides --task_name "$TASK" --task_config demo_clean --ckpt_setting "tanh02_supplement_discovery_${pool_index}" \
        --seed 0 --policy_name model2robotwin_interface --test_num "$count" \
        >"$SUP/$TASK/pool_${pool_index}.discovery.log" 2>&1
  fi
  count=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["seeds"]))' "$pool")
  log "POOL_READY pool=$pool_index count=$count"
  for ((start=0; start<count; start+=10)); do
    status=$("$PY" "$TRACK" status "$TASK")
    needed=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["needed"])' "$status")
    if ((needed == 0)); then break; fi
    index=$((pool_index * 1000 + start))
    read -r seeds n < <("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["seeds"][int(sys.argv[2]):int(sys.argv[2])+10]; print(json.dumps(a,separators=(",",":")),len(a))' "$pool" "$start")
    run_modes=()
    for mode in pure tanh; do
      pointer="$SUP/$TASK/$mode/$index/result_path.txt"
      if [[ -s "$pointer" ]]; then
        read -r result_dir <"$pointer"
        if [[ -s "$result_dir/_fixed_seeds.json" ]]; then continue; fi
      fi
      run_modes+=("$mode")
    done
    if ((${#run_modes[@]} == 0)); then continue; fi
    for mode in "${run_modes[@]}"; do
      dir="$SUP/$TASK/$mode/$index"
      mkdir -p "$dir"
      ckpt="$PURE"; gpu="$GPU_PURE"; port="$BASE_PORT"
      if [[ "$mode" == tanh ]]; then ckpt="$TANH"; gpu="$GPU_TANH"; port=$((BASE_PORT+1)); fi
      if nc -z 127.0.0.1 "$port" 2>/dev/null; then log "ERROR occupied port=$port"; exit 2; fi
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$SERVER_PY" deployment/model_server/server_policy_starvla_compat.py \
        --ckpt_path "$ckpt" --port "$port" --use_bf16 >"$dir/server.log" 2>&1 &
      if [[ "$mode" == pure ]]; then spid_pure=$!; else spid_tanh=$!; fi
    done
    for mode in "${run_modes[@]}"; do
      dir="$SUP/$TASK/$mode/$index"
      pid="$spid_pure"; [[ "$mode" == tanh ]] && pid="$spid_tanh"
      for _ in $(seq 1 300); do
        if grep -q 'server running' "$dir/server.log"; then break; fi
        kill -0 "$pid" 2>/dev/null || { log "ERROR server died $mode index=$index"; exit 1; }
        sleep 2
      done
      grep -q 'server running' "$dir/server.log"
    done
    eval_pids=()
    for mode in "${run_modes[@]}"; do
      dir="$SUP/$TASK/$mode/$index"
      ckpt="$PURE"; gpu="$GPU_PURE"; port="$BASE_PORT"
      if [[ "$mode" == tanh ]]; then ckpt="$TANH"; gpu="$GPU_TANH"; port=$((BASE_PORT+1)); fi
      tag="tanh02_supplement_20261030_${TASK}_${mode}_${index}"
      (
        cd "$SIM"
        ROBOTWIN_FIXED_TEST_SEEDS="$seeds" ROBOTWIN_TEST_NUM="$n" ROBOTWIN_POLICY_PORT="$port" ROBOTWIN_UNNORM_KEY=new_embodiment \
          bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$TASK" demo_clean "$tag" 0 "$gpu" "$ckpt" "$port" 127.0.0.1
      ) >"$dir/eval.log" 2>&1 &
      eval_pids+=("$!")
    done
    rc=0
    for pid in "${eval_pids[@]}"; do wait "$pid" || rc=1; done
    cleanup
    if ((rc != 0)); then log "ERROR eval index=$index; rerun this task to resume"; exit 1; fi
    for mode in "${run_modes[@]}"; do
      dir="$SUP/$TASK/$mode/$index"
      relative=$(grep -o 'wrote .*_fixed_seeds.json' "$dir/eval.log" | tail -n 1 | sed 's/^wrote //; s#/_fixed_seeds.json$##' || true)
      if [[ -z "$relative" || ! -s "$SIM/$relative/_fixed_seeds.json" ]]; then
        log "ERROR missing manifest $mode index=$index"; exit 1
      fi
      echo "$SIM/$relative" >"$dir/result_path.txt"
    done
    log "BATCH_DONE index=$index $("$PY" "$TRACK" status "$TASK")"
  done
done
