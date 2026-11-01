#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
SERVER_PY=/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
TANH="$ROOT/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
RUN_TAG=tanh02_pure_full8_node2_20261029
OUT="$ROOT/logs/$RUN_TAG"
GPU=${GPU:?GPU required}
PORT=${PORT:?PORT required}
WORKER=${WORKER:-$(hostname)_gpu${GPU}}
TASKS=(click_alarmclock beat_block_hammer open_microwave shake_bottle move_playingcard_away pick_dual_bottles handover_block place_object_basket)

[[ -s "$PURE" && -s "$TANH" && -d "$SIM/assets/background_texture/unseen" ]] || exit 2
mkdir -p "$OUT/workers"
export ROBOTWIN_PATH="$SIM" ROBOTWIN_PYTHON="$PY" STARVLA_PATH="$ROOT"
export ROBOTWIN_SEND_STATE=1 ROBOTWIN_ACTION_UNNORM_MODE=wrap ROBOTWIN_SKIP_UNSTABLE_SEEDS=1
export SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$PY"):$PATH" NINJA_PYTHON="$PY" CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=8.0

log() { echo "[$(date -Is)] [$WORKER] $*" | tee -a "$OUT/workers/progress.log"; }
finish_server() {
  if [[ -n "${spid:-}" ]]; then kill "$spid" 2>/dev/null || true; wait "$spid" 2>/dev/null || true; spid=; fi
}
trap finish_server EXIT

while :; do
  pending=0
  available=0
  for task in "${TASKS[@]}"; do
    discovery="$OUT/$task/discovery.json"
    for start in 0 20 40 60 80; do
      for mode in pure tanh; do
        dir="$OUT/$task/$mode/$start"
        result="$SIM/eval_result/$task/model2robotwin_interface/demo_clean/${RUN_TAG}_${task}_${mode}_${start}"
        if [[ -s "$dir/result_path.txt" ]]; then
          read -r prior <"$dir/result_path.txt"
          if [[ -s "$prior/_fixed_seeds.json" || -s "$SIM/$prior/_fixed_seeds.json" ]]; then continue; fi
        fi
        pending=1
        [[ -s "$discovery" ]] || continue
        available=1
        mkdir -p "$dir"
        if ! mkdir "$dir/worker.lock" 2>/dev/null; then continue; fi
        # Other workers can see a finished job only after the result manifest exists.
        if [[ -s "$dir/result_path.txt" ]]; then
          read -r prior <"$dir/result_path.txt"
          if [[ -s "$prior/_fixed_seeds.json" || -s "$SIM/$prior/_fixed_seeds.json" ]]; then rmdir "$dir/worker.lock"; continue; fi
        fi
        if ! read -r seeds n < <("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["seeds"][int(sys.argv[2]):int(sys.argv[2])+20]; assert len(a)==20; print(json.dumps(a,separators=(",",":")),len(a))' "$discovery" "$start"); then
          rmdir "$dir/worker.lock"; continue
        fi
        ckpt="$PURE"; [[ "$mode" == tanh ]] && ckpt="$TANH"
        tag="${RUN_TAG}_${task}_${mode}_${start}"
        log "START $task $mode start=$start gpu=$GPU port=$PORT"
        rc=0
        if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then rc=2; fi
        if ((rc == 0)); then
          cd "$ROOT"
          CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$SERVER_PY" deployment/model_server/server_policy_starvla_compat.py \
            --ckpt_path "$ckpt" --port "$PORT" --use_bf16 >"$dir/server.log" 2>&1 &
          spid=$!
          for _ in $(seq 1 300); do
            if grep -q 'server running' "$dir/server.log"; then break; fi
            if ! kill -0 "$spid" 2>/dev/null; then rc=1; break; fi
            sleep 2
          done
          grep -q 'server running' "$dir/server.log" || rc=1
        fi
        if ((rc == 0)); then
          cd "$SIM"
          ROBOTWIN_FIXED_TEST_SEEDS="$seeds" ROBOTWIN_TEST_NUM="$n" ROBOTWIN_POLICY_PORT="$PORT" ROBOTWIN_UNNORM_KEY=new_embodiment \
            bash "$ROOT/examples/Robotwin/eval_files/eval.sh" "$task" demo_clean "$tag" 0 "$GPU" "$ckpt" "$PORT" 127.0.0.1 \
            >"$dir/eval.log" 2>&1 || rc=$?
        fi
        finish_server
        if ((rc == 0)); then
          relative=$(grep -o 'wrote .*_fixed_seeds.json' "$dir/eval.log" | tail -n 1 | sed 's/^wrote //; s#/_fixed_seeds.json$##' || true)
          if [[ -n "$relative" && -s "$SIM/$relative/_fixed_seeds.json" ]]; then
            echo "$SIM/$relative" >"$dir/result_path.txt"
            log "DONE $task $mode start=$start"
          else rc=1; fi
        fi
        if ((rc != 0)); then log "ERROR $task $mode start=$start rc=$rc"; fi
        rmdir "$dir/worker.lock"
        sleep 1
      done
    done
  done
  if ((pending == 0)); then log ALL_DONE; break; fi
  if ((available == 0)); then sleep 30; else sleep 2; fi
done
