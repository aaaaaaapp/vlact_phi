#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
RUN_TAG=${RUN_TAG:?RUN_TAG required}
COUNT=${COUNT:-100}
GPU=${GPU:-4}
OUT="$ROOT/logs/$RUN_TAG"
TASKS=(beat_block_hammer open_microwave shake_bottle move_playingcard_away pick_dual_bottles handover_block place_object_basket)

[[ "$(hostname)" == *node002* ]] || { echo "Run only on node002" >&2; exit 2; }
export PATH="$(dirname "$PY"):$PATH" NINJA_PYTHON="$PY" CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0
export ROBOTWIN_PATH="$SIM" SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export PYTHONPATH="$SIM:$ROOT/examples/Robotwin/eval_files:$ROOT" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"

for task in "${TASKS[@]}"; do
  mkdir -p "$OUT/$task"
  discovery="$OUT/$task/discovery.json"
  if [[ -s "$discovery" ]] && [[ "$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["seeds"]))' "$discovery")" == "$COUNT" ]]; then
    continue
  fi
  echo "[$(date -Is)] PREFETCH $task GPU=$GPU" | tee -a "$OUT/prefetch.log"
  cd "$SIM"
  ROBOTWIN_DISCOVERY_OUTPUT="$discovery" ROBOTWIN_DISCOVERY_COUNT="$COUNT" \
    "$PY" "$ROOT/scripts/physical_tokens/discover_robotwin_stable_seeds.py" \
      --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --policy_ckpt_path "$PURE" \
      --overrides --task_name "$task" --task_config demo_clean --ckpt_setting "${RUN_TAG}_discovery" \
      --seed 0 --policy_name model2robotwin_interface --test_num "$COUNT" \
      >"$OUT/$task/discovery.log" 2>&1
  echo "[$(date -Is)] PREFETCH_DONE $task" | tee -a "$OUT/prefetch.log"
done
