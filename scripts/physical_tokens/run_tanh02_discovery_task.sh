#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
SIM=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013
PY=/mnt/jfzn/liujn/robotwin_env/bin/python
PURE=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
RUN_TAG=tanh02_pure_full8_node2_20261029
TASK=${TASK:?TASK required}
GPU=${GPU:?GPU required}
OUT="$ROOT/logs/$RUN_TAG/$TASK"
mkdir -p "$OUT"
if [[ -s "$OUT/discovery.json" ]] && "$PY" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))["seeds"])==100' "$OUT/discovery.json"; then exit 0; fi
export PATH="$(dirname "$PY"):$PATH" NINJA_PYTHON="$PY" CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0
export ROBOTWIN_PATH="$SIM" SAPIEN_DISABLE_RAYTRACING=0 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
VULKAN_DIR=/mnt/jfzn/liujn/robotwin_env/lib/python3.10/site-packages/sapien/vulkan_library
export VK_ICD_FILENAMES="$VULKAN_DIR/nvidia_icd.json" __EGL_VENDOR_LIBRARY_FILENAMES="$VULKAN_DIR/10_nvidia.json"
export PYTHONPATH="$SIM:$ROOT/examples/Robotwin/eval_files:$ROOT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU"
cd "$SIM"
ROBOTWIN_DISCOVERY_OUTPUT="$OUT/discovery.json" ROBOTWIN_DISCOVERY_COUNT=100 \
  "$PY" "$ROOT/scripts/physical_tokens/discover_robotwin_stable_seeds.py" \
  --config "$ROOT/examples/Robotwin/eval_files/deploy_policy.yml" --policy_ckpt_path "$PURE" \
  --overrides --task_name "$TASK" --task_config demo_clean --ckpt_setting "${RUN_TAG}_discovery" \
  --seed 0 --policy_name model2robotwin_interface --test_num 100 >"$OUT/discovery.log" 2>&1
"$PY" -c 'import json,sys; assert len(json.load(open(sys.argv[1]))["seeds"])==100' "$OUT/discovery.json"
echo "[$(date -Is)] DISCOVERY_DONE $TASK $(hostname) GPU=$GPU" >>"$ROOT/logs/$RUN_TAG/workers/progress.log"
