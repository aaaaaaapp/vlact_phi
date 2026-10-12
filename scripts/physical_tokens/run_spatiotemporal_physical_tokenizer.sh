#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
input="${SPATIOTEMPORAL_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz}"
output="${SPATIOTEMPORAL_OUTPUT_DIR:-artifacts/physical_language_tokenizers/spatiotemporal_ema_256x128_steps1000}"
action_args=()
if [[ -n "${SPATIOTEMPORAL_ACTION_TARGET_CACHE:-}" ]]; then
  action_args+=(
    --action-target-cache "${SPATIOTEMPORAL_ACTION_TARGET_CACHE}"
    --action-aux-weight "${SPATIOTEMPORAL_ACTION_AUX_WEIGHT:-0.25}"
    --action-velocity-weight "${SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT:-0.25}"
    --action-gripper-weight "${SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT:-0.1}"
    --action-aux-target-mode "${SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE:-absolute}"
  )
fi

"${python_bin}" -m torch.distributed.run --standalone \
  --nproc_per_node="${SPATIOTEMPORAL_NPROC:-8}" \
  scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
  --input "${input}" \
  --output-dir "${output}" \
  "${action_args[@]}" \
  --quantizer "${SPATIOTEMPORAL_QUANTIZER:-ema_vq}" \
  --codebook-size "${SPATIOTEMPORAL_CODEBOOK_SIZE:-128}" \
  --queries-per-transition "${SPATIOTEMPORAL_QUERIES_PER_TRANSITION:-16}" \
  --token-routing "${SPATIOTEMPORAL_TOKEN_ROUTING:-global_qformer}" \
  --batch-size "${SPATIOTEMPORAL_BATCH_SIZE:-4}" \
  --max-steps "${SPATIOTEMPORAL_MAX_STEPS:-1000}" \
  --eval-interval "${SPATIOTEMPORAL_EVAL_INTERVAL:-100}" \
  --save-interval "${SPATIOTEMPORAL_SAVE_INTERVAL:-500}" \
  --learning-rate "${SPATIOTEMPORAL_LEARNING_RATE:-3e-4}" \
  --validation-episodes "${SPATIOTEMPORAL_VALIDATION_EPISODES:-5,20,25,39,49}" \
  --validation-episodes-per-task "${SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK:-0}" \
  --max-train-clips-per-task "${SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK:-0}"
