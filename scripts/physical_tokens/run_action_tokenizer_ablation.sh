#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/jfzn/liujn/code/VLAct-PhysToken}"
PYTHON_BIN="${VLAct_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
INPUT_CACHE="${ACTION_CACHE:-${REPO_ROOT}/artifacts/action_caches/click_alarmclock_clean_wrap32.npz}"
MAX_STEPS="${MAX_STEPS:-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/action_tokenizer_runs}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${EMA_GPU:-0}" "${PYTHON_BIN}" -u \
  scripts/physical_tokens/train_action_tokenizer.py \
  --input "${INPUT_CACHE}" \
  --output-dir "${OUTPUT_ROOT}/ema_vq_steps${MAX_STEPS}" \
  --quantizer ema_vq --num-tokens 8 --codebook-size 512 \
  --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
  --max-steps "${MAX_STEPS}" --eval-interval 500 --save-interval "${MAX_STEPS}" \
  > "${OUTPUT_ROOT}/ema_vq_steps${MAX_STEPS}.log" 2>&1 &
EMA_PID=$!

CUDA_VISIBLE_DEVICES="${FSQ_GPU:-1}" "${PYTHON_BIN}" -u \
  scripts/physical_tokens/train_action_tokenizer.py \
  --input "${INPUT_CACHE}" \
  --output-dir "${OUTPUT_ROOT}/fsq_steps${MAX_STEPS}" \
  --quantizer fsq --num-tokens 8 --fsq-levels 8,8,8 \
  --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
  --max-steps "${MAX_STEPS}" --eval-interval 500 --save-interval "${MAX_STEPS}" \
  > "${OUTPUT_ROOT}/fsq_steps${MAX_STEPS}.log" 2>&1 &
FSQ_PID=$!

status=0
wait "${EMA_PID}" || status=$?
wait "${FSQ_PID}" || status=$?
exit "${status}"
