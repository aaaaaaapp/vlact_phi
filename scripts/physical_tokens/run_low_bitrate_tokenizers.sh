#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/jfzn/liujn/code/VLAct-PhysToken}"
python_bin="${VLACT_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
input_cache="${ACTION_CACHE:-${repo_root}/artifacts/action_caches/click_alarmclock_clean_wrap32.npz}"
max_steps="${MAX_STEPS:-5000}"
output_root="${OUTPUT_ROOT:-${repo_root}/artifacts/action_tokenizer_low_bitrate}"

mkdir -p "${output_root}"
cd "${repo_root}"

CUDA_VISIBLE_DEVICES="${TOKENIZER_4X128_GPU:-0}" "${python_bin}" -u \
  scripts/physical_tokens/train_action_tokenizer.py \
  --input "${input_cache}" \
  --output-dir "${output_root}/ema_4x128_steps${max_steps}" \
  --quantizer ema_vq --num-tokens 4 --codebook-size 128 \
  --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
  --max-steps "${max_steps}" --eval-interval 500 --save-interval "${max_steps}" \
  > "${output_root}/ema_4x128_steps${max_steps}.log" 2>&1 &
pid_4x128=$!

CUDA_VISIBLE_DEVICES="${TOKENIZER_2X64_GPU:-4}" "${python_bin}" -u \
  scripts/physical_tokens/train_action_tokenizer.py \
  --input "${input_cache}" \
  --output-dir "${output_root}/ema_2x64_steps${max_steps}" \
  --quantizer ema_vq --num-tokens 2 --codebook-size 64 \
  --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
  --max-steps "${max_steps}" --eval-interval 500 --save-interval "${max_steps}" \
  > "${output_root}/ema_2x64_steps${max_steps}.log" 2>&1 &
pid_2x64=$!

status=0
wait "${pid_4x128}" || status=$?
wait "${pid_2x64}" || status=$?
exit "${status}"
