#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
output="artifacts/evaluations/d14_history_predictable_residual_ridge.json"

if [[ -s "${output}" ]]; then
  echo "D14 refused to overwrite evaluation: ${output}" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "${python_bin}" -u \
  scripts/physical_tokens/probe_observation_predictable_residual.py \
  --visual-feature-cache artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz \
  --residual-action-cache artifacts/action_caches/click_alarmclock_pred_action4_residual_wrap32.npz \
  --action-token-cache artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz \
  --action-tokenizer-checkpoint artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output "${output}" --projection-dim 512 --history-offsets 0 2 4 8 \
  --seed 42 --device cuda:0
