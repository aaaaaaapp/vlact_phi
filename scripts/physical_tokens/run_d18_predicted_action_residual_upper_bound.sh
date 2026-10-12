#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
action_cache="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
visual_cache="artifacts/future_feature_caches/robotwin_d15_8tasks_all_episodes_current1x1.npz"
tokenizer="artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt"
predicted_cache="artifacts/token_caches/robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz"
residual_cache="artifacts/action_caches/robotwin_d18_8tasks_predicted_action_residual_wrap32.npz"
split="artifacts/splits/click_alarmclock_seed42.json"

if [[ ! -s "${predicted_cache}" ]]; then
  echo "missing merged D17 predicted-token cache: ${predicted_cache}" >&2
  exit 1
fi
if [[ ! -s "${residual_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_action_token_residual_cache.py \
    --action-cache "${action_cache}" --predicted-token-cache "${predicted_cache}" \
    --action-tokenizer-checkpoint "${tokenizer}" --output "${residual_cache}" \
    --batch-size 512 --device cuda:0
fi
"${python_bin}" -u scripts/physical_tokens/probe_observation_predictable_residual.py \
  --visual-feature-cache "${visual_cache}" --residual-action-cache "${residual_cache}" \
  --action-token-cache "${predicted_cache}" --action-tokenizer-checkpoint "${tokenizer}" \
  --split-manifest "${split}" --projection-dim 512 --history-offsets 0 --seed 42 \
  --device cuda:0 --output artifacts/evaluations/d18_predicted_action_current_residual_ridge.json
"${python_bin}" -u scripts/physical_tokens/require_continuous_residual_gate.py \
  artifacts/evaluations/d18_predicted_action_current_residual_ridge.json \
  --minimum-relative-improvement 0.01 \
  --output artifacts/evaluations/d18_predicted_action_current_residual_gate.json
