#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
action_cache="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
visual_cache="artifacts/future_feature_caches/robotwin_d15_8tasks_all_episodes_current1x1.npz"
tokenizer="artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt"
token_cache="artifacts/token_caches/robotwin_d16_8tasks_oracle_shared_ema4x128.npz"
residual_cache="artifacts/action_caches/robotwin_d16_8tasks_oracle_shared_ema4x128_residual_wrap32.npz"
split="artifacts/splits/click_alarmclock_seed42.json"

if [[ ! -s "${token_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_action_token_ids.py \
    --input "${action_cache}" --checkpoint "${tokenizer}" --output "${token_cache}" \
    --batch-size 512 --num-workers 8 --device cuda:0
fi
if [[ ! -s "${residual_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_action_token_residual_cache.py \
    --action-cache "${action_cache}" --predicted-token-cache "${token_cache}" \
    --action-tokenizer-checkpoint "${tokenizer}" --output "${residual_cache}" \
    --batch-size 512 --device cuda:0
fi
"${python_bin}" -u scripts/physical_tokens/probe_observation_predictable_residual.py \
  --visual-feature-cache "${visual_cache}" --residual-action-cache "${residual_cache}" \
  --action-token-cache "${token_cache}" --action-tokenizer-checkpoint "${tokenizer}" \
  --split-manifest "${split}" --projection-dim 512 --history-offsets 0 --seed 42 \
  --device cuda:0 --output artifacts/evaluations/d16_shared_action_current_residual_ridge.json
