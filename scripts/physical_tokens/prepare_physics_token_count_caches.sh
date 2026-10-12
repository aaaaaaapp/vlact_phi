#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="${PHYSICS_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_full.npz}"
action_cache="${ACTION_TOKEN_CACHE:-artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz}"
read -r -a token_counts <<<"${PHYSICS_TOKEN_COUNTS:-16 64 256}"
device="${PHYSICS_EXPORT_DEVICE:-cuda:0}"

for count in "${token_counts[@]}"; do
  checkpoint="artifacts/physical_language_tokenizers/ema_${count}x128_steps1000/step-00001000.pt"
  physics_cache="artifacts/token_caches/click_alarmclock_oracle_physics_ema_${count}x128_step1000.npz"
  aligned_cache="artifacts/token_caches/click_alarmclock_pred_action4_oracle_physics${count}.npz"
  shuffled_cache="artifacts/token_caches/click_alarmclock_pred_action4_shuffled_physics${count}_seed42.npz"

  "${python_bin}" scripts/physical_tokens/export_physics_token_ids.py \
    --input "${feature_cache}" \
    --checkpoint "${checkpoint}" \
    --output "${physics_cache}" \
    --batch-size 256 \
    --device "${device}"
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${action_cache}" \
    --physics-cache "${physics_cache}" \
    --output "${aligned_cache}"
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${action_cache}" \
    --physics-cache "${physics_cache}" \
    --shuffle-physics-seed 42 \
    --output "${shuffled_cache}"
done
