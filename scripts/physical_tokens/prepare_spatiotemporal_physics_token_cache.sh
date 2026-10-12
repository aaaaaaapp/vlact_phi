#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="${SPATIOTEMPORAL_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz}"
checkpoint="${SPATIOTEMPORAL_TOKENIZER_CHECKPOINT:-artifacts/physical_language_tokenizers/spatiotemporal_ema_256x128_steps1000/step-00001000.pt}"
action_cache="${ACTION_TOKEN_CACHE:-artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz}"
physics_cache="${SPATIOTEMPORAL_PHYSICS_CACHE:-artifacts/token_caches/click_alarmclock_oracle_physics_spatiotemporal_ema_256x128_step1000.npz}"
aligned_cache="${SPATIOTEMPORAL_ALIGNED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz}"
shuffled_cache="${SPATIOTEMPORAL_SHUFFLED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_shuffled_spatiotemporal_physics256_seed42.npz}"

"${python_bin}" scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
  --input "${feature_cache}" \
  --checkpoint "${checkpoint}" \
  --output "${physics_cache}" \
  --batch-size "${SPATIOTEMPORAL_EXPORT_BATCH_SIZE:-16}" \
  --device "${SPATIOTEMPORAL_EXPORT_DEVICE:-cuda:0}"

"${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
  --action-cache "${action_cache}" \
  --physics-cache "${physics_cache}" \
  --output "${aligned_cache}"

"${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
  --action-cache "${action_cache}" \
  --physics-cache "${physics_cache}" \
  --shuffle-physics-seed 42 \
  --output "${shuffled_cache}"
