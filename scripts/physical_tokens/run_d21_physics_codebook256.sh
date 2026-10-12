#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

feature_cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz"
sidecar="artifacts/future_feature_caches/robotwin_d20_8tasks_8episodes_predictable_residuals17x14.npz"
output="artifacts/physical_language_tokenizers/d21_predictable_residual_ema_256tokens_codebook256_steps1000"
checkpoint="${output}/step-00001000.pt"

for required in "${feature_cache}" "${sidecar}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing D21 input: ${required}" >&2
    exit 1
  fi
done
if [[ -s "${checkpoint}" ]]; then
  echo "D21 checkpoint already complete: ${checkpoint}"
  exit 0
fi
if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "D21 refused partial output directory: ${output}" >&2
  exit 2
fi

SPATIOTEMPORAL_NPROC=1 SPATIOTEMPORAL_BATCH_SIZE=32 \
SPATIOTEMPORAL_FEATURE_CACHE="${feature_cache}" \
SPATIOTEMPORAL_ACTION_TARGET_CACHE="${sidecar}" \
SPATIOTEMPORAL_ACTION_AUX_WEIGHT=0.25 \
SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=300 \
SPATIOTEMPORAL_MAX_STEPS=1000 \
SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
SPATIOTEMPORAL_QUERIES_PER_TRANSITION=16 \
SPATIOTEMPORAL_CODEBOOK_SIZE=256 \
bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
