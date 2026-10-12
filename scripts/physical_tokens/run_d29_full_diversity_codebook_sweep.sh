#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
feature_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz"
sidecar="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz"

for required in "${feature_cache}" "${feature_cache}.manifest.json" "${sidecar}" "${sidecar}.manifest.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "D29 required input is missing or empty: ${required}" >&2
    exit 1
  fi
done

pids=()
codebooks=(128 256)
gpus=(3 4)
for index in "${!codebooks[@]}"; do
  codebook="${codebooks[$index]}"
  output="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x${codebook}_steps8000"
  checkpoint="${output}/step-00008000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "D29 K${codebook} checkpoint already complete: ${checkpoint}"
    continue
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D29 refused partial output directory: ${output}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
  SPATIOTEMPORAL_NPROC=1 SPATIOTEMPORAL_BATCH_SIZE=32 \
  SPATIOTEMPORAL_FEATURE_CACHE="${feature_cache}" \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${sidecar}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
  SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_CODEBOOK_SIZE="${codebook}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES=5,20,25,39,49 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=0 \
  SPATIOTEMPORAL_MAX_STEPS=8000 \
  SPATIOTEMPORAL_EVAL_INTERVAL=500 \
  SPATIOTEMPORAL_SAVE_INTERVAL=2000 \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  SPATIOTEMPORAL_QUERIES_PER_TRANSITION=16 \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh \
    >"${output}.log" 2>&1 &
  pids+=("$!")
  echo "started D29 K${codebook} on GPU ${gpus[$index]}, pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
