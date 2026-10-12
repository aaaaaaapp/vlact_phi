#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

feature_cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz"
sidecar="artifacts/future_feature_caches/robotwin_d20_8tasks_8episodes_predictable_residuals17x14.npz"
codebooks=(128 256)
gpus=(6 7)
pids=()

for index in "${!codebooks[@]}"; do
  codebook="${codebooks[$index]}"
  output="artifacts/physical_language_tokenizers/d22_predictable_residual_ema_256tokens_codebook${codebook}_cap1000_steps1000"
  checkpoint="${output}/step-00001000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "D22 K=${codebook} checkpoint already complete: ${checkpoint}"
    continue
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D22 refused partial output directory: ${output}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  SPATIOTEMPORAL_NPROC=1 SPATIOTEMPORAL_BATCH_SIZE=32 \
  SPATIOTEMPORAL_FEATURE_CACHE="${feature_cache}" \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${sidecar}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
  SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=1000 \
  SPATIOTEMPORAL_MAX_STEPS=1000 \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  SPATIOTEMPORAL_QUERIES_PER_TRANSITION=16 \
  SPATIOTEMPORAL_CODEBOOK_SIZE="${codebook}" \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh \
    >"${output}.log" 2>&1 &
  pids+=("$!")
  echo "started D22 K=${codebook}, cap=1000 on GPU ${gpus[$index]}, pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
