#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

read -r -a token_counts <<<"${PHYSICS_TOKEN_COUNTS:-16 32 64 128 256}"
read -r -a gpu_ids <<<"${PHYSICS_GPU_IDS:-0 1 2 3 4}"
if [[ "${#token_counts[@]}" -ne "${#gpu_ids[@]}" ]]; then
  echo "PHYSICS_TOKEN_COUNTS and PHYSICS_GPU_IDS must have equal lengths" >&2
  exit 2
fi
max_steps="${PHYSICS_MAX_STEPS:-1000}"
codebook_size="${PHYSICS_CODEBOOK_SIZE:-128}"
output_suffix="${PHYSICS_OUTPUT_SUFFIX:-}"
# A single GPU sees the same global batch as the earlier 8 x 32 runs.
batch_size="${PHYSICS_BATCH_SIZE:-256}"
status=0
pids=()

for index in "${!token_counts[@]}"; do
  count="${token_counts[$index]}"
  gpu="${gpu_ids[$index]}"
  output="artifacts/physical_language_tokenizers/ema_${count}x${codebook_size}_steps${max_steps}${output_suffix}"
  log="${output}/train.log"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PHYSICS_NPROC=1 \
    PHYSICS_NUM_TOKENS="${count}" \
    PHYSICS_CODEBOOK_SIZE="${codebook_size}" \
    PHYSICS_MAX_STEPS="${max_steps}" \
    PHYSICS_BATCH_SIZE="${batch_size}" \
    PHYSICS_OUTPUT_DIR="${output}" \
    bash scripts/physical_tokens/run_physical_language_tokenizer.sh >"${log}" 2>&1 &
  pids+=("$!")
  echo "started ${count} tokens on GPU ${gpu}: pid=${pids[-1]} log=${log}"
done

for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${token_counts[$index]}-token run failed" >&2
    status=1
  fi
done
exit "${status}"
