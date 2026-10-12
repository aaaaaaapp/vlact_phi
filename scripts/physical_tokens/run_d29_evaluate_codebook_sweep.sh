#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz"
sidecar="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz"
k128_checkpoint="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x128_steps8000/step-00008000.pt"
k256_checkpoint="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt"
k128_report="artifacts/evaluations/d29_full_diversity_ema_256x128_steps8000.json"
k256_report="artifacts/evaluations/d29_full_diversity_ema_256x256_steps8000.json"
gate_report="artifacts/evaluations/d29_full_diversity_codebook_gate.json"

for required in "${feature_cache}" "${sidecar}" "${k128_checkpoint}" "${k256_checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    echo "D29 evaluation input is missing or empty: ${required}" >&2
    exit 1
  fi
done

pids=()
for spec in "128:3:${k128_checkpoint}:${k128_report}" "256:4:${k256_checkpoint}:${k256_report}"; do
  IFS=: read -r codebook gpu checkpoint report <<<"${spec}"
  if [[ -s "${report}" ]]; then
    echo "D29 K${codebook} report already complete: ${report}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${feature_cache}" --action-target-cache "${sidecar}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${report}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
"${python_bin}" -u scripts/physical_tokens/require_d29_codebook_gate.py \
  --k128 "${k128_report}" --k256 "${k256_report}" --output "${gate_report}"
