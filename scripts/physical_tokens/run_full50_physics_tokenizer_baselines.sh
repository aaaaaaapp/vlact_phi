#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
output_root="artifacts/robotwin_full50_study/physics_tokenizers"
steps="${ROBOTWIN_PHYSICS_TRAIN_STEPS:-20000}"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_TRAIN_GPUS:-4,5,7}"

if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Merged 50-task physics cache and manifest are required: ${cache}" >&2
  exit 2
fi
if [[ "${#gpus[@]}" -ne 3 ]]; then
  echo "ROBOTWIN_PHYSICS_TRAIN_GPUS must contain three GPU IDs" >&2
  exit 2
fi

feature_cache_args=()
if [[ -n "${ROBOTWIN_FULL50_FRAME_FEATURES_NPY:-}" ]]; then
  if [[ ! -s "${ROBOTWIN_FULL50_FRAME_FEATURES_NPY}" ]]; then
    echo "Missing frame feature NPY: ${ROBOTWIN_FULL50_FRAME_FEATURES_NPY}" >&2
    exit 2
  fi
  feature_cache_args=(--feature-array-cache "${ROBOTWIN_FULL50_FRAME_FEATURES_NPY}")
fi

names=(ema_k256 fsq_levels488 fsq_levels4444)
quantizers=(ema_vq fsq fsq)
levels=(4,8,8 4,8,8 4,4,4,4)
pids=()
mkdir -p "${output_root}"
for i in "${!names[@]}"; do
  output="${output_root}/${names[$i]}_steps${steps}"
  if [[ -s "${output}/step-$(printf '%08d' "${steps}").pt" ]]; then
    echo "Already complete: ${output}"
    continue
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing partial training output (manual review required): ${output}" >&2
    exit 2
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" "${feature_cache_args[@]}" \
    --output-dir "${output}" --quantizer "${quantizers[$i]}" \
    --codebook-size 256 --fsq-levels "${levels[$i]}" \
    --queries-per-transition 16 --token-routing spatial_aligned \
    --batch-size 32 --max-steps "${steps}" --eval-interval 5000 --skip-initial-eval \
    --save-interval 5000 --validation-episodes 5,20,25,39,49 \
    --action-aux-weight 0 --seed 42 \
    >"${output}/train.log" 2>&1 &
  pids+=("$!")
  echo "started ${names[$i]} on GPU ${gpus[$i]}: pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
