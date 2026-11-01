#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
steps="${ROBOTWIN_PHYSICS_TRAIN_STEPS:-20000}"
gpu="${ROBOTWIN_PHYSICS_EVAL_GPU:-0}"
names=(
  ema_k256 fsq_levels488 fsq_levels4444
  ifsq_levels3517 fsq_levels3517
  ifsq_levels577 fsq_levels577
  ifsq_levels5511 fsq_levels5511
)

if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Merged 50-task physics cache and manifest are required: ${cache}" >&2
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

# Require every checkpoint before starting, so a partial comparison is never
# mistaken for the complete nine-run panel.
for name in "${names[@]}"; do
  checkpoint="${root}/${name}_steps${steps}/step-$(printf '%08d' "${steps}").pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing final checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done

for name in "${names[@]}"; do
  run_dir="${root}/${name}_steps${steps}"
  checkpoint="${run_dir}/step-$(printf '%08d' "${steps}").pt"
  report="${run_dir}/heldout_step${steps}.json"
  if [[ -s "${report}" ]]; then
    echo "Already evaluated: ${report}"
    continue
  fi
  if [[ -e "${report}" ]]; then
    echo "Refusing incomplete report: ${report}" >&2
    exit 2
  fi
  echo "Evaluating ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" "${feature_cache_args[@]}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 \
    --batch-size 32 --device cuda:0 \
    >"${run_dir}/heldout_step${steps}.log" 2>&1
done
