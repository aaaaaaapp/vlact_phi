#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
steps="${ROBOTWIN_PHYSICS_TRAIN_STEPS:-20000}"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_IFSQ_GPUS:-0,1,2,3,4,5}"
names=(ifsq_levels3517 fsq_levels3517 ifsq_levels577 fsq_levels577 ifsq_levels5511 fsq_levels5511)
methods=(ifsq fsq ifsq fsq ifsq fsq)
levels=(3,5,17 3,5,17 5,7,7 5,7,7 5,5,11 5,5,11)
sizes=(255 255 245 245 275 275)

if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" || "${#gpus[@]}" -ne 6 ]]; then
  echo "Expected merged 50-task physics cache, manifest, and six GPU IDs" >&2
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
for name in "${names[@]}"; do
  output="${root}/${name}_steps${steps}"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output; review before restarting: ${output}" >&2
    exit 2
  fi
done

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps${steps}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" "${feature_cache_args[@]}" --output-dir "${output}" \
    --quantizer "${methods[$i]}" --codebook-size "${sizes[$i]}" \
    --fsq-levels "${levels[$i]}" --queries-per-transition 16 \
    --token-routing spatial_aligned --batch-size 32 --max-steps "${steps}" \
    --eval-interval 5000 --skip-initial-eval --save-interval 5000 \
    --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
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
