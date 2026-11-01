#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
frame_array="${ROBOTWIN_FULL50_FRAME_FEATURES_NPY:-}"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_EMA_TUNE_GPUS:-1,3}"
names=(ema_k256_decay095_dead100 ema_k256_commit05)
decays=(0.95 0.99)
dead_steps=(100 200)
commitments=(0.25 0.5)

if [[ "$(hostname)" != *node001* || "${#gpus[@]}" -ne 2 ]]; then
  echo "This EMA tuning panel requires node001 and two GPU IDs" >&2
  exit 2
fi
if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" || -z "${frame_array}" ]]; then
  echo "Need merged cache/manifest and ROBOTWIN_FULL50_FRAME_FEATURES_NPY path" >&2
  exit 2
fi
for name in "${names[@]}"; do
  output="${root}/${name}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty EMA tuning output: ${output}" >&2
    exit 2
  fi
done

# The levels sweep may still be producing the same node-local array. Its
# prepare script moves the validated NPY into place atomically.
for ((minute=0; minute<120; minute++)); do
  if [[ -s "${frame_array}" ]]; then break; fi
  if (( minute % 10 == 0 )); then
    echo "$(date -Is) waiting for validated node-local feature array (${minute}/120 min)"
  fi
  sleep 60
done
if [[ ! -s "${frame_array}" ]]; then
  echo "Timed out waiting for feature array: ${frame_array}" >&2
  exit 2
fi
"${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --output-dir "${output}" --quantizer ema_vq --codebook-size 256 \
    --fsq-levels 4,8,8 --ema-decay "${decays[$i]}" \
    --dead-code-steps "${dead_steps[$i]}" --commitment-weight "${commitments[$i]}" \
    --queries-per-transition 16 --token-routing spatial_aligned \
    --batch-size 32 --max-steps 20000 --eval-interval 5000 \
    --skip-initial-eval --save-interval 5000 \
    --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
    >"${output}/train.log" 2>&1 &
  pids+=("$!")
  echo "$(date -Is) train ${names[$i]} GPU=${gpus[$i]} pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "EMA tuning training failed; inspect train.log" >&2
  exit 1
fi

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000.json"
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing report: ${output}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1 &
  pids+=("$!")
  echo "$(date -Is) eval ${names[$i]} GPU=${gpus[$i]} pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
exit "${failed}"
