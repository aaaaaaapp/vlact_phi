#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
frame_array="${ROBOTWIN_FULL50_FRAME_FEATURES_NPY:-}"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_REPEAT_GPUS:-1,2,3,4}"
names=(ema_k256 fsq_levels4444 fsq_levels3517 ifsq_levels3517)
methods=(ema_vq fsq fsq ifsq)
levels=(4,8,8 4,4,4,4 3,5,17 3,5,17)
sizes=(256 256 255 255)

if [[ "$(hostname)" != *node002* ]]; then
  echo "Run this study on node002, which has the local read-only feature array" >&2
  exit 2
fi
if [[ "${#gpus[@]}" -ne 4 || ! -s "${cache}" || ! -s "${cache}.manifest.json" || ! -s "${frame_array}" ]]; then
  echo "Need four GPU IDs, the merged cache/manifest, and ROBOTWIN_FULL50_FRAME_FEATURES_NPY" >&2
  exit 2
fi
for seed in 43 44; do
  for name in "${names[@]}"; do
    output="${root}/${name}_seed${seed}_steps20000"
    if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
      echo "Refusing nonempty repeat output: ${output}" >&2
      exit 2
    fi
  done
done

for seed in 43 44; do
  pids=()
  for i in "${!names[@]}"; do
    output="${root}/${names[$i]}_seed${seed}_steps20000"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
      scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
      --input "${cache}" --feature-array-cache "${frame_array}" \
      --output-dir "${output}" --quantizer "${methods[$i]}" \
      --codebook-size "${sizes[$i]}" --fsq-levels "${levels[$i]}" \
      --queries-per-transition 16 --token-routing spatial_aligned \
      --batch-size 32 --max-steps 20000 --eval-interval 5000 \
      --skip-initial-eval --save-interval 5000 \
      --validation-episodes 5,20,25,39,49 --action-aux-weight 0 \
      --seed "${seed}" \
      >"${output}/train.log" 2>&1 &
    pids+=("$!")
    echo "$(date -Is) started ${names[$i]} seed=${seed} GPU=${gpus[$i]} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "Training failed for seed=${seed}; inspect individual train.log files" >&2
    exit 1
  fi

  pids=()
  for i in "${!names[@]}"; do
    output="${root}/${names[$i]}_seed${seed}_steps20000"
    checkpoint="${output}/step-00020000.pt"
    report="${output}/heldout_step20000.json"
    if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
      echo "Missing final checkpoint or existing report: ${output}" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
      scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
      --input "${cache}" --feature-array-cache "${frame_array}" \
      --checkpoint "${checkpoint}" --output "${report}" \
      --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
      >"${output}/heldout_step20000.log" 2>&1 &
    pids+=("$!")
    echo "$(date -Is) evaluating ${names[$i]} seed=${seed} GPU=${gpus[$i]} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "Evaluation failed for seed=${seed}; inspect individual logs" >&2
    exit 1
  fi
done
echo "$(date -Is) physical seed-repeat panel complete"
