#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_LEVEL_SWEEP_GPUS:-4,5,6,7}"
names=(fsq_levels22222222 fsq_levels22444 fsq_levels3339 ifsq_levels3339 fsq_levels33333 ifsq_levels33333)
methods=(fsq fsq fsq ifsq fsq ifsq)
levels=(2,2,2,2,2,2,2,2 2,2,4,4,4 3,3,3,9 3,3,3,9 3,3,3,3,3 3,3,3,3,3)
sizes=(256 256 243 243 243 243)

if [[ "$(hostname)" != *node001* || "${#gpus[@]}" -ne 4 ]]; then
  echo "This panel requires node001 and four GPU IDs" >&2
  exit 2
fi
if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Merged 50-task physics cache is required" >&2
  exit 2
fi
for name in "${names[@]}"; do
  output="${root}/${name}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty sweep output: ${output}" >&2
    exit 2
  fi
done

# This SHA-named, validated NPY is only a node-local read accelerator.
frame_array="$(bash scripts/physical_tokens/prepare_full50_physics_frame_array.sh)"
echo "$(date -Is) node-local feature array: ${frame_array}"

run_panel() {
  local start="$1" end="$2" phase="$3"
  local i gpu output checkpoint report pid failed
  local -a pids=()
  for ((i=start; i<end; i++)); do
    gpu="${gpus[$((i-start))]}"
    output="${root}/${names[$i]}_steps20000"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
      --input "${cache}" --feature-array-cache "${frame_array}" \
      --output-dir "${output}" --quantizer "${methods[$i]}" \
      --codebook-size "${sizes[$i]}" --fsq-levels "${levels[$i]}" \
      --queries-per-transition 16 --token-routing spatial_aligned \
      --batch-size 32 --max-steps 20000 --eval-interval 5000 \
      --skip-initial-eval --save-interval 5000 \
      --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
      >"${output}/train.log" 2>&1 &
    pids+=("$!")
    echo "$(date -Is) ${phase} train ${names[$i]} GPU=${gpu} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${phase} training failed; inspect train.log" >&2
    exit 1
  fi

  pids=()
  for ((i=start; i<end; i++)); do
    gpu="${gpus[$((i-start))]}"
    output="${root}/${names[$i]}_steps20000"
    checkpoint="${output}/step-00020000.pt"
    report="${output}/heldout_step20000.json"
    if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
      echo "Missing checkpoint or existing report: ${output}" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
      --input "${cache}" --feature-array-cache "${frame_array}" \
      --checkpoint "${checkpoint}" --output "${report}" \
      --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
      >"${output}/heldout_step20000.log" 2>&1 &
    pids+=("$!")
    echo "$(date -Is) ${phase} eval ${names[$i]} GPU=${gpu} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${phase} evaluation failed; inspect heldout logs" >&2
    exit 1
  fi
}

run_panel 0 4 first
run_panel 4 6 second
echo "$(date -Is) physical levels sweep complete"
