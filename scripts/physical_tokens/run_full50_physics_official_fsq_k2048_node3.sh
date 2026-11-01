#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node003* ]]; then
  echo "This FSQ K2048 panel is configured for node003" >&2
  exit 2
fi

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
frame_array="/tmp/robotwin_full50_frame_features_31b49766651217e1.npy"
root="artifacts/robotwin_full50_study/physics_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_OFFICIAL_FSQ_GPUS:-4,6}"
names=(fsq_official_levels84444_k2048 fsq_symmetric_levels84444_k2048)
methods=(fsq_official fsq)

if [[ "${#gpus[@]}" -ne 2 || ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Need two GPU IDs and the merged 50-task cache/manifest" >&2
  exit 2
fi
"${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert m["num_tasks"]==50 and x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"

for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: ${output}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpus[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 24000 ]]; then
    echo "GPU ${gpus[$i]} has only ${free_mib:-unknown} MiB free; need at least 24000" >&2
    exit 2
  fi
done

run_one() {
  local i="$1" gpu output checkpoint report
  gpu="${gpus[$i]}"
  output="${root}/${names[$i]}_steps20000"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000.json"
  mkdir -p "${output}"
  echo "$(date -Is) training ${names[$i]} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --output-dir "${output}" --quantizer "${methods[$i]}" \
    --codebook-size 2048 --fsq-levels 8,4,4,4,4 \
    --queries-per-transition 16 --token-routing spatial_aligned \
    --batch-size 32 --max-steps 20000 --eval-interval 5000 \
    --skip-initial-eval --save-interval 5000 \
    --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
    >"${output}/train.log" 2>&1
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing report: ${output}" >&2
    return 2
  fi
  echo "$(date -Is) evaluating ${names[$i]} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1
  echo "$(date -Is) complete ${names[$i]}"
}

pids=()
for i in "${!names[@]}"; do
  run_one "${i}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "Official FSQ K2048 panel failed; inspect individual logs" >&2
  exit 1
fi
echo "$(date -Is) official FSQ K2048 panel complete"
