#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node001* ]]; then
  echo "This 128-token panel is configured for node001" >&2
  exit 2
fi

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
frame_array="/tmp/robotwin_full50_frame_features_31b49766651217e1.npy"
root="artifacts/robotwin_full50_study/physics_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_128TOKEN_GPUS:-4,5}"
sizes=(1024 2048)
levels=(4,4,4,4,4 4,4,4,4,4,2)

if [[ "${#gpus[@]}" -ne 2 || ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Need two GPU IDs and the merged 50-task cache/manifest" >&2
  exit 2
fi
"${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert m["num_tasks"]==50 and x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"

for i in "${!sizes[@]}"; do
  output="${root}/ema_t128_k${sizes[$i]}_steps20000"
  aligned="${root}/ema_k${sizes[$i]}_steps20000/heldout_step20000.json"
  if [[ ! -s "${aligned}" ]]; then
    echo "Missing completed 256-token reference: ${aligned}" >&2
    exit 2
  fi
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
  local i="$1" gpu size output checkpoint report
  gpu="${gpus[$i]}"
  size="${sizes[$i]}"
  output="${root}/ema_t128_k${size}_steps20000"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000.json"
  mkdir -p "${output}"
  echo "$(date -Is) training 128-token spatial-aligned EMA K${size} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --output-dir "${output}" --quantizer ema_vq --codebook-size "${size}" \
    --fsq-levels "${levels[$i]}" --queries-per-transition 8 \
    --token-routing spatial_aligned --batch-size 32 --max-steps 20000 \
    --eval-interval 5000 --skip-initial-eval --save-interval 5000 \
    --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
    >"${output}/train.log" 2>&1
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing report: ${output}" >&2
    return 2
  fi
  echo "$(date -Is) evaluating 128-token EMA K${size} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1
  echo "$(date -Is) complete 128-token EMA K${size}"
}

pids=()
for i in "${!sizes[@]}"; do
  run_one "${i}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "128-token panel failed; inspect individual logs" >&2
  exit 1
fi
echo "$(date -Is) full50 128-token panel complete"
