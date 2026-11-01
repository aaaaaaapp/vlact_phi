#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node004* ]]; then
  echo "This EMA capacity scan requires node004" >&2
  exit 2
fi
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
frame_array="/tmp/robotwin_full50_frame_features_31b49766651217e1.npy"
root="artifacts/robotwin_full50_study/physics_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_EMA_UTIL_GPUS:-1,2}"
IFS=',' read -r -a sizes <<< "${ROBOTWIN_EMA_UTIL_SIZES:-4096,8192}"
if [[ "${#gpus[@]}" -lt 1 || "${#gpus[@]}" -ne "${#sizes[@]}" ]]; then
  echo "Need one GPU per capacity" >&2
  exit 2
fi

"${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert m["num_tasks"]==50 and x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"
for i in "${!sizes[@]}"; do
  output="${root}/ema_k${sizes[$i]}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: ${output}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpus[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 32000 ]]; then
    echo "GPU ${gpus[$i]} has insufficient free memory: ${free_mib:-unknown} MiB" >&2
    exit 2
  fi
done

run_one() {
  local gpu="$1" size="$2" output checkpoint report
  output="${root}/ema_k${size}_steps20000"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000.json"
  mkdir -p "${output}"
  echo "$(date -Is) training EMA K${size} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/train_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --output-dir "${output}" --quantizer ema_vq --codebook-size "${size}" \
    --fsq-levels 4,8,8 --queries-per-transition 16 --token-routing spatial_aligned \
    --batch-size 32 --max-steps 20000 --eval-interval 20000 \
    --skip-initial-eval --save-interval 5000 \
    --validation-episodes 5,20,25,39,49 --action-aux-weight 0 --seed 42 \
    >"${output}/train.log" 2>&1
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing report: ${output}" >&2
    return 2
  fi
  echo "$(date -Is) evaluating EMA K${size} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1
  "${python_bin}" -c 'import json,sys; x=json.load(open(sys.argv[1],encoding="utf-8")); m=x["metrics"]; k=x["model_config"]["codebook_size"]; print("EMA K%d: %d/%d active, ratio=%.6f, RMSE=%.6f" % (k,m["active_codes"],k,m["active_ratio"],m["rmse"]))' "${report}"
}

pids=()
for i in "${!sizes[@]}"; do
  run_one "${gpus[$i]}" "${sizes[$i]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "EMA utilization scan failed; inspect individual logs" >&2
  exit 1
fi
echo "$(date -Is) EMA capacities ${sizes[*]} utilization scan complete"
