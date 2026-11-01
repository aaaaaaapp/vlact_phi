#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node003* ]]; then
  echo "Run this baseline re-evaluation on node003" >&2
  exit 2
fi
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
frame_array="/tmp/robotwin_full50_frame_features_31b49766651217e1.npy"
"${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert m["num_tasks"]==50 and x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"
root="artifacts/robotwin_full50_study/physics_tokenizers"
names=(ema_k256 fsq_levels4444)
gpus=(5 7)
pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000_schema3.json"
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing schema3 report: ${output}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpus[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 12000 ]]; then
    echo "GPU ${gpus[$i]} has insufficient free memory: ${free_mib:-unknown} MiB" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${checkpoint}" --output "${report}" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/heldout_step20000_schema3.log" 2>&1 &
  pids+=("$!")
  echo "$(date -Is) schema3 re-eval ${names[$i]} GPU=${gpus[$i]} pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "Schema3 baseline re-evaluation failed; inspect individual logs" >&2
  exit 1
fi
echo "$(date -Is) schema3 baseline re-evaluation complete"
