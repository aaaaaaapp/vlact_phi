#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node003* ]]; then
  echo "This structure analysis is configured for node003" >&2
  exit 2
fi

side="${ROBOTWIN_ROUTING_STRUCTURE_SIDE:?Set to aligned or global}"
case "${side}" in
  aligned) names=(ema_k1024 ema_k2048) ;;
  global) names=(ema_global_k1024 ema_global_k2048) ;;
  *) echo "Unknown routing side: ${side}" >&2; exit 2 ;;
esac
IFS=',' read -r -a gpus <<< "${ROBOTWIN_ROUTING_STRUCTURE_GPUS:-6,7}"
if [[ "${#gpus[@]}" -ne 2 ]]; then
  echo "Need two GPU IDs" >&2
  exit 2
fi

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
frame_array="/tmp/robotwin_full50_frame_features_31b49766651217e1.npy"
root="artifacts/robotwin_full50_study/physics_tokenizers"

for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  if [[ ! -s "${output}/heldout_step20000.json" || ! -s "${output}/step-00020000.pt" || -e "${output}/structure_heldout_step20000.json" ]]; then
    echo "Missing final model/report or existing structure report: ${output}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpus[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 16000 ]]; then
    echo "GPU ${gpus[$i]} has only ${free_mib:-unknown} MiB free" >&2
    exit 2
  fi
done

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  echo "$(date -Is) structure ${names[$i]} on GPU${gpus[$i]}"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/analyze_spatiotemporal_token_structure.py \
    --input "${cache}" --feature-array-cache "${frame_array}" \
    --checkpoint "${output}/step-00020000.pt" \
    --output "${output}/structure_heldout_step20000.json" \
    --validation-episodes 5,20,25,39,49 --batch-size 32 --device cuda:0 \
    >"${output}/structure_heldout_step20000.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "Structure analysis failed; inspect individual logs" >&2
  exit 1
fi
echo "$(date -Is) routing ${side} structure analysis complete"
