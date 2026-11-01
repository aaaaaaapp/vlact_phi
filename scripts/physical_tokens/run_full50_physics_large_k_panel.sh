#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
panel="${ROBOTWIN_PHYSICS_LARGE_K_PANEL:?Set ROBOTWIN_PHYSICS_LARGE_K_PANEL to k512, k1024, or k2048}"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_LARGE_K_GPUS:?Set four GPU IDs}"

case "${panel}" in
  k512)
    names=(ema_k512 fsq_levels4448_k512 fsq_levels33319_k513 ifsq_levels33319_k513)
    methods=(ema_vq fsq fsq ifsq)
    levels=(4,4,4,8 4,4,4,8 3,3,3,19 3,3,3,19)
    sizes=(512 512 513 513)
    ;;
  k1024)
    names=(ema_k1024 fsq_levels44444_k1024 fsq_levels3777_k1029 ifsq_levels3777_k1029)
    methods=(ema_vq fsq fsq ifsq)
    levels=(4,4,4,4,4 4,4,4,4,4 3,7,7,7 3,7,7,7)
    sizes=(1024 1024 1029 1029)
    ;;
  k2048)
    names=(ema_k2048 fsq_levels444442_k2048 fsq_levels333355_k2025 ifsq_levels333355_k2025)
    methods=(ema_vq fsq fsq ifsq)
    levels=(4,4,4,4,4,2 4,4,4,4,4,2 3,3,3,3,5,5 3,3,3,3,5,5)
    sizes=(2048 2048 2025 2025)
    ;;
  *) echo "Unknown panel: ${panel}" >&2; exit 2 ;;
esac

if [[ "${#gpus[@]}" -ne 4 || ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Need four GPUs and the merged 50-task physics cache/manifest" >&2
  exit 2
fi
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: ${output}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpus[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 12000 ]]; then
    echo "GPU ${gpus[$i]} has only ${free_mib:-unknown} MiB free; need at least 12000" >&2
    exit 2
  fi
done

frame_array="${ROBOTWIN_FULL50_FRAME_FEATURES_NPY:-}"
if [[ -n "${frame_array}" ]]; then
  "${python_bin}" -c 'import json,numpy as np,sys; m=json.load(open(sys.argv[2],encoding="utf-8")); x=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False); assert m["num_tasks"]==50 and x.shape==tuple(m["frame_feature_shape"]) and x.dtype==np.dtype(m["frame_feature_dtype"])' "${frame_array}" "${cache}.manifest.json"
else
  frame_array="$(bash scripts/physical_tokens/prepare_full50_physics_frame_array.sh)"
fi
echo "$(date -Is) ${panel} on $(hostname); validated local feature array ${frame_array}"
for gpu in "${gpus[@]}"; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 12000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free after cache preparation" >&2
    exit 2
  fi
done

pids=()
for i in "${!names[@]}"; do
  gpu="${gpus[$i]}"
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
  echo "$(date -Is) ${panel} train ${names[$i]} GPU=${gpu} pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "${panel} training failed; inspect individual train.log files" >&2
  exit 1
fi

pids=()
for i in "${!names[@]}"; do
  gpu="${gpus[$i]}"
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
  echo "$(date -Is) ${panel} eval ${names[$i]} GPU=${gpu} pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "${panel} evaluation failed; inspect individual heldout logs" >&2
  exit 1
fi
echo "$(date -Is) ${panel} complete"
