#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
cache="artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz"
root="artifacts/robotwin_full50_study/physics_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_PHYSICS_LARGE_K_GPUS:-4,5,6,7}"
names=(ema_k512 fsq_levels4448_k512 fsq_levels33319_k513 ifsq_levels33319_k513
       ema_k1024 fsq_levels44444_k1024 fsq_levels3777_k1029 ifsq_levels3777_k1029)
methods=(ema_vq fsq fsq ifsq ema_vq fsq fsq ifsq)
levels=(4,4,4,8 4,4,4,8 3,3,3,19 3,3,3,19
        4,4,4,4,4 4,4,4,4,4 3,7,7,7 3,7,7,7)
sizes=(512 512 513 513 1024 1024 1029 1029)

if [[ "$(hostname)" != *node001* || "${#gpus[@]}" -ne 4 ]]; then
  echo "This four-GPU panel requires node001" >&2
  exit 2
fi
if [[ ! -s "${cache}" || ! -s "${cache}.manifest.json" ]]; then
  echo "Merged 50-task cache and manifest are required" >&2
  exit 2
fi
for name in "${names[@]}"; do
  output="${root}/${name}_steps20000"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: ${output}" >&2
    exit 2
  fi
done

# The existing node1 level sweep owns GPUs 4-7. Never overlap it.
while pgrep -f '[r]un_full50_physics_level_sweep_node1.sh' >/dev/null; do
  echo "$(date -Is) waiting for existing node1 level sweep"
  sleep 60
done
frame_array="$(bash scripts/physical_tokens/prepare_full50_physics_frame_array.sh)"
echo "$(date -Is) validated feature array: ${frame_array}"

run_panel() {
  local start="$1" end="$2" label="$3"
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
    echo "$(date -Is) ${label} train ${names[$i]} GPU=${gpu} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if ((failed)); then
    echo "${label} training failed; inspect individual train.log files" >&2
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
    echo "$(date -Is) ${label} eval ${names[$i]} GPU=${gpu} pid=$!"
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if ((failed)); then
    echo "${label} evaluation failed; inspect individual heldout logs" >&2
    exit 1
  fi
}

run_panel 0 4 k512
run_panel 4 8 k1024
echo "$(date -Is) physical large-codebook panel complete"
