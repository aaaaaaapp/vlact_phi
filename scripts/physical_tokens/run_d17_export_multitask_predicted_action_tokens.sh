#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
config="examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml"
checkpoint="results/Checkpoints/qwen_action_token_probe_d16_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt"
oracle="artifacts/token_caches/robotwin_d16_8tasks_oracle_shared_ema4x128.npz"
part_dir="artifacts/token_caches/d17_8tasks_predicted_parts"
log_dir="results/d17_predicted_token_export_logs"
output="artifacts/token_caches/robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz"
tasks=(
  click_alarmclock beat_block_hammer open_microwave shake_bottle
  move_playingcard_away pick_dual_bottles handover_block place_object_basket
)

mkdir -p "${part_dir}" "${log_dir}"
pids=()
parts=()
for gpu in "${!tasks[@]}"; do
  task="${tasks[$gpu]}"
  part="${part_dir}/${task}_predicted.npz"
  parts+=("${part}")
  if [[ -s "${part}" && -s "${part}.manifest.json" ]]; then
    echo "D17 predicted part already complete: ${part}"
    continue
  fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  D17_EXPORT_CPU_THREADS=4 CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/export_multitask_predicted_action_tokens.py \
    --config "${config}" --checkpoint "${checkpoint}" --source-token-cache "${oracle}" \
    --data-name "${task}" --output "${part}" --batch-size "${D17_EXPORT_BATCH_SIZE:-16}" \
    --device cuda:0 --verify-first-batch >"${log_dir}/${task}.log" 2>&1 &
  pids+=("$!")
  echo "started ${task} predicted-token export on GPU ${gpu}, pid=$!"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more D17 predicted-token exports failed" >&2
  exit 1
fi
if [[ -e "${output}" || -e "${output}.manifest.json" ]]; then
  echo "D17 refused to overwrite merged predicted cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/merge_multitask_predicted_token_caches.py \
  --inputs "${parts[@]}" --task-names "${tasks[@]}" \
  --oracle-cache "${oracle}" --output "${output}"
