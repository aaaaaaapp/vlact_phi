#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-${repo_root}/shared_robotwin}"
base_vlm="${BASE_VLM:-${repo_root}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
checkpoint="${PRETRAINED_CHECKPOINT:-/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt}"
part_dir="artifacts/future_feature_caches/d15_8tasks_current1x1_parts"
log_dir="results/d15_visual_export_logs"
output="artifacts/future_feature_caches/robotwin_d15_8tasks_all_episodes_current1x1.npz"
action_cache="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
tasks=(
  click_alarmclock beat_block_hammer open_microwave shake_bottle
  move_playingcard_away pick_dual_bottles handover_block place_object_basket
)

mkdir -p "${part_dir}" "${log_dir}"
pids=()
parts=()
for gpu in "${!tasks[@]}"; do
  task="${tasks[$gpu]}"
  part="${part_dir}/${task}_all_episodes_current1x1.npz"
  parts+=("${part}")
  if [[ -s "${part}" && -s "${part}.manifest.json" ]]; then
    echo "D15 visual part already complete: ${part}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/export_robotwin_spatiotemporal_features.py \
    --data-root "${data_root}" --data-name "${task}" --robot-type robotwin_wrap32 \
    --camera video.cam_high --offsets 0 --spatial-grid-size 1 \
    --base-vlm "${base_vlm}" --pretrained-checkpoint "${checkpoint}" \
    --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
    --output "${part}" --batch-size "${D15_VISUAL_BATCH_SIZE:-32}" --device cuda:0 \
    >"${log_dir}/${task}.log" 2>&1 &
  pids+=("$!")
  echo "started ${task} on GPU ${gpu}, pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more D15 visual exports failed; inspect ${log_dir}" >&2
  exit 1
fi
if [[ -e "${output}" || -e "${output}.manifest.json" ]]; then
  echo "D15 refused to overwrite merged visual cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/merge_multitask_visual_caches.py \
  --inputs "${parts[@]}" --task-names "${tasks[@]}" \
  --action-cache "${action_cache}" --output "${output}"
