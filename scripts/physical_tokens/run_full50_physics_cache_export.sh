#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin}"
base_vlm="${BASE_VLM:-${repo_root}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
checkpoint="${PRETRAINED_CHECKPOINT:-/tmp/robotwin_qwen3_pretrain_steps100000.pt}"
checkpoint_sha256="5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d"
batch_size="${ROBOTWIN_PHYSICS_EXPORT_BATCH_SIZE:-8}"
IFS=',' read -r -a gpu_ids <<< "${ROBOTWIN_PHYSICS_GPUS:-1,3,4}"
IFS=',' read -r -a skip_tasks <<< "${ROBOTWIN_PHYSICS_SKIP_TASKS:-}"
shard_spec="${ROBOTWIN_PHYSICS_EXPORT_SHARD:-0/1}"
merge_only="${ROBOTWIN_PHYSICS_MERGE_ONLY:-0}"
old_part_dir="artifacts/future_feature_caches/d28_8tasks_all_episodes_spatial4x4_parts"
study_dir="artifacts/robotwin_full50_study"
part_dir="${study_dir}/physics_parts"
log_dir="${study_dir}/physics_logs"
output="${study_dir}/physics_all50_spatial4x4_17frames.npz"
action_cache="${study_dir}/action_all50_wrap32.npz"

mapfile -t tasks < docs/robotwin_50task_list.txt
if [[ "${#tasks[@]}" -ne 50 || "${#gpu_ids[@]}" -lt 1 ]]; then
  echo "Expected exactly 50 tasks and at least one GPU" >&2
  exit 2
fi
if ! [[ "${shard_spec}" =~ ^([0-9]+)/([1-9][0-9]*)$ ]]; then
  echo "ROBOTWIN_PHYSICS_EXPORT_SHARD must be index/count" >&2
  exit 2
fi
shard_index="${BASH_REMATCH[1]}"
shard_count="${BASH_REMATCH[2]}"
if [[ "${shard_index}" -ge "${shard_count}" ]]; then
  echo "Invalid physics export shard: ${shard_spec}" >&2
  exit 2
fi
if [[ "${merge_only}" != 1 && ( ! -s "${checkpoint}" || ! -d "${base_vlm}" ) ]]; then
  echo "Missing pretrained checkpoint or base VLM" >&2
  exit 2
fi
mkdir -p "${part_dir}" "${log_dir}"

parts=()
gpu_pids=()
failed=0
next_gpu=0
for task_index in "${!tasks[@]}"; do
  task="${tasks[$task_index]}"
  old_part="${old_part_dir}/${task}_all_episodes_spatial4x4_17frames.npz"
  new_part="${part_dir}/${task}_all_episodes_spatial4x4_17frames.npz"
  if [[ -s "${old_part}" && -s "${old_part}.manifest.json" ]]; then
    parts+=("${old_part}")
    continue
  fi
  parts+=("${new_part}")
  if [[ "${merge_only}" == 1 || $((task_index % shard_count)) -ne "${shard_index}" ]]; then
    continue
  fi
  skip_this_task=0
  for skipped in "${skip_tasks[@]}"; do
    if [[ "${task}" == "${skipped}" ]]; then
      skip_this_task=1
      break
    fi
  done
  if [[ "${skip_this_task}" -eq 1 ]]; then
    continue
  fi
  if [[ -s "${new_part}" && -s "${new_part}.manifest.json" ]]; then
    continue
  fi
  if [[ -e "${new_part}" || -e "${new_part}.manifest.json" ]]; then
    echo "Refusing incomplete output for ${task}: ${new_part}" >&2
    exit 2
  fi
  gpu="${gpu_ids[$next_gpu]}"
  if [[ -n "${gpu_pids[$next_gpu]:-}" ]]; then
    if ! wait "${gpu_pids[$next_gpu]}"; then
      failed=1
    fi
    if [[ "${failed}" -ne 0 ]]; then
      break
    fi
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/export_robotwin_spatiotemporal_features.py \
    --data-root "${data_root}" --data-name "${task}" \
    --robot-type robotwin_wrap32 --camera video.cam_high \
    --offsets 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 \
    --spatial-grid-size 4 --base-vlm "${base_vlm}" \
    --pretrained-checkpoint "${checkpoint}" \
    --pretrained-checkpoint-sha256 "${checkpoint_sha256}" \
    --output "${new_part}" --batch-size "${batch_size}" --device cuda:0 \
    >"${log_dir}/${task}.log" 2>&1 &
  gpu_pids[$next_gpu]="$!"
  echo "started physics export: ${task}, GPU ${gpu}, pid=${gpu_pids[$next_gpu]}"
  next_gpu=$(( (next_gpu + 1) % ${#gpu_ids[@]} ))
done
for pid in "${gpu_pids[@]}"; do
  if [[ -n "${pid}" ]] && ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Physics export failed; inspect ${log_dir}" >&2
  exit 1
fi

if [[ "${shard_count}" -gt 1 && "${merge_only}" != 1 ]]; then
  echo "Physics export shard ${shard_spec} complete; merge after every shard completes"
  exit 0
fi
for part in "${parts[@]}"; do
  if [[ ! -s "${part}" || ! -s "${part}.manifest.json" ]]; then
    echo "Missing physics part for merge: ${part}" >&2
    exit 2
  fi
done

if [[ ! -s "${action_cache}" || ! -s "${action_cache}.manifest.json" ]]; then
  echo "Full 50-task action cache is not ready for alignment validation: ${action_cache}" >&2
  exit 2
fi
if [[ -s "${output}" && -s "${output}.manifest.json" ]]; then
  echo "Full 50-task physics cache already exists: ${output}"
  exit 0
fi
if [[ -e "${output}" || -e "${output}.manifest.json" ]]; then
  echo "Refusing incomplete merged cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/merge_multitask_visual_caches.py \
  --inputs "${parts[@]}" --task-names "${tasks[@]}" \
  --action-cache "${action_cache}" --output "${output}"
