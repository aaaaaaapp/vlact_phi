#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin}"
max_jobs="${ROBOTWIN_ACTION_EXPORT_JOBS:-4}"
old_part_dir="artifacts/action_caches/d15_8tasks_parts"
study_dir="artifacts/robotwin_full50_study"
part_dir="${study_dir}/action_parts"
log_dir="${study_dir}/action_logs"
output="${study_dir}/action_all50_wrap32.npz"

if ! [[ "${max_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROBOTWIN_ACTION_EXPORT_JOBS must be a positive integer" >&2
  exit 2
fi
mapfile -t tasks < docs/robotwin_50task_list.txt
if [[ "${#tasks[@]}" -ne 50 ]]; then
  echo "Expected exactly 50 RoboTwin tasks, found ${#tasks[@]}" >&2
  exit 2
fi
mkdir -p "${part_dir}" "${log_dir}"

parts=()
pids=()
failed=0
for task in "${tasks[@]}"; do
  if [[ ! -s "${data_root}/${task}/meta/info.json" ]]; then
    echo "Missing source metadata for ${task}" >&2
    exit 2
  fi
  old_part="${old_part_dir}/${task}_all_episodes_wrap32.npz"
  new_part="${part_dir}/${task}_all_episodes_wrap32.npz"
  if [[ -s "${old_part}" && -s "${old_part}.manifest.json" ]]; then
    parts+=("${old_part}")
    continue
  fi
  parts+=("${new_part}")
  if [[ -s "${new_part}" && -s "${new_part}.manifest.json" ]]; then
    continue
  fi
  if [[ -e "${new_part}" || -e "${new_part}.manifest.json" ]]; then
    echo "Refusing incomplete output for ${task}: ${new_part}" >&2
    exit 2
  fi
  "${python_bin}" -u scripts/physical_tokens/export_robotwin_action_cache.py \
    --data-root "${data_root}" --data-name "${task}" \
    --robot-type robotwin_wrap32 --output "${new_part}" --stride 1 \
    >"${log_dir}/${task}.log" 2>&1 &
  pids+=("$!")
  echo "started action export: ${task}, pid=${pids[-1]}"
  if [[ "${#pids[@]}" -ge "${max_jobs}" ]]; then
    if ! wait "${pids[0]}"; then
      failed=1
    fi
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "Action export failed; inspect ${log_dir}" >&2
  exit 1
fi

if [[ -s "${output}" && -s "${output}.manifest.json" ]]; then
  echo "Full 50-task action cache already exists: ${output}"
  exit 0
fi
if [[ -e "${output}" || -e "${output}.manifest.json" ]]; then
  echo "Refusing incomplete merged cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/merge_multitask_action_caches.py \
  --inputs "${parts[@]}" --task-names "${tasks[@]}" --output "${output}"
