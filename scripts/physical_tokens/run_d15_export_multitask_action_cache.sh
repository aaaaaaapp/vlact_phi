#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-${repo_root}/shared_robotwin}"
part_dir="artifacts/action_caches/d15_8tasks_parts"
output="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
tasks=(
  click_alarmclock beat_block_hammer open_microwave shake_bottle
  move_playingcard_away pick_dual_bottles handover_block place_object_basket
)

mkdir -p "${part_dir}"
parts=()
for task in "${tasks[@]}"; do
  part="${part_dir}/${task}_all_episodes_wrap32.npz"
  parts+=("${part}")
  if [[ ! -s "${part}" ]]; then
    "${python_bin}" -u scripts/physical_tokens/export_robotwin_action_cache.py \
      --data-root "${data_root}" --data-name "${task}" --robot-type robotwin_wrap32 \
      --output "${part}" --stride 1
  fi
done

if [[ -s "${output}" ]]; then
  echo "D15 refused to overwrite merged action cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/merge_multitask_action_caches.py \
  --inputs "${parts[@]}" --task-names "${tasks[@]}" --output "${output}"
