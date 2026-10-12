#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-${repo_root}/shared_robotwin}"
base_vlm="${BASE_VLM:-${repo_root}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
checkpoint="${NODE_LOCAL_PRETRAINED_CHECKPOINT:-/tmp/vlact_qwen3_pretrain_steps100000.pt}"
part_dir="artifacts/future_feature_caches/d28_open_microwave_shards"
log_dir="artifacts/logs/d28_open_microwave_shards"
output="artifacts/future_feature_caches/d28_8tasks_all_episodes_spatial4x4_parts/open_microwave_all_episodes_spatial4x4_17frames.npz"
mkdir -p "${part_dir}" "${log_dir}"

episode_sets=(
  "0,1,2,3,4,5,6,7,8"
  "9,10,11,12,13,14,15,16,17"
  "18,19,20,21,22,23,24,25"
  "26,27,28,29,30,31,32,33"
  "34,35,36,37,38,39,40,41"
  "42,43,44,45,46,47,48,49"
)
gpus=(2 3 4 5 6 7)
pids=()
parts=()
for index in "${!episode_sets[@]}"; do
  part="${part_dir}/open_microwave_shard${index}.npz"
  parts+=("${part}")
  if [[ -s "${part}" && -s "${part}.manifest.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "${python_bin}" -u \
    scripts/physical_tokens/export_robotwin_spatiotemporal_features.py \
    --data-root "${data_root}" --data-name open_microwave --robot-type robotwin_wrap32 \
    --camera video.cam_high --offsets 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 \
    --spatial-grid-size 4 --base-vlm "${base_vlm}" \
    --pretrained-checkpoint "${checkpoint}" \
    --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
    --include-episode-ids "${episode_sets[$index]}" \
    --output "${part}" --batch-size 32 --device cuda:0 \
    >"${log_dir}/shard${index}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
if [[ ! -s "${output}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/merge_single_task_spatiotemporal_shards.py \
    --inputs "${parts[@]}" --output "${output}"
fi
