#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
data_root="${ROBOTWIN_DATA_ROOT:-./shared_robotwin}"
base_vlm="${BASE_VLM:-./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
checkpoint="${PRETRAINED_CHECKPOINT:-/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt}"
output="${SPATIOTEMPORAL_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz}"
data_name="${SPATIOTEMPORAL_DATA_NAME:-click_alarmclock}"
data_names="${SPATIOTEMPORAL_DATA_NAMES:-}"

extra_args=()
if [[ "${SPATIOTEMPORAL_MAX_EPISODES:-0}" -gt 0 ]]; then
  extra_args+=(--max-episodes "${SPATIOTEMPORAL_MAX_EPISODES}")
fi
if [[ -n "${SPATIOTEMPORAL_EXCLUDE_EPISODE_IDS:-}" ]]; then
  extra_args+=(--exclude-episode-ids "${SPATIOTEMPORAL_EXCLUDE_EPISODE_IDS}")
fi
if [[ -n "${data_names}" ]]; then
  extra_args+=(--data-names "${data_names}")
else
  extra_args+=(--data-name "${data_name}")
fi

CUDA_VISIBLE_DEVICES="${SPATIOTEMPORAL_EXPORT_GPU:-0}" "${python_bin}" \
  scripts/physical_tokens/export_robotwin_spatiotemporal_features.py \
  --data-root "${data_root}" \
  --robot-type robotwin_wrap32 \
  --camera video.cam_high \
  --offsets 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32 \
  --spatial-grid-size 4 \
  --base-vlm "${base_vlm}" \
  --pretrained-checkpoint "${checkpoint}" \
  --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
  --output "${output}" \
  --batch-size "${SPATIOTEMPORAL_EXPORT_BATCH_SIZE:-32}" \
  --device cuda:0 \
  "${extra_args[@]}"
