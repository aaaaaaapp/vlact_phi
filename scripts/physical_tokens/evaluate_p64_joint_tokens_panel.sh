#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
config="${P64_EVAL_CONFIG:-examples/Robotwin/train_files/qwenpi_v4_p64_m32_eval.yaml}"
checkpoint="${P64_EVAL_CHECKPOINT:?set P64_EVAL_CHECKPOINT to a P64 policy checkpoint}"
cache="${P64_EVAL_CACHE:-artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz}"
samples="${P64_EVAL_SAMPLES:-64}"
output_root="${P64_EVAL_OUTPUT:-artifacts/evaluations/p64_joint_token_panel}"
mkdir -p "${output_root}"

tasks=(
  beat_block_hammer click_alarmclock handover_block move_playingcard_away
  open_microwave pick_dual_bottles place_object_basket shake_bottle
)
pids=()
for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" \
    scripts/physical_tokens/evaluate_joint_token_predictions.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --token-cache "${cache}" \
    --task-name "${task}" \
    --num-samples "${samples}" \
    --device cuda:0 \
    --output "${output_root}/${task}.json" \
    > "${output_root}/${task}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
