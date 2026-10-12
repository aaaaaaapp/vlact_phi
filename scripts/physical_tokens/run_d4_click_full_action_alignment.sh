#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
cache="artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz"
action_cache="artifacts/future_feature_caches/click_alarmclock_actions17x14.npz"
data_root="/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin/click_alarmclock"
control="${D4_CONTROL_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d4_click_full_control_ema_256x128_steps1000}"
candidate="${D4_ACTION_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d4_click_full_action_aligned_w1_ema_256x128_steps1000}"
steps="${D4_MAX_STEPS:-1000}"
if [[ "${steps}" != "1000" ]]; then
  echo "Preregistered D4 run requires exactly 1000 steps" >&2
  exit 2
fi

if [[ ! -s "${action_cache}" ]]; then
  "${python_bin}" scripts/physical_tokens/export_robotwin_action_target_sidecar.py \
    --feature-cache "${cache}" --data-root "${data_root}" --output "${action_cache}"
fi

run_tokenizer() {
  local output="$1"
  shift
  local checkpoint="${output}/step-00001000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "Reusing completed D4 checkpoint: ${checkpoint}"
  elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D4 refused to overwrite partial output: ${output}" >&2
    exit 2
  else
    SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
    SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
    SPATIOTEMPORAL_VALIDATION_EPISODES="5,20,25,39,49" \
    SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=0 \
    SPATIOTEMPORAL_MAX_STEPS=1000 \
    SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
    "$@" bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
  fi
}

run_tokenizer "${control}" env
run_tokenizer "${candidate}" env \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${action_cache}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=1.0 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1

"${python_bin}" scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" --checkpoint "${control}/step-00001000.pt" \
  --output "${control}/evaluation.json" --validation-episodes 5,20,25,39,49 \
  --device "${D4_EVAL_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" --action-target-cache "${action_cache}" \
  --checkpoint "${candidate}/step-00001000.pt" \
  --output "${candidate}/evaluation.json" --validation-episodes 5,20,25,39,49 \
  --device "${D4_EVAL_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/compare_spatiotemporal_tokenizers.py \
  --baseline "${control}/evaluation.json" --candidate "${candidate}/evaluation.json" \
  --allowed-model-config-differences action_aux_dim,action_aux_weight,action_gripper_weight,action_velocity_weight \
  --comparison-name paired_same_cache_d4_control_vs_action_aux \
  --output "${candidate}/comparison_control_vs_action_aux.json"
