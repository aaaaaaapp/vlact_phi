#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz"
action_cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_actions17x14.npz"
d2_output="artifacts/physical_language_tokenizers/spatiotemporal_d2_8tasks_spatial_aligned_ema_256x128_steps1000"
output="${D3_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d3_8tasks_action_aligned_ema_256x128_steps1000}"
steps="${D3_MAX_STEPS:-1000}"
if [[ "${steps}" != "1000" ]]; then
  echo "Preregistered D3 run requires exactly 1000 steps" >&2
  exit 2
fi
checkpoint="${output}/step-00001000.pt"
for required in "${cache}" "${action_cache}" "${d2_output}/evaluation_per_task.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required D3 input is missing: ${required}" >&2
    exit 2
  fi
done
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D3 checkpoint: ${checkpoint}"
elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "D3 refused to overwrite partial output: ${output}" >&2
  exit 2
else
  SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${action_cache}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=300 \
  SPATIOTEMPORAL_MAX_STEPS=1000 \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
fi

"${python_bin}" scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" --action-target-cache "${action_cache}" \
  --checkpoint "${checkpoint}" --output "${output}/evaluation_per_task.json" \
  --validation-episodes-per-task 2 --device "${D3_EVAL_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/analyze_spatiotemporal_token_structure.py \
  --input "${cache}" --checkpoint "${checkpoint}" \
  --output "${output}/token_structure_per_task.json" \
  --validation-episodes-per-task 2 --device "${D3_EVAL_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/compare_spatiotemporal_tokenizers.py \
  --baseline "${d2_output}/evaluation_per_task.json" \
  --candidate "${output}/evaluation_per_task.json" \
  --allowed-model-config-differences action_aux_dim,action_aux_weight,action_gripper_weight,action_velocity_weight \
  --comparison-name paired_same_cache_d2_vs_d3_action_aux \
  --output "${output}/comparison_d2_vs_d3.json"
"${python_bin}" scripts/physical_tokens/require_d3_action_aligned_gate.py \
  --comparison "${output}/comparison_d2_vs_d3.json" \
  --structure "${output}/token_structure_per_task.json" \
  --output "${output}/joint_gate_pending_action_relevance.json"
