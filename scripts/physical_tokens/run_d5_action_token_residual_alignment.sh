#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
cache="artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz"
action_targets="artifacts/future_feature_caches/click_alarmclock_actions17x14.npz"
action_tokens="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
action_tokenizer="artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt"
residual_cache="artifacts/future_feature_caches/click_alarmclock_pred_action4_residuals17x14.npz"
control="artifacts/physical_language_tokenizers/spatiotemporal_d4_click_full_control_ema_256x128_steps1000"
output="${D5_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d5_click_full_action_token_residual_w1_ema_256x128_steps1000}"
checkpoint="${output}/step-00001000.pt"

if [[ ! -s "${residual_cache}" ]]; then
  "${python_bin}" scripts/physical_tokens/export_action_token_residual_sidecar.py \
    --action-target-cache "${action_targets}" --action-token-cache "${action_tokens}" \
    --action-tokenizer-checkpoint "${action_tokenizer}" --output "${residual_cache}" \
    --device "${D5_DEVICE:-cuda:0}"
fi
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D5 checkpoint: ${checkpoint}"
elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "D5 refused to overwrite partial output: ${output}" >&2
  exit 2
else
  SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${residual_cache}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=1.0 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
  SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES="5,20,25,39,49" \
  SPATIOTEMPORAL_MAX_STEPS=1000 \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
fi

"${python_bin}" scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" --action-target-cache "${residual_cache}" \
  --checkpoint "${checkpoint}" --output "${output}/evaluation.json" \
  --validation-episodes 5,20,25,39,49 --device "${D5_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/compare_spatiotemporal_tokenizers.py \
  --baseline "${control}/evaluation.json" --candidate "${output}/evaluation.json" \
  --allowed-model-config-differences action_aux_dim,action_aux_weight,action_aux_target_mode,action_gripper_weight,action_velocity_weight \
  --comparison-name paired_same_cache_d5_control_vs_action_token_residual \
  --output "${output}/comparison_control_vs_residual_aux.json"
