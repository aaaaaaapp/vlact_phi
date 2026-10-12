#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
action_cache="artifacts/action_caches/click_alarmclock_clean_wrap32.npz"
predicted_tokens="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
action_tokenizer="artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt"
residual_cache="artifacts/action_caches/click_alarmclock_pred_action4_residual_wrap32.npz"
output="${D6_TOKENIZER_OUTPUT:-artifacts/action_tokenizer_residual/ema_16x128_steps5000}"
checkpoint="${output}/step-00005000.pt"
residual_ids="artifacts/token_caches/click_alarmclock_oracle_action_residual_ema_16x128_step5000.npz"
visual_ids="artifacts/token_caches/click_alarmclock_oracle_physics_spatiotemporal_d4_full_control_ema_256x128_step1000.npz"
combined="artifacts/token_caches/click_alarmclock_oracle_physics_d6_visual256_residual16.npz"
aligned="artifacts/token_caches/click_alarmclock_pred_action4_oracle_d6_visual256_residual16.npz"
shuffled="artifacts/token_caches/click_alarmclock_pred_action4_shuffled_d6_visual256_residual16_seed42.npz"

if [[ ! -s "${residual_cache}" ]]; then
  "${python_bin}" scripts/physical_tokens/export_action_token_residual_cache.py \
    --action-cache "${action_cache}" --predicted-token-cache "${predicted_tokens}" \
    --action-tokenizer-checkpoint "${action_tokenizer}" --output "${residual_cache}" \
    --device "${D6_DEVICE:-cuda:0}"
fi
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D6 residual tokenizer: ${checkpoint}"
elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "D6 refused to overwrite partial output: ${output}" >&2
  exit 2
else
  "${python_bin}" scripts/physical_tokens/train_action_tokenizer.py \
    --input "${residual_cache}" --output-dir "${output}" \
    --quantizer ema_vq --num-tokens 16 --codebook-size 128 \
    --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
    --max-steps 5000 --eval-interval 500 --save-interval 5000 \
    --gripper-dims "" --angular-dims 0,1,2,3,4,5,6,7,8,9,10,11 \
    --velocity-weight 0.25 --gripper-weight 0 --wrap-weight 0.25 \
    --device "${D6_DEVICE:-cuda:0}"
fi
"${python_bin}" scripts/physical_tokens/export_action_token_ids.py \
  --input "${residual_cache}" --checkpoint "${checkpoint}" --output "${residual_ids}" \
  --device "${D6_DEVICE:-cuda:0}"
"${python_bin}" scripts/physical_tokens/concatenate_physics_token_caches.py \
  --visual-cache "${visual_ids}" --residual-cache "${residual_ids}" --output "${combined}"
"${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
  --action-cache "${predicted_tokens}" --physics-cache "${combined}" --output "${aligned}"
"${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
  --action-cache "${predicted_tokens}" --physics-cache "${combined}" \
  --shuffle-physics-seed 42 --output "${shuffled}"
"${python_bin}" scripts/physical_tokens/probe_physics_action_relevance.py \
  --aligned-cache "${aligned}" --shuffled-cache "${shuffled}" \
  --dataset-dir /mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin/click_alarmclock \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output artifacts/evaluations/d6_visual256_residual16_action_relevance_decoder.json \
  --physics-representation residual_token_decoder \
  --physics-tokenizer-checkpoint "${checkpoint}" --residual-token-count 16 \
  --device "${D6_DEVICE:-cuda:0}"
