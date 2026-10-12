#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
device="${D8_DEVICE:-cuda:0}"
checkpoint="results/Checkpoints/qwen_action_token_probe_residual4x64_click_alarmclock_seed42_steps500/checkpoints/steps_500_pytorch_model.pt"
oracle_residual="artifacts/token_caches/click_alarmclock_oracle_action_residual_ema_4x64_step5000.npz"
predicted_residual="artifacts/token_caches/click_alarmclock_predicted_residual_ema_4x64_probe_step500.npz"
predicted_action="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
tokenizer="artifacts/action_tokenizer_residual/ema_4x64_steps5000/step-00005000.pt"
aligned="artifacts/token_caches/click_alarmclock_pred_action4_pred_residual4x64_step500.npz"
shuffled="artifacts/token_caches/click_alarmclock_pred_action4_shuffled_pred_residual4x64_step500_seed42.npz"
evaluation="artifacts/evaluations/d8_predicted_residual4x64_learned_gating.json"

if [[ ! -s "${predicted_residual}" ]]; then
  CUDA_VISIBLE_DEVICES="${D8_GPU:-0}" "${python_bin}" -u \
    scripts/physical_tokens/export_predicted_action_token_ids.py \
    --config examples/Robotwin/train_files/qwen_action_token_probe_click_alarmclock.yaml \
    --checkpoint "${checkpoint}" --source-token-cache "${oracle_residual}" \
    --output "${predicted_residual}" --token-count 4 --vocab-size 64 \
    --batch-size 8 --device "${device}"
fi

if [[ ! -s "${aligned}" ]]; then
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${predicted_action}" --physics-cache "${predicted_residual}" \
    --physics-token-key action_token_ids --output "${aligned}"
fi
if [[ ! -s "${shuffled}" ]]; then
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${predicted_action}" --physics-cache "${predicted_residual}" \
    --physics-token-key action_token_ids --shuffle-physics-seed 42 --output "${shuffled}"
fi
if [[ -s "${evaluation}" ]]; then
  echo "D8 refused to overwrite evaluation: ${evaluation}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${D8_GPU:-0}" "${python_bin}" -u \
  scripts/physical_tokens/probe_physics_action_relevance.py \
  --aligned-cache "${aligned}" --shuffled-cache "${shuffled}" \
  --dataset-dir /mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin/click_alarmclock \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output "${evaluation}" --physics-representation residual_token_decoder \
  --physics-tokenizer-checkpoint "${tokenizer}" --residual-token-count 4 \
  --device "${device}"
