#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
variant="residual4x64"
steps=500
run_dir="results/Checkpoints/qwen_action_token_probe_${variant}_click_alarmclock_seed42_steps${steps}"
checkpoint="${run_dir}/checkpoints/steps_${steps}_pytorch_model.pt"
residual_cache="artifacts/token_caches/click_alarmclock_oracle_action_residual_ema_4x64_step5000.npz"
residual_tokenizer="artifacts/action_tokenizer_residual/ema_4x64_steps5000/step-00005000.pt"
action_tokenizer="artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt"
predicted_actions="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
evaluation="artifacts/evaluations/d7_residual4x64_vlm_probe_steps500_380samples.json"

if [[ ! -s "${checkpoint}" ]]; then
  if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "D7 refused to overwrite partial probe output: ${run_dir}" >&2
    exit 2
  fi
  PROBE_VARIANT="${variant}" \
  PROBE_TOKEN_COUNT=4 \
  PROBE_VOCAB_SIZE=64 \
  PROBE_TOKEN_CACHE="${residual_cache}" \
  NUM_PROCESSES=8 \
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29647}" \
  WANDB_MODE=disabled \
    bash scripts/physical_tokens/run_action_token_probe.sh "${steps}" 1
else
  echo "Reusing completed D7 VLM probe: ${checkpoint}"
fi

if [[ -s "${evaluation}" ]]; then
  echo "D7 refused to overwrite evaluation: ${evaluation}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "${python_bin}" -u \
  scripts/physical_tokens/evaluate_residual_token_probe.py \
  --config examples/Robotwin/train_files/qwen_action_token_probe_click_alarmclock.yaml \
  --checkpoint "${checkpoint}" \
  --residual-token-cache "${residual_cache}" \
  --predicted-action-token-cache "${predicted_actions}" \
  --residual-tokenizer-checkpoint "${residual_tokenizer}" \
  --action-tokenizer-checkpoint "${action_tokenizer}" \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output "${evaluation}" \
  --num-samples 380 --batch-size 4 --seed 42 --device cuda:0 \
  --token-count 4 --vocab-size 64
