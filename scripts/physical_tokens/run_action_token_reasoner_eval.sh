#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-}"
num_samples="${2:-32}"
gpu="${3:-0}"
if [[ -z "${checkpoint}" ]]; then
  echo "Usage: $0 CHECKPOINT [num_samples] [gpu]" >&2
  exit 2
fi

checkpoint_name="$(basename "${checkpoint}" .pt)"
output="artifacts/evaluations/action_reasoner_${checkpoint_name}_${num_samples}samples.json"
CUDA_VISIBLE_DEVICES="${gpu}" /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python \
  scripts/physical_tokens/evaluate_action_token_reasoner.py \
  --config examples/Robotwin/train_files/qwen_action_token_reasoner_click_alarmclock.yaml \
  --checkpoint "${checkpoint}" \
  --action-tokenizer-checkpoint artifacts/action_tokenizer_runs_v2/ema_vq_steps5000/step-00005000.pt \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --num-samples "${num_samples}" \
  --seed 42 \
  --device cuda:0 \
  --output "${output}"

echo "Wrote ${output}"
