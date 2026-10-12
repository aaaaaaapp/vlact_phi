#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:?usage: $0 CHECKPOINT [NUM_SAMPLES] [GPU]}"
num_samples="${2:-32}"
gpu="${3:-0}"
stem="$(basename "${checkpoint}" .pt)"
variant="${PROBE_VARIANT:-8x512}"
token_count="${PROBE_TOKEN_COUNT:-8}"
vocab_size="${PROBE_VOCAB_SIZE:-512}"
token_cache="${PROBE_TOKEN_CACHE:-artifacts/token_caches/click_alarmclock_clean_ema_step5000.npz}"
tokenizer_checkpoint="${PROBE_TOKENIZER_CHECKPOINT:-artifacts/action_tokenizer_runs_v2/ema_vq_steps5000/step-00005000.pt}"
output="artifacts/evaluations/action_probe_${variant}_${stem}_${num_samples}samples.json"

CUDA_VISIBLE_DEVICES="${gpu}" /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python -u \
  scripts/physical_tokens/evaluate_action_token_reasoner.py \
  --config examples/Robotwin/train_files/qwen_action_token_probe_click_alarmclock.yaml \
  --checkpoint "${checkpoint}" \
  --action-tokenizer-checkpoint "${tokenizer_checkpoint}" \
  --token-cache "${token_cache}" \
  --token-count "${token_count}" \
  --vocab-size "${vocab_size}" \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output "${output}" \
  --num-samples "${num_samples}" \
  --batch-size 4 \
  --device cuda:0
