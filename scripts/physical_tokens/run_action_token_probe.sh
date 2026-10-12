#!/usr/bin/env bash
set -euo pipefail

max_train_steps="${1:-500}"
per_device_batch_size="${2:-1}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-29640}"
warmup_steps="${NUM_WARMUP_STEPS:-$((max_train_steps / 10))}"
variant="${PROBE_VARIANT:-8x512}"
token_count="${PROBE_TOKEN_COUNT:-8}"
vocab_size="${PROBE_VOCAB_SIZE:-512}"
token_cache="${PROBE_TOKEN_CACHE:-./artifacts/token_caches/click_alarmclock_clean_ema_step5000.npz}"
run_id="qwen_action_token_probe_${variant}_click_alarmclock_seed42_steps${max_train_steps}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwen_action_token_probe_click_alarmclock.yaml \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.discrete_token_cache "${token_cache}" \
  --framework.action_token_probe.token_count "${token_count}" \
  --framework.action_token_probe.vocab_size "${vocab_size}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${warmup_steps}" \
  --trainer.eval_interval "$((max_train_steps + 1))" \
  --trainer.save_interval "${max_train_steps}" \
  --run_id "${run_id}"
