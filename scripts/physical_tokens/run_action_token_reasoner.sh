#!/usr/bin/env bash
set -euo pipefail

max_train_steps="${1:-500}"
per_device_batch_size="${2:-1}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-29630}"
warmup_steps="${NUM_WARMUP_STEPS:-$((max_train_steps / 10))}"
run_id="qwen_action_token_reasoner_click_alarmclock_seed42_steps${max_train_steps}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwen_action_token_reasoner_click_alarmclock.yaml \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${warmup_steps}" \
  --trainer.eval_interval "$((max_train_steps + 1))" \
  --trainer.save_interval "${max_train_steps}" \
  --run_id "${run_id}"
