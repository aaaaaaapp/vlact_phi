#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
steps="${D17_STEPS:-1000}"
run_id="qwen_action_token_probe_d16_8tasks_seed42_steps${steps}"
run_dir="results/Checkpoints/${run_id}"
checkpoint="${run_dir}/checkpoints/steps_${steps}_pytorch_model.pt"

if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D17 action-token probe: ${checkpoint}"
  exit 0
fi
if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "D17 refused to overwrite partial output: ${run_dir}" >&2
  exit 2
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 --main_process_port "${MAIN_PROCESS_PORT:-29657}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml \
  --trainer.max_train_steps "${steps}" --trainer.num_warmup_steps "$((steps / 10))" \
  --trainer.eval_interval "$((steps + 1))" --trainer.save_interval "${steps}" \
  --run_id "${run_id}"
