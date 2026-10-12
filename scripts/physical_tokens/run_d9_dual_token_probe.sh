#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
action_targets="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
physics_targets="artifacts/token_caches/click_alarmclock_oracle_action_residual_ema_4x64_step5000.npz"
dual_targets="artifacts/token_caches/click_alarmclock_dual_targets_action4_residual4x64.npz"
run_id="qwen_dual_token_probe_action4x128_residual4x64_click_alarmclock_seed42_steps500"
run_dir="results/Checkpoints/${run_id}"
checkpoint="${run_dir}/checkpoints/steps_500_pytorch_model.pt"
evaluation="artifacts/evaluations/d9_dual_action4x128_residual4x64_steps500_380samples.json"

if [[ ! -s "${dual_targets}" ]]; then
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${action_targets}" --physics-cache "${physics_targets}" \
    --physics-token-key action_token_ids --output "${dual_targets}"
fi

if [[ ! -s "${checkpoint}" ]]; then
  if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "D9 refused to overwrite partial probe output: ${run_dir}" >&2
    exit 2
  fi
  WANDB_MODE=disabled /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 --main_process_port "${MAIN_PROCESS_PORT:-29648}" \
    starVLA/training/train_starvla.py \
    --config_yaml examples/Robotwin/train_files/qwen_dual_token_probe_click_alarmclock.yaml \
    --datasets.vla_data.discrete_token_cache "${dual_targets}" \
    --trainer.max_train_steps 500 --trainer.num_warmup_steps 50 \
    --trainer.eval_interval 501 --trainer.save_interval 500 --run_id "${run_id}"
else
  echo "Reusing completed D9 dual-token probe: ${checkpoint}"
fi

if [[ -s "${evaluation}" ]]; then
  echo "D9 refused to overwrite evaluation: ${evaluation}" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "${python_bin}" -u \
  scripts/physical_tokens/evaluate_dual_token_probe.py \
  --config examples/Robotwin/train_files/qwen_dual_token_probe_click_alarmclock.yaml \
  --checkpoint "${checkpoint}" --token-cache "${dual_targets}" \
  --action-tokenizer-checkpoint artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt \
  --physics-tokenizer-checkpoint artifacts/action_tokenizer_residual/ema_4x64_steps5000/step-00005000.pt \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --output "${evaluation}" --num-samples 380 --batch-size 4 --seed 42 --device cuda:0
