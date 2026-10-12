#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
config="examples/Robotwin/train_files/qwenpi_v4_physical_tokens_d48_8tasks.yaml"
steps="${D49_STEPS:-20}"
num_processes="${NUM_PROCESSES:-7}"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
memories="${D49_MEMORIES:-8 32}"

if [[ ! "${steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "D49_STEPS must be a positive integer" >&2
  exit 2
fi

base_checkpoint="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt"
action_tokenizer="artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt"
physics_tokenizer="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt"
token_cache="artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz"
for required in "${config}" "${base_checkpoint}" "${action_tokenizer}" "${physics_tokenizer}" "${token_cache}"; do
  [[ -s "${required}" ]] || { echo "missing D49 input: ${required}" >&2; exit 2; }
done

port="${D49_PORT_BASE:-30049}"
for memory in ${memories}; do
  case "${memory}" in
    8) group_size=32 ;;
    16) group_size=16 ;;
    32) group_size=8 ;;
    *) echo "D49 memory must be one of 8, 16, or 32; got ${memory}" >&2; exit 2 ;;
  esac
  run_id="qwenpi_v4_d49_physics_memory${memory}_seed42_steps${steps}"
  checkpoint="results/Checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "Reusing completed D49 checkpoint: ${checkpoint}"
    port=$((port + 1))
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" WANDB_MODE=disabled "${accelerate_bin}" launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${num_processes}" \
    --main_process_port "${port}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config}" \
    --framework.discrete_conditioning.physics_group_size "${group_size}" \
    --framework.discrete_conditioning.physics_condition_enabled true \
    --trainer.max_train_steps "${steps}" \
    --trainer.num_warmup_steps "$(( steps < 20 ? 2 : steps / 10 ))" \
    --trainer.save_interval "${steps}" \
    --trainer.eval_interval "$((steps + 1))" \
    --run_id "${run_id}"
  port=$((port + 1))
done
