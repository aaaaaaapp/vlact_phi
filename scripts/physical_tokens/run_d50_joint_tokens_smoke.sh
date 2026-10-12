#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
steps="${D50_STEPS:-2}"
condition_source="${D50_CONDITION_SOURCE:-cached}"
if [[ "${condition_source}" != "cached" && "${condition_source}" != "predicted" ]]; then
  echo "D50_CONDITION_SOURCE must be cached or predicted" >&2
  exit 2
fi
run_tag="${D50_RUN_TAG:-joint_${condition_source}}"
policy_lr="${D50_POLICY_LR:-1.0e-4}"
gate_lr="${D50_GATE_LR:-1.0e-2}"
run_id="qwenpi_v4_d50_${run_tag}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D50 checkpoint: ${checkpoint}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
WANDB_MODE=disabled "${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-7}" \
  --main_process_port "${D50_PORT:-30069}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml \
  --framework.joint_token_prediction.condition_source "${condition_source}" \
  --trainer.learning_rate.pi_discrete_conditioner "${policy_lr}" \
  --trainer.learning_rate.pi_action_layer_gates "${gate_lr}" \
  --trainer.learning_rate.pi_physics_layer_gates "${gate_lr}" \
  --trainer.max_train_steps "${steps}" \
  --trainer.save_interval "${steps}" \
  --trainer.eval_interval "$((steps + 1))" \
  --run_id "${run_id}"
