#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
steps="${D53_STEPS:-20}"
run_tag="${D53_RUN_TAG:-soft_joint_low_lr}"
run_id="qwenpi_v4_d53_${run_tag}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"
d52_checkpoint="results/Checkpoints/qwenpi_v4_d50_frozen_policy_predicted_seed42_steps500/checkpoints/steps_500_pytorch_model.pt"
d52_sha256="f1591e1877f78798488f5bea17202ff5cee815c6a185e05d9849fcad4a385c29"

if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed D53 checkpoint: ${checkpoint}"
  exit 0
fi
if [[ ! -s "${d52_checkpoint}" ]]; then
  echo "Missing D52 initialization checkpoint: ${d52_checkpoint}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
WANDB_MODE=disabled "${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-7}" \
  --main_process_port "${D53_PORT:-30073}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml \
  --framework.joint_token_prediction.condition_source soft \
  --framework.joint_token_prediction.soft_temperature 1.0 \
  --trainer.joint_token_predictor_init_checkpoint "${d52_checkpoint}" \
  --trainer.joint_token_predictor_init_sha256 "${d52_sha256}" \
  --trainer.policy_extension_init_checkpoint "${d52_checkpoint}" \
  --trainer.policy_extension_init_sha256 "${d52_sha256}" \
  --trainer.learning_rate.pi_discrete_conditioner "${D53_CONDITIONER_LR:-1.0e-5}" \
  --trainer.learning_rate.pi_action_layer_gates "${D53_GATE_LR:-1.0e-4}" \
  --trainer.learning_rate.pi_physics_layer_gates "${D53_GATE_LR:-1.0e-4}" \
  --trainer.learning_rate.action_token_probe "${D53_ACTION_PROBE_LR:-1.0e-5}" \
  --trainer.learning_rate.action_condition_embedding "${D53_ACTION_EMBED_LR:-1.0e-5}" \
  --trainer.learning_rate.physics_token_probe "${D53_PHYSICS_PROBE_LR:-1.0e-5}" \
  --trainer.max_train_steps "${steps}" \
  --trainer.save_interval "${steps}" \
  --trainer.eval_interval "$((steps + 1))" \
  --run_id "${run_id}"
