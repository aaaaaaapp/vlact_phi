#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

mode="${1:?usage: $0 action_only|physics_only}"
case "${mode}" in
  action_only)
    run_id="${P64_ABLATION_RUN_ID:-qwenpi_v4_p64_actiononly_seed42_steps20000}"
    port="${P64_ABLATION_PORT:-30179}"
    physics_enabled=false
    action_enabled=true
    ;;
  physics_only)
    run_id="${P64_ABLATION_RUN_ID:-qwenpi_v4_p64_physicsonly_seed42_steps20000}"
    port="${P64_ABLATION_PORT:-30180}"
    physics_enabled=true
    action_enabled=false
    ;;
  *) echo "unknown ablation mode: ${mode}" >&2; exit 2 ;;
esac

steps="${P64_ABLATION_STEPS:-20000}"
checkpoint="results/Checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed checkpoint: ${checkpoint}"
  exit 0
fi

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
cache="artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz"
physics_tokenizer="artifacts/physical_language_tokenizers/p64_phi_residual_ema_64x128_full_steps20000/step-00020000.pt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" WANDB_MODE=disabled "${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  --main_process_port "${port}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml \
  --datasets.vla_data.per_device_batch_size "${P64_PER_DEVICE_BATCH_SIZE:-16}" \
  --framework.discrete_conditioning.max_physics_tokens 64 \
  --framework.discrete_conditioning.physics_group_size 2 \
  --framework.discrete_conditioning.physics_vocab_size 128 \
  --framework.discrete_conditioning.physics_condition_enabled "${physics_enabled}" \
  --framework.discrete_conditioning.action_condition_enabled "${action_enabled}" \
  --framework.joint_token_prediction.physics_token_count 64 \
  --framework.joint_token_prediction.physics_vocab_size 128 \
  --framework.joint_token_prediction.condition_source predicted \
  --datasets.vla_data.discrete_token_cache "./${cache}" \
  --trainer.physics_tokenizer_init_checkpoint "./${physics_tokenizer}" \
  --trainer.physics_tokenizer_init_sha256 632c065745d8ee2754b91cbe0ebe494b77ca09fc589bb741a43848f26d396605 \
  --trainer.joint_token_predictor_init_checkpoint null \
  --trainer.policy_extension_init_checkpoint null \
  --trainer.max_train_steps "${steps}" \
  --trainer.save_interval "${P64_ABLATION_SAVE_INTERVAL:-5000}" \
  --trainer.eval_interval "$((steps + 1))" \
  --run_id "${run_id}"
