#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
steps="${P64_POLICY_STEPS:-20000}"
run_id="${P64_POLICY_RUN_ID:-qwenpi_v4_p64_m32_distill_medium_seed42_steps${steps}}"
checkpoint="results/Checkpoints/${run_id}/checkpoints/steps_${steps}_pytorch_model.pt"
export P64_ACTION_DISTILL_WEIGHT_START="${P64_ACTION_DISTILL_WEIGHT_START:-0.2}"
export P64_ACTION_DISTILL_WEIGHT_END="${P64_ACTION_DISTILL_WEIGHT_END:-0.02}"
cache="artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz"
physics_tokenizer="artifacts/physical_language_tokenizers/p64_phi_residual_ema_64x128_full_steps20000/step-00020000.pt"
physics_sha256="632c065745d8ee2754b91cbe0ebe494b77ca09fc589bb741a43848f26d396605"

[[ -s "${cache}" ]] || { echo "missing P64 token cache: ${cache}" >&2; exit 2; }
[[ -s "${physics_tokenizer}" ]] || { echo "missing P64 tokenizer: ${physics_tokenizer}" >&2; exit 2; }
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed P64-M32 checkpoint: ${checkpoint}"
  exit 0
fi

# P64 has 64 raw IDs from an EMA-VQ K=128 codebook.  Pairwise grouping gives
# 32 Pi memory tokens.  Predictor and conditioner parameters that depend on
# the old 256-token layout deliberately initialize afresh from the base VLM.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" WANDB_MODE=disabled "${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  --main_process_port "${P64_POLICY_PORT:-30178}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml \
  --datasets.vla_data.per_device_batch_size "${P64_PER_DEVICE_BATCH_SIZE:-16}" \
  --framework.discrete_conditioning.max_physics_tokens 64 \
  --framework.discrete_conditioning.physics_group_size 2 \
  --framework.discrete_conditioning.physics_vocab_size 128 \
  --framework.joint_token_prediction.physics_token_count 64 \
  --framework.joint_token_prediction.physics_vocab_size 128 \
  --framework.joint_token_prediction.condition_source predicted \
  --framework.joint_token_prediction.action_distill_weight_start "${P64_ACTION_DISTILL_WEIGHT_START:-0}" \
  --framework.joint_token_prediction.action_distill_weight_end "${P64_ACTION_DISTILL_WEIGHT_END:-0}" \
  --framework.joint_token_prediction.action_distill_warmup_steps "${P64_ACTION_DISTILL_WARMUP_STEPS:-20000}" \
  --datasets.vla_data.discrete_token_cache "./${cache}" \
  --trainer.physics_tokenizer_init_checkpoint "./${physics_tokenizer}" \
  --trainer.physics_tokenizer_init_sha256 "${physics_sha256}" \
  --trainer.joint_token_predictor_init_checkpoint null \
  --trainer.policy_extension_init_checkpoint null \
  --trainer.max_train_steps "${steps}" \
  --trainer.save_interval "${P64_POLICY_SAVE_INTERVAL:-5000}" \
  --trainer.eval_interval "$((steps + 1))" \
  --run_id "${run_id}"
