#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
steps="${P64_D52_STEPS:-500}"
run_id="${P64_D52_RUN_ID:-qwenpi_v4_p64_d52style_predictor_only_seed42_steps${steps}}"
init="results/Checkpoints/qwenpi_v4_p64_m32_predicted_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt"
cache="artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz"
tok="artifacts/physical_language_tokenizers/p64_phi_residual_ema_64x128_full_steps20000/step-00020000.pt"
[[ -s "$init" && -s "$cache" && -s "$tok" ]] || { echo missing input >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" WANDB_MODE=disabled "$accelerate_bin" launch \
 --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes "${NUM_PROCESSES:-8}" --main_process_port "${P64_D52_PORT:-30221}" \
 starVLA/training/train_starvla.py --config_yaml examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml \
 --datasets.vla_data.per_device_batch_size "${P64_D52_BATCH:-16}" \
 --framework.discrete_conditioning.max_physics_tokens 64 --framework.discrete_conditioning.physics_group_size 2 --framework.discrete_conditioning.physics_vocab_size 128 \
 --framework.joint_token_prediction.physics_token_count 64 --framework.joint_token_prediction.physics_vocab_size 128 --framework.joint_token_prediction.condition_source predicted \
 --datasets.vla_data.discrete_token_cache "./$cache" --trainer.physics_tokenizer_init_checkpoint "./$tok" --trainer.physics_tokenizer_init_sha256 632c065745d8ee2754b91cbe0ebe494b77ca09fc589bb741a43848f26d396605 \
 --trainer.joint_token_predictor_init_checkpoint "./$init" --trainer.policy_extension_init_checkpoint "./$init" \
 --trainer.learning_rate.base 0.0 --trainer.learning_rate.action_condition_embedding 0.0 --trainer.learning_rate.pi_discrete_conditioner 0.0 --trainer.learning_rate.pi_action_layer_gates 0.0 --trainer.learning_rate.pi_physics_layer_gates 0.0 \
 --trainer.learning_rate.action_token_probe 1.0e-5 --trainer.learning_rate.physics_token_probe 1.0e-5 \
 --trainer.max_train_steps "$steps" --trainer.save_interval "$steps" --trainer.eval_interval "$((steps+1))" --run_id "$run_id"
