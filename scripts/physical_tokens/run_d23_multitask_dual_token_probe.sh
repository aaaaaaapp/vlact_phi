#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz"
physics_checkpoint="artifacts/physical_language_tokenizers/d20_predictable_residual_ema_256x128_steps1000/step-00001000.pt"
physics_cache="artifacts/token_caches/robotwin_d23_8tasks_oracle_physics256_ema128.npz"
action_cache="artifacts/token_caches/robotwin_d16_8tasks_oracle_shared_ema4x128.npz"
dual_cache="artifacts/token_caches/robotwin_d23_8tasks_dual_action4_physics256.npz"
config="examples/Robotwin/train_files/qwen_dual_token_probe_d23_8tasks.yaml"
steps="${D23_STEPS:-500}"
run_id="qwen_dual_token_probe_d23_action4_physics256_8tasks_seed42_steps${steps}"
run_dir="results/Checkpoints/${run_id}"
checkpoint="${run_dir}/checkpoints/steps_${steps}_pytorch_model.pt"
evaluation="artifacts/evaluations/d23_multitask_dual_action4_physics256_steps${steps}.json"

if [[ ! -s "${physics_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
    --input "${feature_cache}" --checkpoint "${physics_checkpoint}" \
    --output "${physics_cache}" --batch-size 32 --num-workers 2 \
    --device "${D23_CACHE_DEVICE:-cuda:0}"
fi
if [[ ! -s "${dual_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/merge_multitask_dual_token_targets.py \
    --action-cache "${action_cache}" --physics-cache "${physics_cache}" \
    --output "${dual_cache}"
fi

if [[ ! -s "${checkpoint}" ]]; then
  if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "D23 refused partial output directory: ${run_dir}" >&2
    exit 2
  fi
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${D23_NPROC:-8}" --main_process_port "${MAIN_PROCESS_PORT:-29659}" \
    starVLA/training/train_starvla.py --config_yaml "${config}" \
    --trainer.max_train_steps "${steps}" --trainer.num_warmup_steps "$((steps / 10))" \
    --trainer.eval_interval "$((steps + 1))" --trainer.save_interval "${steps}" \
    --run_id "${run_id}"
else
  echo "Reusing completed D23 checkpoint: ${checkpoint}"
fi

if [[ ! -s "${evaluation}" ]]; then
  CUDA_VISIBLE_DEVICES="${D23_EVAL_GPU:-0}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_multitask_dual_token_probe.py \
    --config "${config}" --checkpoint "${checkpoint}" \
    --baseline-config examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml \
    --baseline-checkpoint results/Checkpoints/qwen_action_token_probe_d16_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt \
    --token-cache "${dual_cache}" \
    --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt \
    --physics-tokenizer-checkpoint "${physics_checkpoint}" \
    --output "${evaluation}" --validation-episodes 7,8 \
    --samples-per-task "${D23_SAMPLES_PER_TASK:-32}" \
    --batch-size "${D23_EVAL_BATCH_SIZE:-2}" --device cuda:0
fi
