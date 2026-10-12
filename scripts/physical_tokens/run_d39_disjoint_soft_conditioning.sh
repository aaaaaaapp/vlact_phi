#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
common=(
  --config artifacts/configs/qwen_dual_token_probe_d31_full_diversity_selected_steps8000.yaml
  --checkpoint results/Checkpoints/qwen_dual_token_probe_d31_full_diversity_k256_steps8000/checkpoints/steps_8000_pytorch_model.pt
  --baseline-config examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml
  --baseline-checkpoint results/Checkpoints/qwen_action_token_probe_d16_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt
  --token-cache artifacts/token_caches/robotwin_d30_8tasks_dual_action4_physics256_selected_ema.npz
  --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt
  --physics-tokenizer-checkpoint artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt
  --conditioned-decoder-checkpoint artifacts/physical_language_tokenizers/d34_actual_residual_conditioned_steps3000.pt
  --soft-physics-temperatures 0.5,1,2
  --samples-per-task 128 --batch-size 2 --device cuda:0
)
calibration="artifacts/evaluations/d39_soft_calibration_episodes_5_20_128samples.json"
confirmation="artifacts/evaluations/d39_soft_confirmation_episodes_25_39_49_128samples.json"
selection="artifacts/evaluations/d39_disjoint_soft_confirmation.json"

if [[ ! -s "${calibration}" ]]; then
  CUDA_VISIBLE_DEVICES="${D39_CALIBRATION_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_multitask_dual_token_probe.py "${common[@]}" \
    --validation-episodes 5,20 --output "${calibration}" &
  calibration_pid=$!
else
  calibration_pid=""
fi

if [[ ! -s "${confirmation}" ]]; then
  CUDA_VISIBLE_DEVICES="${D39_CONFIRMATION_GPU:-6}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_multitask_dual_token_probe.py "${common[@]}" \
    --validation-episodes 25,39,49 --output "${confirmation}" &
  confirmation_pid=$!
else
  confirmation_pid=""
fi

if [[ -n "${calibration_pid}" ]]; then wait "${calibration_pid}"; fi
if [[ -n "${confirmation_pid}" ]]; then wait "${confirmation_pid}"; fi

"${python_bin}" -u scripts/physical_tokens/select_d39_soft_conditioning.py \
  --calibration "${calibration}" --confirmation "${confirmation}" --output "${selection}"
