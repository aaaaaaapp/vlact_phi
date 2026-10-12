#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
samples_per_task="${D31B_SAMPLES_PER_TASK:-256}"
output_tag="${D31B_OUTPUT_TAG:-d31b}"
steps="${D31_EVAL_STEPS:-2000}"
if [[ "${samples_per_task}" != "256" ]]; then
  echo "D31b is preregistered for exactly 256 samples per task" >&2
  exit 2
fi
if [[ ! "${output_tag}" =~ ^[a-z0-9_]+$ ]]; then
  echo "D31B_OUTPUT_TAG must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi

config="artifacts/configs/qwen_dual_token_probe_d31_full_diversity_selected_steps${steps}.yaml"
checkpoint="results/Checkpoints/qwen_dual_token_probe_d31_full_diversity_k256_steps${steps}/checkpoints/steps_${steps}_pytorch_model.pt"
evaluation="artifacts/evaluations/${output_tag}_full_diversity_dual_k256_steps${steps}_256samples.json"
gate="artifacts/evaluations/${output_tag}_full_diversity_predicted_gate_k256_steps${steps}_256samples.json"
conditioned_checkpoint="${D31_CONDITIONED_CHECKPOINT:-artifacts/physical_language_tokenizers/d30_action_conditioned_selected_steps3000.pt}"

for required in "${config}" "${checkpoint}" "${conditioned_checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    echo "D31b required input is missing: ${required}" >&2
    exit 2
  fi
done

if [[ ! -s "${evaluation}" ]]; then
  CUDA_VISIBLE_DEVICES="${D31B_EVAL_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_multitask_dual_token_probe.py \
    --config "${config}" --checkpoint "${checkpoint}" \
    --baseline-config examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml \
    --baseline-checkpoint results/Checkpoints/qwen_action_token_probe_d16_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt \
    --token-cache artifacts/token_caches/robotwin_d30_8tasks_dual_action4_physics256_selected_ema.npz \
    --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt \
    --physics-tokenizer-checkpoint artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt \
    --conditioned-decoder-checkpoint "${conditioned_checkpoint}" \
    --output "${evaluation}" --validation-episodes 5,20,25,39,49 \
    --samples-per-task "${samples_per_task}" --batch-size "${D31B_EVAL_BATCH_SIZE:-2}" \
    --device cuda:0
fi

"${python_bin}" -u scripts/physical_tokens/require_d31_predicted_physics_gate.py \
  --input "${evaluation}" --output "${gate}"
