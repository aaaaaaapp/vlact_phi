#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
physics_cache="artifacts/token_caches/robotwin_d30_8tasks_oracle_physics256_selected_ema.npz"
predicted_action_cache="artifacts/token_caches/robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz"
target_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz"
feature_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz"
physics_checkpoint="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt"
action_tokenizer="artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt"
output="artifacts/physical_language_tokenizers/d32_predicted_action_conditioned_steps3000.pt"
report="${output}.json"
gate="artifacts/evaluations/d32_predicted_action_conditioned_oracle_gate.json"

for required in "${physics_cache}" "${predicted_action_cache}" "${target_cache}" \
  "${feature_cache}" "${physics_checkpoint}" "${action_tokenizer}"; do
  if [[ ! -s "${required}" ]]; then
    echo "D32 required input is missing: ${required}" >&2
    exit 2
  fi
done

if [[ ! -s "${output}" || ! -s "${report}" ]]; then
  if [[ -e "${output}" || -e "${report}" ]]; then
    echo "D32 refused partial conditioned-head output" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${D32_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/train_action_conditioned_physics_residual_probe.py \
    --physics-cache "${physics_cache}" --action-token-cache "${predicted_action_cache}" \
    --action-target-cache "${target_cache}" --feature-cache "${feature_cache}" \
    --physics-checkpoint "${physics_checkpoint}" \
    --action-tokenizer-checkpoint "${action_tokenizer}" \
    --output "${output}" --validation-episodes 5,20,25,39,49 \
    --steps 3000 --batch-size 512 --eval-interval 100 --device cuda:0
fi

"${python_bin}" -u scripts/physical_tokens/require_d30_oracle_conditioned_gate.py \
  --input "${report}" --output "${gate}"
