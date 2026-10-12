#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
selection="artifacts/evaluations/d29_full_diversity_codebook_gate.json"
feature_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz"
target_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz"
action_cache="artifacts/token_caches/robotwin_d16_8tasks_oracle_shared_ema4x128.npz"
action_tokenizer="artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt"
physics_cache="artifacts/token_caches/robotwin_d30_8tasks_oracle_physics256_selected_ema.npz"
dual_cache="artifacts/token_caches/robotwin_d30_8tasks_dual_action4_physics256_selected_ema.npz"
conditioned_checkpoint="artifacts/physical_language_tokenizers/d30_action_conditioned_selected_steps3000.pt"
conditioned_report="${conditioned_checkpoint}.json"
gate_report="artifacts/evaluations/d30_action_conditioned_oracle_gate.json"

if [[ ! -s "${selection}" ]]; then
  echo "D30 selection report is missing: ${selection}" >&2
  exit 1
fi
physics_checkpoint=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${selection}")
if [[ ! -s "${physics_checkpoint}" ]]; then
  echo "D30 selected physics checkpoint is missing: ${physics_checkpoint}" >&2
  exit 1
fi

if [[ ! -s "${physics_cache}" ]]; then
  CUDA_VISIBLE_DEVICES="${D30_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
    --input "${feature_cache}" --checkpoint "${physics_checkpoint}" \
    --output "${physics_cache}" --batch-size 32 --num-workers 2 --device cuda:0
fi
if [[ ! -s "${dual_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/merge_multitask_dual_token_targets.py \
    --action-cache "${action_cache}" --physics-cache "${physics_cache}" --output "${dual_cache}"
fi
if [[ ! -s "${conditioned_checkpoint}" || ! -s "${conditioned_report}" ]]; then
  if [[ -e "${conditioned_checkpoint}" || -e "${conditioned_report}" ]]; then
    echo "D30 refused partial conditioned-probe output" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${D30_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/train_action_conditioned_physics_residual_probe.py \
    --physics-cache "${physics_cache}" --action-token-cache "${action_cache}" \
    --action-target-cache "${target_cache}" --feature-cache "${feature_cache}" \
    --physics-checkpoint "${physics_checkpoint}" \
    --action-tokenizer-checkpoint "${action_tokenizer}" \
    --output "${conditioned_checkpoint}" --validation-episodes 5,20,25,39,49 \
    --steps 3000 --batch-size 512 --eval-interval 100 --device cuda:0 \
    >"${conditioned_checkpoint}.log" 2>&1
fi
"${python_bin}" -u scripts/physical_tokens/require_d30_oracle_conditioned_gate.py \
  --input "${conditioned_report}" --output "${gate_report}"
