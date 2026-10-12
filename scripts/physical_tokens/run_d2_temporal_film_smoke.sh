#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${1:-20}"
batch_size="${2:-8}"
if [[ "${steps}" != "20" ]]; then
  echo "Preregistered temporal-FiLM smoke requires exactly 20 steps" >&2
  exit 2
fi

d2_output="artifacts/physical_language_tokenizers/spatiotemporal_d2_8tasks_spatial_aligned_ema_256x128_steps1000"
tokenizer_checkpoint="${d2_output}/step-00001000.pt"
token_cache="artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_d2_spatial_aligned_ema_physics256.npz"
action_init="results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt"
for required in "${tokenizer_checkpoint}" "${token_cache}" "${action_init}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required temporal-FiLM input is missing: ${required}" >&2
    exit 2
  fi
done

D1_COMPARISON_REPORT="${d2_output}/comparison_d1_vs_d2.json" \
D1_STRUCTURE_REPORT="${d2_output}/token_structure_per_task.json" \
D1_ALIGNED_CACHE="${token_cache}" \
ACTION_INIT_CHECKPOINT="${action_init}" \
PHYSICS_EXPERIMENT_TAG=d2_spatial_st256_cbinit_temporalfilm_8gpu \
PHYSICS_TOKENIZER_INIT_CHECKPOINT="${tokenizer_checkpoint}" \
PHYSICS_TOKENIZER_INIT_SHA256=24403f39af17178a9285f678883d1fae814c0003dcb1b9fc515ddcb9e63cef17 \
DIT_PHYSICS_ADAPTER_TYPE=temporal_film \
DIT_PHYSICS_MEMORY_POOLING=transition_mean \
PHYSICS_TOKENS_PER_TRANSITION=16 \
DIT_ACTION_TOKENS_ONLY=true \
PHYSICS_USEFULNESS_WEIGHT=0.0 \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29698}" \
bash scripts/physical_tokens/run_dit_physics_adapter_smoke.sh first "${steps}" "${batch_size}"
