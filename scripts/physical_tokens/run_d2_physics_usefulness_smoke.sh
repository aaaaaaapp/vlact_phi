#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${1:-20}"
batch_size="${2:-8}"
weight="${PHYSICS_USEFULNESS_WEIGHT:-0.5}"
margin="${PHYSICS_USEFULNESS_MARGIN:-0.01}"
if [[ "${steps}" != "20" || "${weight}" != "0.5" || "${margin}" != "0.01" ]]; then
  echo "Preregistered usefulness smoke requires steps=20, weight=0.5, margin=0.01" >&2
  exit 2
fi

d2_output="artifacts/physical_language_tokenizers/spatiotemporal_d2_8tasks_spatial_aligned_ema_256x128_steps1000"
tokenizer_checkpoint="${d2_output}/step-00001000.pt"
tokenizer_sha256="24403f39af17178a9285f678883d1fae814c0003dcb1b9fc515ddcb9e63cef17"
token_cache="artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_d2_spatial_aligned_ema_physics256.npz"
action_init="results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt"
for required in "${tokenizer_checkpoint}" "${token_cache}" "${action_init}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required usefulness-smoke input is missing: ${required}" >&2
    exit 2
  fi
done

D1_COMPARISON_REPORT="${d2_output}/comparison_d1_vs_d2.json" \
D1_STRUCTURE_REPORT="${d2_output}/token_structure_per_task.json" \
D1_ALIGNED_CACHE="${token_cache}" \
ACTION_INIT_CHECKPOINT="${action_init}" \
PHYSICS_EXPERIMENT_TAG=d2_spatial_st256_cbinit_usefulpsw05_8gpu \
PHYSICS_TOKENIZER_INIT_CHECKPOINT="${tokenizer_checkpoint}" \
PHYSICS_TOKENIZER_INIT_SHA256="${tokenizer_sha256}" \
PHYSICS_USEFULNESS_WEIGHT="${weight}" \
PHYSICS_USEFULNESS_MARGIN="${margin}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29690}" \
bash scripts/physical_tokens/run_dit_physics_adapter_smoke.sh first "${steps}" "${batch_size}"
