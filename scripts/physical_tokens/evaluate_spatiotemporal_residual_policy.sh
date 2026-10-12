#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${SPATIOTEMPORAL_RESIDUAL_STEPS:-500}"
num_samples="${SPATIOTEMPORAL_POLICY_EVAL_SAMPLES:-380}"
suffix="${SPATIOTEMPORAL_RESIDUAL_SUFFIX:-_st256_resxattn_drop20_pg05}"
run="qwenpi_physical_tokens_click_alarmclock_b3${suffix}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"

TOKEN_CACHE=artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz \
  MAX_PHYSICS_TOKENS=256 \
  PHYSICS_FUSION_MODE=residual_cross_attention \
  PHYSICS_FUSION_DIM=256 \
  PHYSICS_FUSION_HEADS=8 \
  PHYSICS_CONDITION_ENABLED="${PHYSICS_CONDITION_ENABLED:-true}" \
  EVAL_VARIANT_SUFFIX="${suffix}" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
    b3 "${checkpoint}" "${num_samples}" "${SPATIOTEMPORAL_EVAL_GPU:-0}"
