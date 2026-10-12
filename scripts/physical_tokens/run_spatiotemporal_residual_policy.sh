#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${SPATIOTEMPORAL_RESIDUAL_STEPS:-500}"
batch_size="${SPATIOTEMPORAL_RESIDUAL_BATCH_SIZE:-8}"
physics_gate_init="${PHYSICS_GATE_INIT:-0.5}"
suffix="${SPATIOTEMPORAL_RESIDUAL_SUFFIX:-_st256_resxattn_drop20_pg05}"
log_dir="artifacts/logs/spatiotemporal_policy_ablation"
log="${log_dir}/b3${suffix}_steps${steps}.log"
mkdir -p "${log_dir}"

TOKEN_CACHE=artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz \
  MAX_PHYSICS_TOKENS=256 \
  PHYSICS_GATE_INIT="${physics_gate_init}" \
  PHYSICS_CONDITION_DROPOUT=0.2 \
  ACTION_CONDITION_DROPOUT=0.0 \
  PHYSICS_FUSION_MODE=residual_cross_attention \
  PHYSICS_FUSION_DIM="${PHYSICS_FUSION_DIM:-256}" \
  PHYSICS_FUSION_HEADS="${PHYSICS_FUSION_HEADS:-8}" \
  PHYSICS_CONDITION_ENABLED="${PHYSICS_CONDITION_ENABLED:-true}" \
  VARIANT_SUFFIX="${suffix}" \
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29670}" \
  WANDB_MODE=disabled \
  bash scripts/physical_tokens/run_qwenpi_oracle_ablation.sh \
    b3 "${steps}" "${batch_size}" >"${log}" 2>&1

tail -n 8 "${log}"
