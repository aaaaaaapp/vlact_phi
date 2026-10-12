#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

max_steps="${SPATIOTEMPORAL_POLICY_STEPS:-500}"
batch_size="${SPATIOTEMPORAL_POLICY_BATCH_SIZE:-8}"
port="${MAIN_PROCESS_PORT:-29660}"
log_dir="artifacts/logs/spatiotemporal_policy_ablation"
mkdir -p "${log_dir}"

run_one() {
  local mode="$1"
  local cache="$2"
  local suffix="$3"
  local log="${log_dir}/${mode}${suffix}_steps${max_steps}.log"
  echo "starting ${mode}${suffix}: cache=${cache}"
  TOKEN_CACHE="${cache}" \
    MAX_PHYSICS_TOKENS=256 \
    PHYSICS_GATE_INIT="${PHYSICS_GATE_INIT:-0.5}" \
    PHYSICS_CONDITION_DROPOUT="${PHYSICS_CONDITION_DROPOUT:-0.2}" \
    ACTION_CONDITION_DROPOUT=0.0 \
    VARIANT_SUFFIX="${suffix}" \
    MAIN_PROCESS_PORT="${port}" \
    WANDB_MODE=disabled \
    bash scripts/physical_tokens/run_qwenpi_oracle_ablation.sh \
      "${mode}" "${max_steps}" "${batch_size}" >"${log}" 2>&1
  tail -n 8 "${log}"
}

run_one \
  b3 \
  artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz \
  _st256_drop20_pg05

run_one \
  b4 \
  artifacts/token_caches/click_alarmclock_pred_action4_shuffled_spatiotemporal_physics256_seed42.npz \
  _st256_shuffled_drop20_pg05
