#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

max_steps="${POLICY_SWEEP_STEPS:-500}"
batch_size="${POLICY_SWEEP_BATCH_SIZE:-8}"
port="${MAIN_PROCESS_PORT:-29650}"
log_dir="artifacts/logs/physics_token_count_policy_sweep"
mkdir -p "${log_dir}"

run_one() {
  local mode="$1"
  local count="$2"
  local cache="$3"
  local suffix="$4"
  local log="${log_dir}/${mode}${suffix}_steps${max_steps}.log"
  echo "starting ${mode}${suffix}: physics_tokens=${count} cache=${cache}"
  TOKEN_CACHE="${cache}" \
    MAX_PHYSICS_TOKENS="${count}" \
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

if [[ "${POLICY_SWEEP_SKIP_E2:-0}" != "1" ]]; then
  run_one \
    e2 \
    4 \
    artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz \
    _count_sweep
fi

read -r -a physics_counts <<<"${POLICY_SWEEP_COUNTS:-16 64 256}"
for count in "${physics_counts[@]}"; do
  run_one \
    b3 \
    "${count}" \
    "artifacts/token_caches/click_alarmclock_pred_action4_oracle_physics${count}.npz" \
    "_p${count}_drop20_pg05"
done
