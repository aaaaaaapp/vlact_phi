#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${SPATIOTEMPORAL_POLICY_STEPS:-500}"
num_samples="${SPATIOTEMPORAL_POLICY_EVAL_SAMPLES:-380}"
log_dir="artifacts/logs/spatiotemporal_policy_eval"
mkdir -p "${log_dir}"
pids=()

evaluate_one() {
  local mode="$1"
  local cache="$2"
  local suffix="$3"
  local gpu="$4"
  local run="qwenpi_physical_tokens_click_alarmclock_${mode}${suffix}_seed42_steps${steps}"
  local checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"
  local log="${log_dir}/${mode}${suffix}_steps${steps}.log"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  TOKEN_CACHE="${cache}" \
    MAX_PHYSICS_TOKENS=256 \
    EVAL_VARIANT_SUFFIX="${suffix}" \
    bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
      "${mode}" "${checkpoint}" "${num_samples}" "${gpu}" >"${log}" 2>&1 &
  pids+=("$!")
}

evaluate_one \
  b3 \
  artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz \
  _st256_drop20_pg05 \
  0

evaluate_one \
  b4 \
  artifacts/token_caches/click_alarmclock_pred_action4_shuffled_spatiotemporal_physics256_seed42.npz \
  _st256_shuffled_drop20_pg05 \
  1

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
