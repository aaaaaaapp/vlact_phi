#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${POLICY_SWEEP_STEPS:-500}"
num_samples="${POLICY_SWEEP_EVAL_SAMPLES:-380}"
log_dir="artifacts/logs/physics_token_count_policy_eval"
mkdir -p "${log_dir}"
pids=()

evaluate_one() {
  local mode="$1"
  local count="$2"
  local cache="$3"
  local suffix="$4"
  local gpu="$5"
  local run="qwenpi_physical_tokens_click_alarmclock_${mode}${suffix}_seed42_steps${steps}"
  local checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"
  local log="${log_dir}/${mode}${suffix}_steps${steps}.log"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  echo "starting evaluation ${mode}${suffix}: physics_tokens=${count} gpu=${gpu}"
  TOKEN_CACHE="${cache}" \
    MAX_PHYSICS_TOKENS="${count}" \
    EVAL_VARIANT_SUFFIX="${suffix}" \
    bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
      "${mode}" "${checkpoint}" "${num_samples}" "${gpu}" >"${log}" 2>&1 &
  pids+=("$!")
}

evaluate_one \
  e2 \
  4 \
  artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz \
  _count_sweep \
  0

gpu=1
for count in 16 64 256; do
  evaluate_one \
    b3 \
    "${count}" \
    "artifacts/token_caches/click_alarmclock_pred_action4_oracle_physics${count}.npz" \
    "_p${count}_drop20_pg05" \
    "${gpu}"
  gpu=$((gpu + 1))
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
