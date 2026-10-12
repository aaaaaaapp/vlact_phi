#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="${P64_FEATURE_CACHE:-artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz}"
template="${P64_ACTION_TEMPLATE:-artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_actions17x14.npz}"
teacher="${P64_ACTION_TEACHER:-artifacts/action_caches/robotwin_d19_8tasks_predictable_residual_teacher_wrap32.npz}"
sidecar="${P64_ACTION_SIDECAR:-artifacts/future_feature_caches/robotwin_d20_8tasks_8episodes_predictable_residuals17x14.npz}"
output="${P64_OUTPUT:-artifacts/physical_language_tokenizers/p64_phi_residual_ema_64x128_steps20000}"
max_steps="${P64_MAX_STEPS:-20000}"

for required in "${feature_cache}" "${template}" "${teacher}"; do
  [[ -s "${required}" ]] || { echo "missing P64 input: ${required}" >&2; exit 2; }
done
if [[ ! -s "${sidecar}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_predictable_residual_spatiotemporal_sidecar.py \
    --key-template "${template}" --teacher-cache "${teacher}" --output "${sidecar}"
fi
if [[ -s "${output}/step-$(printf '%08d' "${max_steps}").pt" ]]; then
  echo "P64 checkpoint already complete: ${output}/step-$(printf '%08d' "${max_steps}").pt"
  exit 0
fi
if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite partial P64 output: ${output}" >&2
  exit 2
fi

SPATIOTEMPORAL_NPROC="${P64_NPROC:-1}" \
SPATIOTEMPORAL_BATCH_SIZE="${P64_BATCH_SIZE:-32}" \
SPATIOTEMPORAL_FEATURE_CACHE="${feature_cache}" \
SPATIOTEMPORAL_ACTION_TARGET_CACHE="${sidecar}" \
SPATIOTEMPORAL_ACTION_AUX_WEIGHT="${P64_ACTION_AUX_WEIGHT:-0.25}" \
SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT="${P64_ACTION_VELOCITY_WEIGHT:-0.25}" \
SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT="${P64_ACTION_GRIPPER_WEIGHT:-0.1}" \
SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK="${P64_VALIDATION_EPISODES_PER_TASK:-2}" \
SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK="${P64_MAX_TRAIN_CLIPS_PER_TASK:-300}" \
SPATIOTEMPORAL_MAX_STEPS="${max_steps}" \
SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
SPATIOTEMPORAL_QUERIES_PER_TRANSITION=4 \
bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
