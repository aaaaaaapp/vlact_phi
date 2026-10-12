#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz"
action_cache="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
teacher_cache="artifacts/action_caches/robotwin_d19_8tasks_predictable_residual_teacher_wrap32.npz"
sidecar="artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz"
report="artifacts/evaluations/d28_full_feature_cache_validation.json"

"${python_bin}" -u scripts/physical_tokens/validate_d28_full_feature_cache.py \
  --input "${feature_cache}" --action-cache "${action_cache}" --report "${report}"

if [[ -s "${sidecar}" && -s "${sidecar}.manifest.json" ]]; then
  echo "D28 predictable-residual sidecar already complete: ${sidecar}"
else
  if [[ -e "${sidecar}" || -e "${sidecar}.manifest.json" ]]; then
    echo "D28 refused partial sidecar: ${sidecar}" >&2
    exit 2
  fi
  "${python_bin}" -u scripts/physical_tokens/export_predictable_residual_spatiotemporal_sidecar.py \
    --key-template "${feature_cache}" --teacher-cache "${teacher_cache}" --output "${sidecar}"
fi
