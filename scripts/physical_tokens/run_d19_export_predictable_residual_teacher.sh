#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
gate="artifacts/evaluations/d18_predicted_action_current_residual_gate.json"
output="artifacts/action_caches/robotwin_d19_8tasks_predictable_residual_teacher_wrap32.npz"

if [[ ! -s "${gate}" ]]; then
  echo "missing D18 gate report: ${gate}" >&2
  exit 1
fi
if [[ -e "${output}" || -e "${output}.manifest.json" ]]; then
  echo "D19 refused to overwrite teacher cache: ${output}" >&2
  exit 2
fi
"${python_bin}" -u scripts/physical_tokens/export_observation_predictable_residual_teacher.py \
  --visual-feature-cache artifacts/future_feature_caches/robotwin_d15_8tasks_all_episodes_current1x1.npz \
  --residual-action-cache artifacts/action_caches/robotwin_d18_8tasks_predicted_action_residual_wrap32.npz \
  --action-token-cache artifacts/token_caches/robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --gate-report "${gate}" --projection-dim 512 --folds 5 --seed 42 \
  --device cuda:0 --output "${output}"
