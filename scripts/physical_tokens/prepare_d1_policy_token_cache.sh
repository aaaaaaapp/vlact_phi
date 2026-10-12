#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
d1_output="${D1_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d1_8tasks_ema_256x128_steps1000}"
gate_report="${D1_COMPARISON_REPORT:-${d1_output}/comparison_d0_vs_d1.json}"
structure_report="${D1_STRUCTURE_REPORT:-${d1_output}/token_structure_per_task.json}"

"${python_bin}" scripts/physical_tokens/require_d1_representation_gate.py "${gate_report}"
"${python_bin}" scripts/physical_tokens/require_physics_token_structure_gate.py "${structure_report}"

SPATIOTEMPORAL_FEATURE_CACHE="${D1_POLICY_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_spatial4x4_17frames.npz}" \
SPATIOTEMPORAL_TOKENIZER_CHECKPOINT="${D1_TOKENIZER_CHECKPOINT:-${d1_output}/step-00001000.pt}" \
SPATIOTEMPORAL_PHYSICS_CACHE="${D1_PHYSICS_CACHE:-artifacts/token_caches/click_alarmclock_oracle_physics_spatiotemporal_d1_ema_256x128_step1000.npz}" \
SPATIOTEMPORAL_ALIGNED_CACHE="${D1_ALIGNED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_d1_physics256.npz}" \
SPATIOTEMPORAL_SHUFFLED_CACHE="${D1_SHUFFLED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_shuffled_spatiotemporal_d1_physics256_seed42.npz}" \
bash scripts/physical_tokens/prepare_spatiotemporal_physics_token_cache.sh
