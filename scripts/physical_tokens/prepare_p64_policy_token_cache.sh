#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="${P64_FEATURE_CACHE:-artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz}"
checkpoint="${P64_TOKENIZER_CHECKPOINT:-artifacts/physical_language_tokenizers/p64_phi_residual_ema_64x128_steps20000/step-00020000.pt}"
action_cache="${P64_ACTION_CACHE:-artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz}"
physics_cache="${P64_PHYSICS_CACHE:-artifacts/token_caches/robotwin_p64_physics64_oracle_step20000.npz}"
output="${P64_POLICY_CACHE:-artifacts/token_caches/robotwin_p64_predicted_action4_physics64_step20000.npz}"

for required in "${feature_cache}" "${checkpoint}" "${action_cache}"; do
  [[ -s "${required}" ]] || { echo "missing P64 cache input: ${required}" >&2; exit 2; }
done
if [[ ! -s "${physics_cache}" ]]; then
  "${python_bin}" scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
    --input "${feature_cache}" --checkpoint "${checkpoint}" \
    --output "${physics_cache}" --batch-size 256 --device "${P64_EXPORT_DEVICE:-cuda:0}"
fi
"${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
  --action-cache "${action_cache}" --physics-cache "${physics_cache}" \
  --output "${output}"
