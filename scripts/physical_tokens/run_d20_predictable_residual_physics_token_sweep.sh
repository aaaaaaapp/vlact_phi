#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
feature_cache="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz"
template="artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_actions17x14.npz"
teacher="artifacts/action_caches/robotwin_d19_8tasks_predictable_residual_teacher_wrap32.npz"
sidecar="artifacts/future_feature_caches/robotwin_d20_8tasks_8episodes_predictable_residuals17x14.npz"

if [[ ! -s "${sidecar}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/export_predictable_residual_spatiotemporal_sidecar.py \
    --key-template "${template}" --teacher-cache "${teacher}" --output "${sidecar}"
fi

pids=()
queries=(1 4 16)
gpus=(1 4 5)
for index in "${!queries[@]}"; do
  query_count="${queries[$index]}"
  total_tokens=$((16 * query_count))
  output="artifacts/physical_language_tokenizers/d20_predictable_residual_ema_${total_tokens}x128_steps1000"
  checkpoint="${output}/step-00001000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "D20 ${total_tokens}-token checkpoint already complete: ${checkpoint}"
    continue
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D20 refused partial output directory: ${output}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
  SPATIOTEMPORAL_NPROC=1 SPATIOTEMPORAL_BATCH_SIZE=32 \
  SPATIOTEMPORAL_FEATURE_CACHE="${feature_cache}" \
  SPATIOTEMPORAL_ACTION_TARGET_CACHE="${sidecar}" \
  SPATIOTEMPORAL_ACTION_AUX_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_VELOCITY_WEIGHT=0.25 \
  SPATIOTEMPORAL_ACTION_GRIPPER_WEIGHT=0.1 \
  SPATIOTEMPORAL_ACTION_AUX_TARGET_MODE=action_token_residual \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=300 \
  SPATIOTEMPORAL_MAX_STEPS=1000 \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  SPATIOTEMPORAL_QUERIES_PER_TRANSITION="${query_count}" \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh \
    >"${output}.log" 2>&1 &
  pids+=("$!")
  echo "started D20 ${total_tokens}-token EMA run on GPU ${gpus[$index]}, pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
