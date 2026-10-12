#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
input_cache="${PHYSICS_FEATURE_CACHE:-artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_full.npz}"
num_tokens="${PHYSICS_NUM_TOKENS:-4}"
codebook_size="${PHYSICS_CODEBOOK_SIZE:-128}"
max_steps="${PHYSICS_MAX_STEPS:-2000}"
output_dir="${PHYSICS_OUTPUT_DIR:-artifacts/physical_language_tokenizers/ema_${num_tokens}x${codebook_size}_steps${max_steps}}"
nproc="${PHYSICS_NPROC:-8}"

"${python_bin}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" \
  scripts/physical_tokens/train_physical_language_tokenizer.py \
  --input "${input_cache}" \
  --output-dir "${output_dir}" \
  --quantizer ema_vq \
  --num-tokens "${num_tokens}" \
  --codebook-size "${codebook_size}" \
  --max-steps "${max_steps}" \
  --batch-size "${PHYSICS_BATCH_SIZE:-32}"
