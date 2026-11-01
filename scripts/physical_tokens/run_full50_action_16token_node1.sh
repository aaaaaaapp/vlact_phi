#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(hostname)" != *node001* ]]; then
  echo "This action token-length comparison is configured for node001" >&2
  exit 2
fi

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
gpu="${ROBOTWIN_ACTION_16TOKEN_GPU:-0}"
input="artifacts/robotwin_full50_study/action_all50_wrap32.npz"
split="docs/robotwin_tokenizer_episode_split.json"
root="artifacts/robotwin_full50_study/action_tokenizers"
baseline="${root}/ema_8x525_decay095_dead100_steps20000/heldout_step20000.json"
output="${root}/ema_16x525_decay095_dead100_steps20000"
checkpoint="${output}/step-00020000.pt"
report="${output}/heldout_step20000.json"

if [[ ! -s "${input}" || ! -s "${split}" || ! -s "${baseline}" ]]; then
  echo "Missing action cache, fixed split, or complete 8-token reference" >&2
  exit 2
fi
if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing nonempty output: ${output}" >&2
  exit 2
fi
free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "${free_mib}" || "${free_mib}" -lt 24000 ]]; then
  echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free; need at least 24000" >&2
  exit 2
fi

mkdir -p "${output}"
echo "$(date -Is) training action EMA K525, 16 tokens per 32-step chunk, GPU${gpu}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  scripts/physical_tokens/train_action_tokenizer.py \
  --input "${input}" --output-dir "${output}" \
  --quantizer ema_vq --num-tokens 16 \
  --codebook-size 525 --fsq-levels 8,8,8 \
  --ema-decay 0.95 --dead-code-steps 100 \
  --split-manifest "${split}" \
  --batch-size 256 --max-steps 20000 --eval-interval 2000 \
  --save-interval 5000 --num-workers 2 --seed 42 --device cuda:0 \
  >"${output}/train.log" 2>&1
if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
  echo "Missing final checkpoint or existing report: ${output}" >&2
  exit 2
fi

echo "$(date -Is) evaluating action EMA K525, 16 tokens per chunk"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  scripts/physical_tokens/evaluate_action_tokenizer.py \
  --checkpoint "${checkpoint}" --input "${input}" --output "${report}" \
  --split-manifest "${split}" \
  --batch-size 256 --num-workers 2 --device cuda:0 \
  >"${output}/heldout_step20000.log" 2>&1
echo "$(date -Is) full50 action 8-vs-16-token comparison complete"
