#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
input=artifacts/robotwin_full50_study/action_all50_wrap32.npz
split=docs/robotwin_tokenizer_episode_split.json
root=artifacts/robotwin_full50_study/action_tokenizers
output8="${root}/ema_8x525_decay095_dead100_bs128_steps20000"
output32="${root}/ema_32x525_decay095_dead100_bs128_steps20000"
gpu8="${ROBOTWIN_ACTION_8TOKEN_GPU:-4}"
gpu32="${ROBOTWIN_ACTION_32TOKEN_GPU:-7}"

if [[ ! -s "${input}" || ! -s "${split}" ]]; then
  echo "Missing full50 action cache or episode split" >&2
  exit 2
fi
if [[ "${gpu8}" == "${gpu32}" ]]; then
  echo "Select two distinct GPUs for the matched-length comparison" >&2
  exit 2
fi
for output in "${output8}" "${output32}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to reuse existing experiment directory: ${output}" >&2
    exit 2
  fi
done
for gpu in "${gpu8}" "${gpu32}"; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 16000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free; need at least 16000" >&2
    exit 2
  fi
done

train_one() {
  local count="$1" gpu="$2" output="$3"
  echo "$(date -Is) start 32-step -> ${count}-token EMA-VQ K525 on GPU${gpu}, global batch 128"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/train_action_tokenizer.py \
    --input "${input}" --output-dir "${output}" \
    --quantizer ema_vq --num-tokens "${count}" \
    --codebook-size 525 --fsq-levels 8,8,8 \
    --ema-decay 0.95 --dead-code-steps 100 \
    --split-manifest "${split}" \
    --batch-size 128 --max-steps 20000 --eval-interval 2000 \
    --save-interval 5000 --num-workers 2 --seed 42 --device cuda:0 \
    >"${output}/train.log" 2>&1
}

mkdir -p "${output8}" "${output32}"
train_one 8 "${gpu8}" "${output8}" &
pid8=$!
train_one 32 "${gpu32}" "${output32}" &
pid32=$!
failed=0
wait "${pid8}" || failed=1
wait "${pid32}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one tokenizer training run failed; inspect the two train.log files" >&2
  exit 1
fi

evaluate_one() {
  local gpu="$1" output="$2"
  local checkpoint="${output}/step-00020000.pt"
  local report="${output}/heldout_step20000.json"
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing heldout report: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_action_tokenizer.py \
    --checkpoint "${checkpoint}" --input "${input}" --output "${report}" \
    --split-manifest "${split}" \
    --batch-size 256 --num-workers 2 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1
}

evaluate_one "${gpu8}" "${output8}" &
eval_pid8=$!
evaluate_one "${gpu32}" "${output32}" &
eval_pid32=$!
failed=0
wait "${eval_pid8}" || failed=1
wait "${eval_pid32}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one heldout evaluation failed; inspect the evaluation logs" >&2
  exit 1
fi
echo "$(date -Is) matched 8-token vs 32-token action tokenizer comparison complete"
