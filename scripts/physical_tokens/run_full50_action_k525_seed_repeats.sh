#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
input="artifacts/robotwin_full50_study/action_all50_wrap32.npz"
root="artifacts/robotwin_full50_study/action_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_ACTION_REPEAT_GPUS:-0,1,2,3,4,5}"
names=(ema_8x525_decay095_dead100_seed43_steps20000 fsq_8x525_levels3557_seed43_steps20000 ifsq_8x525_levels3557_seed43_steps20000 ema_8x525_decay095_dead100_seed44_steps20000 fsq_8x525_levels3557_seed44_steps20000 ifsq_8x525_levels3557_seed44_steps20000)
methods=(ema_vq fsq ifsq ema_vq fsq ifsq)
seeds=(43 43 43 44 44 44)

if [[ "${#gpus[@]}" -ne "${#names[@]}" || ! -s "${input}" ]]; then
  echo "Expected six GPU IDs and the full 50-task action cache" >&2
  exit 2
fi
for name in "${names[@]}"; do
  output="${root}/${name}"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty repeat output: ${output}" >&2
    exit 2
  fi
done

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/train_action_tokenizer.py \
    --input "${input}" --output-dir "${output}" \
    --quantizer "${methods[$i]}" --num-tokens 8 \
    --codebook-size 525 --fsq-levels 3,5,5,7 \
    --ema-decay 0.95 --dead-code-steps 100 \
    --split-manifest docs/robotwin_tokenizer_episode_split.json \
    --batch-size 256 --max-steps 20000 --eval-interval 2000 \
    --save-interval 5000 --num-workers 2 --seed "${seeds[$i]}" --device cuda:0 \
    >"${output}/train.log" 2>&1 &
  pids+=("$!")
  echo "started ${names[$i]} on GPU ${gpus[$i]}: pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one repeat training run failed; inspect train.log" >&2
  exit 1
fi

pids=()
for i in "${!names[@]}"; do
  output="${root}/${names[$i]}"
  checkpoint="${output}/step-00020000.pt"
  report="${output}/heldout_step20000.json"
  if [[ ! -s "${checkpoint}" || -e "${report}" ]]; then
    echo "Missing checkpoint or existing report for ${names[$i]}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_action_tokenizer.py \
    --checkpoint "${checkpoint}" --input "${input}" --output "${report}" \
    --split-manifest docs/robotwin_tokenizer_episode_split.json \
    --batch-size 256 --num-workers 2 --device cuda:0 \
    >"${output}/heldout_step20000.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
