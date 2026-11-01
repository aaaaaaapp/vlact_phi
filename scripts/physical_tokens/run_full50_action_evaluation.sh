#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
root="artifacts/robotwin_full50_study/action_tokenizers"
input="artifacts/robotwin_full50_study/action_all50_wrap32.npz"
steps="${ROBOTWIN_ACTION_EVAL_STEPS:-20000}"
gpu="${ROBOTWIN_ACTION_EVAL_GPU:-5}"
parallel_gpus=()
if [[ -n "${ROBOTWIN_ACTION_EVAL_GPUS:-}" ]]; then
  IFS=',' read -r -a parallel_gpus <<< "${ROBOTWIN_ACTION_EVAL_GPUS}"
fi
names=(
  ema_8x512_steps20000
  ema_8x512_decay095_dead100_steps20000
  fsq_8x512_levels4448_steps20000
  fsq_8x512_levels888_steps20000
  fsq_8x512_levels2488_steps20000
  fsq_8x512_levels44442_steps20000
)
if [[ "${ROBOTWIN_ACTION_EVAL_GROUP:-baseline}" == "ifsq" ]]; then
  names=(
    ifsq_8x525_levels3557_steps20000
    fsq_8x525_levels3557_steps20000
    ifsq_8x513_levels33319_steps20000
    fsq_8x513_levels33319_steps20000
    ifsq_8x539_levels7711_steps20000
    fsq_8x539_levels7711_steps20000
    ema_8x525_decay095_dead100_steps20000
  )
elif [[ "${ROBOTWIN_ACTION_EVAL_GROUP:-baseline}" != "baseline" ]]; then
  echo "ROBOTWIN_ACTION_EVAL_GROUP must be baseline or ifsq" >&2
  exit 2
fi
if [[ "${#parallel_gpus[@]}" -gt 0 && "${#parallel_gpus[@]}" -ne "${#names[@]}" ]]; then
  echo "ROBOTWIN_ACTION_EVAL_GPUS must contain one GPU ID per candidate" >&2
  exit 2
fi

for name in "${names[@]}"; do
  checkpoint="${root}/${name}/step-$(printf '%08d' "${steps}").pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint; no partial comparison will be produced: ${checkpoint}" >&2
    exit 2
  fi
done

pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  checkpoint="${root}/${name}/step-$(printf '%08d' "${steps}").pt"
  output="${root}/${name}/heldout_step$(printf '%05d' "${steps}").json"
  if [[ -s "${output}" ]]; then
    echo "Already evaluated: ${output}"
    continue
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing empty or partial evaluation: ${output}" >&2
    exit 2
  fi
  echo "Evaluating ${name}"
  selected_gpu="${gpu}"
  if [[ "${#parallel_gpus[@]}" -gt 0 ]]; then
    selected_gpu="${parallel_gpus[$i]}"
  fi
  CUDA_VISIBLE_DEVICES="${selected_gpu}" "${python_bin}" \
    scripts/physical_tokens/evaluate_action_tokenizer.py \
    --checkpoint "${checkpoint}" --input "${input}" --output "${output}" \
    --split-manifest docs/robotwin_tokenizer_episode_split.json \
    --batch-size 256 --num-workers 2 --device cuda:0 \
    >"${root}/${name}/heldout_step$(printf '%05d' "${steps}").log" 2>&1 &
  pids+=("$!")
  if [[ "${#parallel_gpus[@]}" -eq 0 ]]; then
    wait "${pids[-1]}"
  fi
done
if [[ "${#parallel_gpus[@]}" -gt 0 ]]; then
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  exit "${failed}"
fi
