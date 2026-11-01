#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python}"
input="artifacts/robotwin_full50_study/action_all50_wrap32.npz"
root="artifacts/robotwin_full50_study/action_tokenizers"
IFS=',' read -r -a gpus <<< "${ROBOTWIN_ACTION_IFSQ_GPUS:-0,1,2,3,4,5,6}"
names=(
  ifsq_8x525_levels3557_steps20000
  fsq_8x525_levels3557_steps20000
  ifsq_8x513_levels33319_steps20000
  fsq_8x513_levels33319_steps20000
  ifsq_8x539_levels7711_steps20000
  fsq_8x539_levels7711_steps20000
  ema_8x525_decay095_dead100_steps20000
)
methods=(ifsq fsq ifsq fsq ifsq fsq ema_vq)
levels=(3,5,5,7 3,5,5,7 3,3,3,19 3,3,3,19 7,7,11 7,7,11 8,8,8)
sizes=(525 525 513 513 539 539 525)

if [[ "${#gpus[@]}" -ne "${#names[@]}" || ! -s "${input}" ]]; then
  echo "Expected seven GPU IDs and the complete 50-task action cache" >&2
  exit 2
fi
for name in "${names[@]}"; do
  output="${root}/${name}"
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output; review before restarting: ${output}" >&2
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
    --codebook-size "${sizes[$i]}" --fsq-levels "${levels[$i]}" \
    --ema-decay 0.95 --dead-code-steps 100 \
    --split-manifest docs/robotwin_tokenizer_episode_split.json \
    --batch-size 256 --max-steps 20000 --eval-interval 2000 \
    --save-interval 5000 --num-workers 2 --device cuda:0 \
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
exit "${failed}"
