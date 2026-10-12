#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
input="artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz"
split="artifacts/splits/click_alarmclock_seed42.json"
output="artifacts/action_tokenizer_multitask/ema_4x128_steps5000"
checkpoint="${output}/step-00005000.pt"

if [[ ! -s "${checkpoint}" ]]; then
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D16 refused to overwrite partial output: ${output}" >&2
    exit 2
  fi
  "${python_bin}" -u scripts/physical_tokens/train_action_tokenizer.py \
    --input "${input}" --output-dir "${output}" --split-manifest "${split}" \
    --quantizer ema_vq --num-tokens 4 --codebook-size 128 \
    --hidden-dim 256 --latent-dim 128 --batch-size 512 --num-workers 8 \
    --max-steps 5000 --eval-interval 500 --save-interval 5000 \
    --gripper-dims 12,13 --angular-dims 0,1,2,3,4,5,6,7,8,9,10,11 \
    --velocity-weight 0.25 --gripper-weight 0.25 --wrap-weight 0.25 \
    --device cuda:0
fi

"${python_bin}" -u scripts/physical_tokens/evaluate_action_tokenizer.py \
  --checkpoint "${checkpoint}" --input "${input}" --split-manifest "${split}" \
  --batch-size 512 --num-workers 8 --device cuda:0 \
  --output "${output}/step-00005000-fixed-heldout-evaluation.json"
