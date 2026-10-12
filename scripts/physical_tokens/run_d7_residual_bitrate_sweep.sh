#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
residual_cache="artifacts/action_caches/click_alarmclock_pred_action4_residual_wrap32.npz"
action_cache="artifacts/action_caches/click_alarmclock_clean_wrap32.npz"
predicted_actions="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
action_tokenizer="artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt"
variants=("4x32" "4x64" "8x64")
tokens=(4 4 8)
codes=(32 64 64)
gpus=(0 1 2)
pids=()

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  output="artifacts/action_tokenizer_residual/ema_${variant}_steps5000"
  checkpoint="${output}/step-00005000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "Reusing completed D7 tokenizer: ${checkpoint}"
    continue
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "D7 refused to overwrite partial output: ${output}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "${python_bin}" \
    scripts/physical_tokens/train_action_tokenizer.py \
    --input "${residual_cache}" --output-dir "${output}" \
    --quantizer ema_vq --num-tokens "${tokens[$index]}" --codebook-size "${codes[$index]}" \
    --hidden-dim 256 --latent-dim 128 --batch-size 256 --num-workers 4 \
    --max-steps 5000 --eval-interval 500 --save-interval 5000 \
    --gripper-dims "" --angular-dims 0,1,2,3,4,5,6,7,8,9,10,11 \
    --velocity-weight 0.25 --gripper-weight 0 --wrap-weight 0.25 \
    --device cuda:0 >"${output}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  output="artifacts/action_tokenizer_residual/ema_${variant}_steps5000"
  checkpoint="${output}/step-00005000.pt"
  token_cache="artifacts/token_caches/click_alarmclock_oracle_action_residual_ema_${variant}_step5000.npz"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/physical_tokens/export_action_token_ids.py \
    --input "${residual_cache}" --checkpoint "${checkpoint}" --output "${token_cache}" \
    --device cuda:0
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/physical_tokens/evaluate_oracle_residual_tokenizer.py \
    --action-cache "${action_cache}" --predicted-action-token-cache "${predicted_actions}" \
    --residual-token-cache "${token_cache}" --action-tokenizer-checkpoint "${action_tokenizer}" \
    --residual-tokenizer-checkpoint "${checkpoint}" \
    --output "artifacts/evaluations/d7_residual_${variant}_oracle_utility.json" --device cuda:0
done
