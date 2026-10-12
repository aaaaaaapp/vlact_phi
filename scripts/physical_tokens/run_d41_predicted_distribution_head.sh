#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
config="artifacts/configs/qwen_dual_token_probe_d31_full_diversity_selected_steps8000.yaml"
checkpoint="results/Checkpoints/qwen_dual_token_probe_d31_full_diversity_k256_steps8000/checkpoints/steps_8000_pytorch_model.pt"
source_cache="artifacts/token_caches/robotwin_d30_8tasks_dual_action4_physics256_selected_ema.npz"
part_dir="artifacts/token_caches/d41_8tasks_predicted_dual_parts"
log_dir="artifacts/logs/d41_predicted_dual_export"
predicted_cache="artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz"
target_cache="artifacts/action_caches/robotwin_d41_dual_predicted_actual_residuals17x14.npz"
head="artifacts/physical_language_tokenizers/d41_predicted_distribution_head_steps3000.pt"
tasks=(
  click_alarmclock beat_block_hammer open_microwave shake_bottle
  move_playingcard_away pick_dual_bottles handover_block place_object_basket
)

mkdir -p "${part_dir}" "${log_dir}"
pids=()
parts=()
for gpu in "${!tasks[@]}"; do
  task="${tasks[$gpu]}"
  part="${part_dir}/${task}_predicted_dual.npz"
  parts+=("${part}")
  if [[ -s "${part}" && -s "${part}.manifest.json" ]]; then continue; fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  D17_EXPORT_CPU_THREADS=4 CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    scripts/physical_tokens/export_multitask_predicted_action_tokens.py \
    --config "${config}" --checkpoint "${checkpoint}" \
    --source-token-cache "${source_cache}" --data-name "${task}" \
    --output "${part}" --batch-size "${D41_EXPORT_BATCH_SIZE:-16}" \
    --device cuda:0 --verify-first-batch >"${log_dir}/${task}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if [[ "${failed}" -ne 0 ]]; then exit 1; fi

if [[ ! -s "${predicted_cache}" ]]; then
  "${python_bin}" -u scripts/physical_tokens/merge_multitask_predicted_token_caches.py \
    --inputs "${parts[@]}" --task-names "${tasks[@]}" \
    --oracle-cache "${source_cache}" --output "${predicted_cache}"
fi
if [[ ! -s "${target_cache}" ]]; then
  CUDA_VISIBLE_DEVICES="${D41_HEAD_GPU:-0}" "${python_bin}" -u \
    scripts/physical_tokens/export_actual_action_residual_targets.py \
    --action-cache artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz \
    --predicted-action-token-cache "${predicted_cache}" \
    --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt \
    --output "${target_cache}" --batch-size 2048 --device cuda:0
fi
if [[ ! -s "${head}" || ! -s "${head}.json" ]]; then
  CUDA_VISIBLE_DEVICES="${D41_HEAD_GPU:-0}" "${python_bin}" -u \
    scripts/physical_tokens/train_action_conditioned_physics_residual_probe.py \
    --physics-cache "${predicted_cache}" --action-token-cache "${predicted_cache}" \
    --action-target-cache "${target_cache}" \
    --feature-cache artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz \
    --physics-checkpoint artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt \
    --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt \
    --output "${head}" --validation-episodes 0,10 \
    --excluded-episodes 5,20,25,39,49 --steps 3000 --batch-size 512 \
    --eval-interval 100 --device cuda:0
fi

D31B_OUTPUT_TAG=d41 D31_EVAL_STEPS=8000 D31_CONDITIONED_CHECKPOINT="${head}" \
  bash scripts/physical_tokens/run_d31b_expanded_evaluation.sh
