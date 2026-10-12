#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
steps="${D53_STEPS:-20}"
samples="${D53_EVAL_SAMPLES:-64}"
config="results/Checkpoints/qwenpi_v4_d53_soft_joint_low_lr_seed42_steps${steps}/config.yaml"
checkpoint="results/Checkpoints/qwenpi_v4_d53_soft_joint_low_lr_seed42_steps${steps}/checkpoints/steps_${steps}_pytorch_model.pt"
token_cache="artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz"
statistics="results/Checkpoints/qwenpi_v4_d53_soft_joint_low_lr_seed42_steps${steps}/dataset_statistics.json"
tasks=(
  click_alarmclock
  beat_block_hammer
  open_microwave
  shake_bottle
  move_playingcard_away
  pick_dual_bottles
  handover_block
  place_object_basket
)

for required in "${config}" "${checkpoint}" "${token_cache}" "${statistics}"; do
  [[ -s "${required}" ]] || { echo "missing D53 evaluation input: ${required}" >&2; exit 2; }
done

pids=()
for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  gpu="$((index % 8))"
  output_dir="artifacts/evaluations/d53_soft_joint_steps${steps}_${task}_${samples}samples"
  output="${output_dir}/dual_soft_t1.json"
  mkdir -p "${output_dir}"
  if [[ -s "${output}" ]]; then
    echo "Reusing ${output}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    scripts/physical_tokens/evaluate_qwenpi_oracle.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --mode b3 \
    --task-name "${task}" \
    --token-cache "${token_cache}" \
    --statistics "${statistics}" \
    --joint-condition-source soft \
    --joint-soft-temperature 1.0 \
    --physics-condition-enabled \
    --num-samples "${samples}" \
    --seed 42 \
    --device cuda:0 \
    --output "${output}" \
    > "${output_dir}/eval.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || { echo "one or more D53 evaluations failed" >&2; exit 1; }

for task in "${tasks[@]}"; do
  d52="artifacts/evaluations/d52_frozen_policy_steps500_${task}_${samples}samples/dual_soft_t1.json"
  d53="artifacts/evaluations/d53_soft_joint_steps${steps}_${task}_${samples}samples/dual_soft_t1.json"
  comparison="artifacts/evaluations/d53_soft_joint_steps${steps}_${task}_${samples}samples/d52_vs_d53_soft_t1.json"
  [[ -s "${d52}" ]] || { echo "missing D52 reference: ${d52}" >&2; exit 2; }
  "${python_bin}" scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
    --control "${d52}" --enabled "${d53}" --allow-checkpoint-change \
    --output "${comparison}"
done
