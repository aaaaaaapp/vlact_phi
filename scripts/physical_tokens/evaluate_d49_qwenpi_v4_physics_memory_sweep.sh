#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
config="examples/Robotwin/train_files/qwenpi_v4_physical_tokens_d48_8tasks.yaml"
steps="${D49_STEPS:-20}"
task="${D49_TASK:-pick_dual_bottles}"
samples="${D49_EVAL_SAMPLES:-64}"
gpu="${D49_EVAL_GPU:-0}"
memory="${D49_MEMORY:-16}"
aligned_cache="artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz"
shuffled_cache="artifacts/token_caches/robotwin_d45_8tasks_predicted_dual_episode_shuffled_seed42.npz"
statistics="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/dataset_statistics.json"

case "${memory}" in
  8) group_size=32 ;;
  16) group_size=16 ;;
  32) group_size=8 ;;
  *) echo "D49_MEMORY must be one of 8, 16, or 32; got ${memory}" >&2; exit 2 ;;
esac

if [[ "${memory}" == "16" ]]; then
  checkpoint="results/Checkpoints/qwenpi_v4_d48_separate_attention_action_physics_seed42_steps${steps}/checkpoints/steps_${steps}_pytorch_model.pt"
else
  checkpoint="results/Checkpoints/qwenpi_v4_d49_physics_memory${memory}_seed42_steps${steps}/checkpoints/steps_${steps}_pytorch_model.pt"
fi
output_dir="artifacts/evaluations/d49_physics_memory${memory}_steps${steps}_${task}_${samples}samples"

for required in "${config}" "${checkpoint}" "${aligned_cache}" "${shuffled_cache}" "${statistics}"; do
  [[ -s "${required}" ]] || { echo "missing D49 evaluation input: ${required}" >&2; exit 2; }
done
mkdir -p "${output_dir}"

evaluate_one() {
  local cache="$1"
  local physics_enabled="$2"
  local output="$3"
  local physics_flag="--no-physics-condition-enabled"
  if [[ "${physics_enabled}" == "true" ]]; then
    physics_flag="--physics-condition-enabled"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" scripts/physical_tokens/evaluate_qwenpi_oracle.py \
    --config "${config}" --checkpoint "${checkpoint}" --mode b3 \
    --task-name "${task}" --token-cache "${cache}" --statistics "${statistics}" \
    --physics-group-size "${group_size}" --num-samples "${samples}" --seed 42 \
    "${physics_flag}" --device cuda:0 --output "${output}"
}

evaluate_one "${aligned_cache}" false "${output_dir}/dual_hard_disabled.json"
evaluate_one "${aligned_cache}" true "${output_dir}/aligned_physics.json"
evaluate_one "${shuffled_cache}" true "${output_dir}/shuffled_physics.json"

"${python_bin}" scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
  --control "${output_dir}/dual_hard_disabled.json" --enabled "${output_dir}/aligned_physics.json" \
  --output "${output_dir}/aligned_vs_dual_disabled.json"
"${python_bin}" scripts/physical_tokens/compare_qwenpi_aligned_shuffled.py \
  --shuffled "${output_dir}/shuffled_physics.json" --aligned "${output_dir}/aligned_physics.json" \
  --output "${output_dir}/aligned_vs_shuffled.json"
