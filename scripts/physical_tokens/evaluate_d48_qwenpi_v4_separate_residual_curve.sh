#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
config="examples/Robotwin/train_files/qwenpi_v4_physical_tokens_d48_8tasks.yaml"
steps="${D48_STEPS:-20}"
task="${D48_TASK:-click_alarmclock}"
samples="${D48_EVAL_SAMPLES:-64}"
gpu="${D48_EVAL_GPU:-0}"
run_tag="${D48_RUN_TAG:-d48_separate_attention}"
action_run="qwenpi_v4_${run_tag}_action_only_seed42_steps${steps}"
dual_run="qwenpi_v4_${run_tag}_action_physics_seed42_steps${steps}"
action_checkpoint="results/Checkpoints/${action_run}/checkpoints/steps_${steps}_pytorch_model.pt"
dual_checkpoint="results/Checkpoints/${dual_run}/checkpoints/steps_${steps}_pytorch_model.pt"
aligned_cache="artifacts/token_caches/robotwin_d41_8tasks_predicted_dual_steps8000.npz"
shuffled_cache="artifacts/token_caches/robotwin_d45_8tasks_predicted_dual_episode_shuffled_seed42.npz"
statistics="/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/dataset_statistics.json"
output_dir="artifacts/evaluations/${run_tag}_steps${steps}_${task}_${samples}samples"

for required in "${config}" "${action_checkpoint}" "${dual_checkpoint}" "${aligned_cache}" "${shuffled_cache}" "${statistics}"; do
  [[ -s "${required}" ]] || { echo "missing D48 evaluation input: ${required}" >&2; exit 2; }
done
mkdir -p "${output_dir}"

evaluate_one() {
  local checkpoint="$1"
  local mode="$2"
  local cache="$3"
  local physics_enabled="$4"
  local output="$5"
  local physics_flag="--no-physics-condition-enabled"
  if [[ "${physics_enabled}" == "true" ]]; then
    physics_flag="--physics-condition-enabled"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" scripts/physical_tokens/evaluate_qwenpi_oracle.py \
    --config "${config}" --checkpoint "${checkpoint}" --mode "${mode}" \
    --task-name "${task}" --token-cache "${cache}" --statistics "${statistics}" \
    --num-samples "${samples}" --seed 42 "${physics_flag}" --device cuda:0 \
    --output "${output}"
}

evaluate_one "${action_checkpoint}" e0 "${aligned_cache}" false "${output_dir}/pure_pi.json"
evaluate_one "${action_checkpoint}" b3 "${aligned_cache}" false "${output_dir}/action_only.json"
evaluate_one "${dual_checkpoint}" b3 "${aligned_cache}" false "${output_dir}/dual_hard_disabled.json"
evaluate_one "${dual_checkpoint}" b3 "${aligned_cache}" true "${output_dir}/aligned_physics.json"
evaluate_one "${dual_checkpoint}" b3 "${shuffled_cache}" true "${output_dir}/shuffled_physics.json"

"${python_bin}" scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
  --control "${output_dir}/dual_hard_disabled.json" --enabled "${output_dir}/aligned_physics.json" \
  --output "${output_dir}/aligned_vs_dual_disabled.json"
"${python_bin}" scripts/physical_tokens/compare_qwenpi_aligned_shuffled.py \
  --shuffled "${output_dir}/shuffled_physics.json" --aligned "${output_dir}/aligned_physics.json" \
  --output "${output_dir}/aligned_vs_shuffled.json"
"${python_bin}" scripts/physical_tokens/compare_qwenpi_matched_checkpoints.py \
  --baseline "${output_dir}/pure_pi.json" --candidate "${output_dir}/action_only.json" \
  --label pure_pi_vs_action_only --output "${output_dir}/pure_pi_vs_action_only.json"
"${python_bin}" scripts/physical_tokens/compare_qwenpi_matched_checkpoints.py \
  --baseline "${output_dir}/action_only.json" --candidate "${output_dir}/aligned_physics.json" \
  --label matched_action_only_vs_action_physics \
  --output "${output_dir}/action_only_vs_action_physics.json"
