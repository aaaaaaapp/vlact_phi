#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

selection="${1:-first}"
steps="${2:-20}"
num_samples="${3:-380}"
gpu="${4:-0}"
case "${selection}" in
  first|early-4|all) ;;
  *) echo "Usage: $0 {first|early-4|all} [steps] [num_samples] [gpu]" >&2; exit 2 ;;
esac

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
d1_output="${D1_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d1_8tasks_ema_256x128_steps1000}"
gate_report="${D1_COMPARISON_REPORT:-${d1_output}/comparison_d0_vs_d1.json}"
structure_report="${D1_STRUCTURE_REPORT:-${d1_output}/token_structure_per_task.json}"
token_cache="${D1_ALIGNED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_d1_physics256.npz}"
action_init="${ACTION_INIT_CHECKPOINT:-results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt}"
experiment_tag="${PHYSICS_EXPERIMENT_TAG:-d1_st256}"
if [[ ! "${experiment_tag}" =~ ^[a-zA-Z0-9_]+$ ]]; then
  echo "PHYSICS_EXPERIMENT_TAG must contain only letters, digits, and underscores" >&2
  exit 2
fi
safe_selection="${selection//-/_}"
suffix="_${experiment_tag}_dit_${safe_selection}_adapteronly"
run="qwenpi_physical_tokens_click_alarmclock_b3${suffix}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"
output_dir="artifacts/evaluations/${run}_${num_samples}samples"

"${python_bin}" scripts/physical_tokens/require_d1_representation_gate.py "${gate_report}"
"${python_bin}" scripts/physical_tokens/require_physics_token_structure_gate.py "${structure_report}"
for required in "${token_cache}" "${action_init}" "${checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required paired-evaluation input is missing: ${required}" >&2
    exit 2
  fi
done
mkdir -p "${output_dir}"

common_env=(
  TOKEN_CACHE="${token_cache}"
  MAX_ACTION_TOKENS=4
  MAX_PHYSICS_TOKENS=256
  PHYSICS_FUSION_MODE=dit_multilayer_cross_attention
  PHYSICS_FUSION_HEADS=8
  DIT_INJECTION_LAYERS="${selection}"
  DIT_ATTENTION_DROPOUT=0.0
  DIT_ACTION_TOKENS_ONLY="${DIT_ACTION_TOKENS_ONLY:-false}"
  DIT_PHYSICS_ADAPTER_TYPE="${DIT_PHYSICS_ADAPTER_TYPE:-cross_attention}"
  DIT_PHYSICS_MEMORY_POOLING="${DIT_PHYSICS_MEMORY_POOLING:-none}"
  PHYSICS_TOKENS_PER_TRANSITION="${PHYSICS_TOKENS_PER_TRANSITION:-16}"
  PHYSICS_CONDITION_DROPOUT=0.2
  ACTION_INIT_CHECKPOINT="${action_init}"
)

env "${common_env[@]}" \
  PHYSICS_CONDITION_ENABLED=true \
  EVAL_OUTPUT="${output_dir}/enabled.json" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
    b3 "${checkpoint}" "${num_samples}" "${gpu}"

env "${common_env[@]}" \
  PHYSICS_CONDITION_ENABLED=false \
  EVAL_OUTPUT="${output_dir}/hard_disabled.json" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
    b3 "${checkpoint}" "${num_samples}" "${gpu}"

"${python_bin}" scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
  --control "${output_dir}/hard_disabled.json" \
  --enabled "${output_dir}/enabled.json" \
  --output "${output_dir}/comparison.json"
