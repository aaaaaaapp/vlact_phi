#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

selection="${1:-first}"
steps="${2:-20}"
batch_size="${3:-8}"
case "${selection}" in
  first|early-4|all) ;;
  *) echo "Usage: $0 {first|early-4|all} [steps] [batch_size]" >&2; exit 2 ;;
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

"${python_bin}" scripts/physical_tokens/require_d1_representation_gate.py "${gate_report}"
"${python_bin}" scripts/physical_tokens/require_physics_token_structure_gate.py "${structure_report}"
for required in "${token_cache}" "${action_init}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required adapter-smoke input is missing: ${required}" >&2
    exit 2
  fi
done

safe_selection="${selection//-/_}"
variant_suffix="_${experiment_tag}_dit_${safe_selection}_adapteronly"
run="qwenpi_physical_tokens_click_alarmclock_b3${variant_suffix}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"
run_dir="results/Checkpoints/${run}"
if [[ -s "${checkpoint}" ]]; then
  echo "Reusing completed adapter smoke checkpoint: ${checkpoint}"
  exit 0
fi
if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "Adapter smoke refused to overwrite partial output directory: ${run_dir}" >&2
  exit 2
fi

TOKEN_CACHE="${token_cache}" \
MAX_ACTION_TOKENS=4 \
MAX_PHYSICS_TOKENS=256 \
PHYSICS_FUSION_MODE=dit_multilayer_cross_attention \
PHYSICS_FUSION_HEADS=8 \
PHYSICS_CONDITION_DROPOUT=0.2 \
PHYSICS_CONDITION_ENABLED=true \
DIT_INJECTION_LAYERS="${selection}" \
DIT_ATTENTION_DROPOUT=0.0 \
DIT_FREEZE_BASE_ACTION_MODEL=true \
ACTION_INIT_CHECKPOINT="${action_init}" \
FREEZE_MODULES=qwen_vl_interface,discrete_conditioner \
VARIANT_SUFFIX="${variant_suffix}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29680}" \
WANDB_MODE=disabled \
bash scripts/physical_tokens/run_qwenpi_oracle_ablation.sh \
  b3 "${steps}" "${batch_size}"
