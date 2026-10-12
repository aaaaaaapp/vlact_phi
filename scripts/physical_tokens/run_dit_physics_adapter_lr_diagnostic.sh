#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

steps="${1:-20}"
batch_size="${2:-8}"
adapter_lr="${ADAPTER_LEARNING_RATE:-2.5e-5}"
token_cache="${D0_ALIGNED_CACHE:-artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_d0_spatial_aligned_ema_physics256.npz}"
action_init="${ACTION_INIT_CHECKPOINT:-results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt}"

case "${adapter_lr}" in
  2.5e-5|2.5E-5) ;;
  *)
    echo "This preregistered diagnostic requires ADAPTER_LEARNING_RATE=2.5e-5" >&2
    exit 2
    ;;
esac
if [[ "${steps}" != "20" ]]; then
  echo "This first-stage diagnostic requires exactly 20 steps" >&2
  exit 2
fi
for required in "${token_cache}" "${action_init}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Required LR-diagnostic input is missing: ${required}" >&2
    exit 2
  fi
done

variant_suffix="_d0_spatial_st256_lr2p5e5_8gpu_dit_first_adapteronly"
TOKEN_CACHE="${token_cache}" \
MAX_ACTION_TOKENS=4 \
MAX_PHYSICS_TOKENS=256 \
PHYSICS_FUSION_MODE=dit_multilayer_cross_attention \
PHYSICS_FUSION_HEADS=8 \
PHYSICS_CONDITION_DROPOUT=0.2 \
PHYSICS_CONDITION_ENABLED=true \
DIT_INJECTION_LAYERS=first \
DIT_ATTENTION_DROPOUT=0.0 \
DIT_FREEZE_BASE_ACTION_MODEL=true \
ACTION_INIT_CHECKPOINT="${action_init}" \
FREEZE_MODULES=qwen_vl_interface,discrete_conditioner \
ACTION_MODEL_LEARNING_RATE="${adapter_lr}" \
VARIANT_SUFFIX="${variant_suffix}" \
NUM_PROCESSES="${NUM_PROCESSES:-8}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29684}" \
WANDB_MODE=disabled \
bash scripts/physical_tokens/run_qwenpi_oracle_ablation.sh b3 "${steps}" "${batch_size}"

echo "Training complete. Run the same-checkpoint paired evaluator before any longer run."
