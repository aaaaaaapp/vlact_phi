#!/usr/bin/env bash
# Four A100-80GB: 4 x 32 x 1 = global batch 128, with one LR step per update.
# Gradient checkpointing is needed at microbatch 32 on an 80GB card.
set -euo pipefail
repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:/mnt/jfzn/liujn/code/VLAct-main/.acp_deps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${repo}"
mkdir -p artifacts/robotwin_full50_study/acp_logs

export WANDB_MODE=offline
export DUAL_AR_ACTION_TOKEN_COUNT=32
export DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml
export DUAL_AR_LORA=1
export DUAL_AR_LORA_RANK=32
export DUAL_AR_LORA_ALPHA=32
# The warm-start trainable checkpoint already contains learned physics rows.
export DUAL_AR_UNIFORM_PHYSICS_INIT=0
export DUAL_AR_NUM_PROCESSES=4
export DUAL_AR_MICRO_BATCH_SIZE=32
export DUAL_AR_GRAD_ACCUM_STEPS=1
export DUAL_AR_GLOBAL_BATCH_SIZE=128
export DUAL_AR_GRADIENT_CHECKPOINTING="${DUAL_AR_GRADIENT_CHECKPOINTING:-1}"
export DUAL_AR_MAX_STEPS="${DUAL_AR_MAX_STEPS:-40000}"
export DUAL_AR_SAVE_INTERVAL="${DUAL_AR_SAVE_INTERVAL:-5000}"
export DUAL_AR_SAVE_FULL_STATE=1
export DUAL_AR_MAIN_PROCESS_PORT="${DUAL_AR_MAIN_PROCESS_PORT:-29691}"
export DUAL_AR_RUN_TAG="${DUAL_AR_RUN_TAG:-robotwin_ft_acp_4gpu_bs128_r32_32action_m32_ga1_fixedlr_v1}"
export DUAL_AR_LOG_FILE="${DUAL_AR_LOG_FILE:-artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_4gpu_m32_ga1_bs128_r32_40k_fixedlr_v1.log}"

if [[ -z "${DUAL_AR_FULL_STATE_RESUME_DIR:-}" ]]; then
  source_step="${DUAL_AR_INITIAL_COMPLETED_STEPS:-10000}"
  source_run=results/Checkpoints/qwen_dual_ar_32a64p_full50_seed42_steps40000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_1gpu_bs128_r32_32action_m16_gc0_40k_offline_v1
  export DUAL_AR_INITIAL_COMPLETED_STEPS="${source_step}"
  export DUAL_AR_INIT_TRAINABLE_CHECKPOINT="${source_run}/checkpoints/steps_${source_step}_pytorch_model.pt"
  if [[ ! -s "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT}" ]]; then
    echo "Missing one-GPU trainable checkpoint: ${DUAL_AR_INIT_TRAINABLE_CHECKPOINT}" >&2
    exit 2
  fi
fi

bash "${repo}/scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh" 2>&1 | tee -a "${DUAL_AR_LOG_FILE}"
