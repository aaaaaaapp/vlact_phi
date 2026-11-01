#!/usr/bin/env bash
# One A100-80GB, global batch 128. Full ZeRO state enables later resume.
set -euo pipefail

export WANDB_MODE=offline
export DUAL_AR_MICRO_BATCH_SIZE=16
export DUAL_AR_GRADIENT_CHECKPOINTING=0
export DUAL_AR_MAX_STEPS="${DUAL_AR_MAX_STEPS:-40000}"
export DUAL_AR_SAVE_INTERVAL="${DUAL_AR_SAVE_INTERVAL:-5000}"
export DUAL_AR_SAVE_FULL_STATE=1
export DUAL_AR_MAIN_PROCESS_PORT="${DUAL_AR_MAIN_PROCESS_PORT:-29690}"
export DUAL_AR_RUN_TAG="${DUAL_AR_RUN_TAG:-robotwin_ft_acp_1gpu_bs128_r32_32action_m16_gc0_40k_offline_v1}"
export DUAL_AR_LOG_FILE="${DUAL_AR_LOG_FILE:-artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m16_gc0_bs128_r32_40k_offline_v1.log}"

# For a retry, set DUAL_AR_FULL_STATE_RESUME_DIR to this run's training_state
# directory. Never combine it with the legacy trainable-only weight loader.
bash /mnt/jfzn/liujn/code/VLAct-PhysToken/scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_tuned_20k.sh
