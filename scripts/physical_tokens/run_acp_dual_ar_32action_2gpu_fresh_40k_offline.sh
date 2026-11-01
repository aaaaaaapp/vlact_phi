#!/usr/bin/env bash
# Fresh dual-AR post-training from the immutable 100k-step RoboTwin VLA base.
# Two A100-80GB GPUs x microbatch 16 x accumulation 4 = global batch 128.
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:/mnt/jfzn/liujn/code/VLAct-main/.acp_deps:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${repo}"
mkdir -p artifacts/robotwin_full50_study/acp_logs

for resume_variable in DUAL_AR_FULL_STATE_RESUME_DIR DUAL_AR_INIT_TRAINABLE_CHECKPOINT DUAL_AR_INITIAL_COMPLETED_STEPS; do
  if [[ -n "${!resume_variable:-}" ]]; then
    echo "Fresh training refuses ${resume_variable}; no earlier dual-AR weights may be loaded" >&2
    exit 2
  fi
done

export WANDB_MODE=offline
export DUAL_AR_ACTION_TOKEN_COUNT=32
export DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml
export DUAL_AR_LORA=1
export DUAL_AR_LORA_RANK=32
export DUAL_AR_LORA_ALPHA=32
export DUAL_AR_UNIFORM_PHYSICS_INIT=1
export DUAL_AR_NUM_PROCESSES=2
export DUAL_AR_MICRO_BATCH_SIZE=16
export DUAL_AR_GRAD_ACCUM_STEPS=4
export DUAL_AR_GLOBAL_BATCH_SIZE=128
export DUAL_AR_GRADIENT_CHECKPOINTING=0
export DUAL_AR_MAX_STEPS="${DUAL_AR_MAX_STEPS:-40000}"
export DUAL_AR_WARMUP_STEPS="${DUAL_AR_WARMUP_STEPS:-4000}"
export DUAL_AR_SAVE_INTERVAL="${DUAL_AR_SAVE_INTERVAL:-5000}"
export DUAL_AR_SAVE_FULL_STATE="${DUAL_AR_SAVE_FULL_STATE:-1}"
export DUAL_AR_LOG_GRAD_NORM=1
export DUAL_AR_LOGGING_FREQUENCY="${DUAL_AR_LOGGING_FREQUENCY:-100}"
export DUAL_AR_EVAL_ALL_TASKS="${DUAL_AR_EVAL_ALL_TASKS:-1}"
export DUAL_AR_EVAL_SAMPLES_PER_TASK="${DUAL_AR_EVAL_SAMPLES_PER_TASK:-16}"
export DUAL_AR_EVAL_FREE_SAMPLES="${DUAL_AR_EVAL_FREE_SAMPLES:-1}"
export DUAL_AR_MAIN_PROCESS_PORT=29694
export DUAL_AR_RUN_TAG="${DUAL_AR_RUN_TAG:-robotwin_ft_acp_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_v2}"
export DUAL_AR_LOG_FILE="${DUAL_AR_LOG_FILE:-artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_40k_v2.log}"

echo "ACP fresh dual-AR: 2 x 16 x 4 = global batch 128, step 0/${DUAL_AR_MAX_STEPS}, warmup ${DUAL_AR_WARMUP_STEPS}, gradient checkpointing off, grad norm logging on"
echo "Only the immutable VLA base checkpoint in ${DUAL_AR_CONFIG} will be loaded"
bash "${repo}/scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh" 2>&1 | tee -a "${DUAL_AR_LOG_FILE}"
