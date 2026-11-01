#!/usr/bin/env bash
# ACP single A100-80GB, 32 micro-batch, 4 accumulation, no gradient checkpointing.
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:/mnt/jfzn/liujn/code/VLAct-main/.acp_deps:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export DUAL_AR_ACTION_TOKEN_COUNT=32
export DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml
export DUAL_AR_LORA=1
export DUAL_AR_LORA_RANK=32
export DUAL_AR_LORA_ALPHA=32
export DUAL_AR_UNIFORM_PHYSICS_INIT=1
export DUAL_AR_NUM_PROCESSES=1
export DUAL_AR_MICRO_BATCH_SIZE=32
export DUAL_AR_GRAD_ACCUM_STEPS=4
export DUAL_AR_GLOBAL_BATCH_SIZE=128
export DUAL_AR_GRADIENT_CHECKPOINTING=0
export DUAL_AR_MAX_STEPS=20000
export DUAL_AR_WARMUP_STEPS=500
export DUAL_AR_SAVE_INTERVAL=1000
export DUAL_AR_MAIN_PROCESS_PORT=29684
export DUAL_AR_RUN_TAG=robotwin_ft_acp_1gpu_bs128_r32_32action_m32_nogc_v1
export DUAL_AR_EVAL_ALL_TASKS=1
export DUAL_AR_EVAL_SAMPLES_PER_TASK=16
export DUAL_AR_EVAL_FREE_SAMPLES=1
export DUAL_AR_EVAL_GPU=0

cd "${repo}"
mkdir -p artifacts/robotwin_full50_study/acp_logs
log_file=artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m32_nogc_bs128_r32_v1.log
echo "ACP one A100-80GB: micro 32 x accumulation 4 = global batch 128; Qwen gradient checkpointing disabled"
echo "Python: $(command -v python)"
echo "Persistent log: ${repo}/${log_file}"
bash scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh 2>&1 | tee -a "${log_file}"
