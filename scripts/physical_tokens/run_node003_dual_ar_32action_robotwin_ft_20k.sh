#!/usr/bin/env bash
# Node003 GPU 4-7 concurrent trial; separate checkpoint/log from the ACP job.
set -euo pipefail

repo=/mnt/jfzn/liujn/code/VLAct-PhysToken
if [[ "$(hostname)" != *node003* ]]; then
  echo "This launcher is only for node003" >&2
  exit 2
fi
if [[ ! -s "${repo}/artifacts/robotwin_full50_study/dual_ar_32action_64physics_targets.npz" ]]; then
  echo "Missing aligned 32-action/64-physics targets" >&2
  exit 2
fi
for gpu in 4 5 6 7; do
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${free_mib}" || "${free_mib}" -lt 35000 ]]; then
    echo "GPU ${gpu} has only ${free_mib:-unknown} MiB free; refusing shared-GPU trial" >&2
    exit 2
  fi
done

export PATH="/mnt/jfzn/liujn/miniconda3/envs/vlact/bin:${PATH}"
export PYTHONPATH="${repo}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:/mnt/jfzn/liujn/code/VLAct-main/.acp_deps:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export DUAL_AR_ACTION_TOKEN_COUNT=32
export DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml
export DUAL_AR_LORA=1
export DUAL_AR_LORA_RANK=32
export DUAL_AR_LORA_ALPHA=32
export DUAL_AR_UNIFORM_PHYSICS_INIT=1
export DUAL_AR_NUM_PROCESSES=4
export DUAL_AR_MICRO_BATCH_SIZE="${NODE003_DUAL_AR_MICRO_BATCH_SIZE:-8}"
if ! [[ "${DUAL_AR_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( 32 % DUAL_AR_MICRO_BATCH_SIZE != 0 )); then
  echo "Micro batch must be a positive divisor of 32 to preserve global batch 128" >&2
  exit 2
fi
export DUAL_AR_GRAD_ACCUM_STEPS="$((32 / DUAL_AR_MICRO_BATCH_SIZE))"
export DUAL_AR_GLOBAL_BATCH_SIZE=128
export DUAL_AR_MAX_STEPS=20000
export DUAL_AR_WARMUP_STEPS=500
export DUAL_AR_SAVE_INTERVAL=1000
export DUAL_AR_MAIN_PROCESS_PORT=29682
export DUAL_AR_RUN_TAG="robotwin_ft_node003_bs128_r32_32action_m${DUAL_AR_MICRO_BATCH_SIZE}_v2"
export DUAL_AR_EVAL_ALL_TASKS=1
export DUAL_AR_EVAL_SAMPLES_PER_TASK=16
export DUAL_AR_EVAL_FREE_SAMPLES=1
export DUAL_AR_EVAL_GPU=4

cd "${repo}"
mkdir -p artifacts/robotwin_full50_study/node003_logs
log_file="artifacts/robotwin_full50_study/node003_logs/dual_ar_32action_bs128_r32_m${DUAL_AR_MICRO_BATCH_SIZE}_v2.log"
echo "$(date -Is) node003 GPUs 4-7, micro batch ${DUAL_AR_MICRO_BATCH_SIZE}, accumulation ${DUAL_AR_GRAD_ACCUM_STEPS}, global batch 128"
echo "Run tag: ${DUAL_AR_RUN_TAG}"
echo "Persistent log: ${repo}/${log_file}"
bash scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh 2>&1 | tee -a "${log_file}"
