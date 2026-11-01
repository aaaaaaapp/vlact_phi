#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python_bin="${DUAL_AR_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
accelerate_bin="${DUAL_AR_ACCELERATE:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/accelerate}"
study="artifacts/robotwin_full50_study"
action_count="${DUAL_AR_ACTION_TOKEN_COUNT:-8}"
if [[ "${action_count}" != "8" && "${action_count}" != "32" ]]; then
  echo "DUAL_AR_ACTION_TOKEN_COUNT must be 8 or 32" >&2
  exit 2
fi
action_ids="${study}/action_${action_count}x525_ids_vlact_for_dual_ar.npz"
physics_ids="${study}/physics_64x2048_ids_for_dual_ar.npz"
dual_ids="${study}/dual_ar_${action_count}action_64physics_targets.npz"
if [[ "${action_count}" == "32" ]]; then
  action_checkpoint="${study}/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt"
else
  action_checkpoint="${study}/action_tokenizers/ema_8x525_decay095_dead100_steps20000/step-00020000.pt"
fi
if [[ ! -s "${action_checkpoint}" ]]; then
  echo "Missing action tokenizer checkpoint: ${action_checkpoint}" >&2
  exit 2
fi

if [[ -n "${DUAL_AR_WAIT_EXPORT_PID:-}" ]]; then
  if ! [[ "${DUAL_AR_WAIT_EXPORT_PID}" =~ ^[0-9]+$ ]]; then
    echo "DUAL_AR_WAIT_EXPORT_PID must be a numeric PID" >&2
    exit 2
  fi
  echo "Waiting for our physics export PID ${DUAL_AR_WAIT_EXPORT_PID} to finish"
  while kill -0 "${DUAL_AR_WAIT_EXPORT_PID}" 2>/dev/null; do
    sleep 30
  done
  if [[ ! -s "${physics_ids}" ]]; then
    echo "Physics export ended without a complete ID cache: ${physics_ids}" >&2
    exit 1
  fi
fi

if [[ ! -s "${action_ids}" ]]; then
  CUDA_VISIBLE_DEVICES="${DUAL_AR_ACTION_EXPORT_GPU:-0}" "${python_bin}" \
    scripts/physical_tokens/export_action_token_ids.py \
    --input "${study}/action_all50_wrap32.npz" \
    --checkpoint "${action_checkpoint}" \
    --output "${action_ids}" --batch-size 512 --num-workers 4 --device cuda:0
fi
if [[ ! -s "${physics_ids}" ]]; then
  CUDA_VISIBLE_DEVICES="${DUAL_AR_PHYSICS_EXPORT_GPU:-1}" "${python_bin}" \
    scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py \
    --input "${study}/physics_all50_spatial4x4_17frames.npz" \
    --checkpoint "${study}/physics_tokenizers/ema_t64_k2048_steps20000/step-00020000.pt" \
    --output "${physics_ids}" --batch-size 128 --num-workers 4 --device cuda:0
fi
if [[ ! -s "${dual_ids}" ]]; then
  "${python_bin}" scripts/physical_tokens/merge_multitask_dual_token_targets.py \
    --action-cache "${action_ids}" --physics-cache "${physics_ids}" \
    --physics-feature-cache "${study}/physics_all50_spatial4x4_17frames.npz" \
    --output "${dual_ids}"
fi

"${python_bin}" -c '
import sys
import numpy as np
action_path, physics_path, dual_path = sys.argv[1:4]
action_count = int(sys.argv[4])
for path, key, width in (
    (action_path, "action_token_ids", action_count),
    (physics_path, "physics_token_ids", 64),
    (dual_path, "action_token_ids", action_count),
    (dual_path, "physics_token_ids", 64),
):
    with np.load(path, allow_pickle=False) as cache:
        shape = cache[key].shape
    if len(shape) != 2 or shape[1] != width:
        raise ValueError(f"{path}: {key} has shape {shape}, expected [N,{width}]")
' \
  "${action_ids}" "${physics_ids}" "${dual_ids}" "${action_count}"

num_processes="${DUAL_AR_NUM_PROCESSES:-4}"
micro_batch="${DUAL_AR_MICRO_BATCH_SIZE:-1}"
grad_accum="${DUAL_AR_GRAD_ACCUM_STEPS:-1}"
global_batch="${DUAL_AR_GLOBAL_BATCH_SIZE:-}"
if ! [[ "${num_processes}" =~ ^[1-9][0-9]*$ && "${micro_batch}" =~ ^[1-9][0-9]*$ && "${grad_accum}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU count, micro batch and gradient accumulation must be positive integers" >&2
  exit 2
fi
actual_global_batch=$((num_processes * micro_batch * grad_accum))
if [[ -n "${global_batch}" && ! "${global_batch}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DUAL_AR_GLOBAL_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "${global_batch}" && "${actual_global_batch}" -ne "${global_batch}" ]]; then
  echo "Global batch mismatch: ${num_processes} GPUs x ${micro_batch} per GPU x ${grad_accum} accumulation = ${actual_global_batch}, expected ${global_batch}" >&2
  exit 2
fi
echo "Training global batch: ${num_processes} x ${micro_batch} x ${grad_accum} = ${actual_global_batch}"
max_train_steps="${DUAL_AR_MAX_STEPS:-500}"
warmup_steps="${DUAL_AR_WARMUP_STEPS:-$((max_train_steps / 10))}"
config_path="${DUAL_AR_CONFIG:-examples/Robotwin/train_files/qwen_dual_ar_8action_64physics_full50.yaml}"
save_interval="${DUAL_AR_SAVE_INTERVAL:-${max_train_steps}}"
if [[ ! -s "${config_path}" ]]; then
  echo "Missing dual-AR config: ${config_path}" >&2
  exit 2
fi
"${python_bin}" -c '
import sys
from omegaconf import OmegaConf
config = OmegaConf.load(sys.argv[1])
action_count = int(sys.argv[2])
cache_path = sys.argv[3]
if int(config.framework.action_token_reasoner.token_count) != action_count:
    raise ValueError("action length differs between config and exported targets")
if int(config.framework.dual_autoregressive_reasoner.physics_token_count) != 64:
    raise ValueError("expected 64 physics tokens")
if str(config.datasets.vla_data.discrete_token_cache).lstrip("./") != cache_path:
    raise ValueError("config does not point to the selected dual-token target cache")
' "${config_path}" "${action_count}" "${dual_ids}"
if ! [[ "${save_interval}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DUAL_AR_SAVE_INTERVAL must be a positive integer" >&2
  exit 2
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
run_id="qwen_dual_ar_${action_count}a64p_full50_seed42_steps${max_train_steps}"
extra_train_args=()
extra_eval_args=()
if [[ -n "${DUAL_AR_GRADIENT_CHECKPOINTING:-}" ]]; then
  if [[ "${DUAL_AR_GRADIENT_CHECKPOINTING}" == "1" ]]; then
    extra_train_args+=(--trainer.enable_gradient_checkpointing true)
  elif [[ "${DUAL_AR_GRADIENT_CHECKPOINTING}" == "0" ]]; then
    extra_train_args+=(--trainer.enable_gradient_checkpointing false)
  else
    echo "DUAL_AR_GRADIENT_CHECKPOINTING must be 0 or 1" >&2
    exit 2
  fi
fi
if [[ "${DUAL_AR_SAVE_FULL_STATE:-0}" == "1" ]]; then
  extra_train_args+=(--trainer.dual_ar_save_full_state true)
elif [[ "${DUAL_AR_SAVE_FULL_STATE:-0}" != "0" ]]; then
  echo "DUAL_AR_SAVE_FULL_STATE must be 0 or 1" >&2
  exit 2
fi
if [[ "${DUAL_AR_LOG_GRAD_NORM:-0}" == "1" ]]; then
  extra_train_args+=(--trainer.log_grad_norm true)
elif [[ "${DUAL_AR_LOG_GRAD_NORM:-0}" != "0" ]]; then
  echo "DUAL_AR_LOG_GRAD_NORM must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${DUAL_AR_LOGGING_FREQUENCY:-}" ]]; then
  if ! [[ "${DUAL_AR_LOGGING_FREQUENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DUAL_AR_LOGGING_FREQUENCY must be a positive integer" >&2
    exit 2
  fi
  extra_train_args+=(--trainer.logging_frequency "${DUAL_AR_LOGGING_FREQUENCY}")
fi
if [[ -n "${DUAL_AR_FULL_STATE_RESUME_DIR:-}" ]]; then
  if [[ ! -s "${DUAL_AR_FULL_STATE_RESUME_DIR}/latest" ]]; then
    echo "Missing full-state latest tag: ${DUAL_AR_FULL_STATE_RESUME_DIR}/latest" >&2
    exit 2
  fi
  extra_train_args+=(--trainer.dual_ar_full_state_resume_dir "${DUAL_AR_FULL_STATE_RESUME_DIR}")
fi
if [[ "${DUAL_AR_SEPARATE_HEADS:-1}" == "1" ]]; then
  extra_train_args+=(--framework.dual_autoregressive_reasoner.separate_heads true)
  if [[ "${DUAL_AR_STRICT_MODALITY_TYING:-1}" == "1" ]]; then
    run_id="${run_id}_strict_tied_heads"
    extra_train_args+=(--framework.dual_autoregressive_reasoner.strict_modality_tying true)
    extra_eval_args+=(--strict-modality-tying)
  elif [[ "${DUAL_AR_STRICT_MODALITY_TYING:-1}" == "0" ]]; then
    run_id="${run_id}_output_only_heads"
    extra_train_args+=(--framework.dual_autoregressive_reasoner.strict_modality_tying false)
    extra_eval_args+=(--output-only-heads)
  else
    echo "DUAL_AR_STRICT_MODALITY_TYING must be 0 or 1" >&2
    exit 2
  fi
elif [[ "${DUAL_AR_SEPARATE_HEADS:-1}" == "0" ]]; then
  run_id="${run_id}_shared_heads"
  extra_train_args+=(--framework.dual_autoregressive_reasoner.separate_heads false)
  extra_train_args+=(--framework.dual_autoregressive_reasoner.strict_modality_tying false)
  extra_eval_args+=(--shared-heads)
else
  echo "DUAL_AR_SEPARATE_HEADS must be 0 or 1" >&2
  exit 2
fi
if [[ "${DUAL_AR_UNIFORM_PHYSICS_INIT:-0}" == "1" ]]; then
  if [[ -n "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT:-}" ]]; then
    echo "Uniform physics initialization requires a fresh run without a trainable checkpoint" >&2
    exit 2
  fi
  run_id="${run_id}_uniform_physics_init"
  extra_train_args+=(--framework.dual_autoregressive_reasoner.uniform_physics_init true)
elif [[ "${DUAL_AR_UNIFORM_PHYSICS_INIT:-}" == "0" ]]; then
  # The YAML default may be true. A trained-codebook warm-start must override it.
  extra_train_args+=(--framework.dual_autoregressive_reasoner.uniform_physics_init false)
fi
if [[ "${DUAL_AR_LORA:-0}" == "1" ]]; then
  if [[ "${DUAL_AR_STRICT_MODALITY_TYING:-1}" != "1" ]]; then
    echo "LoRA configuration requires strict modality tying" >&2
    exit 2
  fi
  lora_rank="${DUAL_AR_LORA_RANK:-16}"
  if ! [[ "${lora_rank}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DUAL_AR_LORA_RANK must be a positive integer" >&2
    exit 2
  fi
  lora_alpha="${DUAL_AR_LORA_ALPHA:-$((lora_rank * 2))}"
  if ! [[ "${lora_alpha}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DUAL_AR_LORA_ALPHA must be a positive integer" >&2
    exit 2
  fi
  run_id="${run_id}_lora_r${lora_rank}"
  extra_train_args+=(--framework.dual_autoregressive_reasoner.lora.enabled true)
  extra_train_args+=(--framework.dual_autoregressive_reasoner.lora.rank "${lora_rank}")
  extra_train_args+=(--framework.dual_autoregressive_reasoner.lora.alpha "${lora_alpha}")
fi
if [[ -n "${DUAL_AR_RUN_TAG:-}" ]]; then
  if ! [[ "${DUAL_AR_RUN_TAG}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "DUAL_AR_RUN_TAG may only contain letters, digits, _ or -" >&2
    exit 2
  fi
  run_id="${run_id}_${DUAL_AR_RUN_TAG}"
fi
if [[ -e "results/Checkpoints/${run_id}" && -z "${DUAL_AR_FULL_STATE_RESUME_DIR:-}" ]]; then
  echo "Run directory already exists; refusing to overwrite: results/Checkpoints/${run_id}" >&2
  exit 2
fi
if [[ -n "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT:-}" ]]; then
  extra_train_args+=(--trainer.dual_ar_trainable_checkpoint "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT}")
fi
if [[ -n "${DUAL_AR_INITIAL_COMPLETED_STEPS:-}" ]]; then
  if ! [[ "${DUAL_AR_INITIAL_COMPLETED_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DUAL_AR_INITIAL_COMPLETED_STEPS must be a positive integer" >&2
    exit 2
  fi
  if [[ -z "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT:-}" ]] || [[ -n "${DUAL_AR_FULL_STATE_RESUME_DIR:-}" ]]; then
    echo "Initial completed steps require a trainable-only checkpoint warm-start" >&2
    exit 2
  fi
  if [[ "${DUAL_AR_INIT_TRAINABLE_CHECKPOINT}" != *"/steps_${DUAL_AR_INITIAL_COMPLETED_STEPS}_pytorch_model.pt" ]]; then
    echo "Warm-start checkpoint filename does not match initial completed steps" >&2
    exit 2
  fi
  extra_train_args+=(--trainer.dual_ar_initial_completed_steps "${DUAL_AR_INITIAL_COMPLETED_STEPS}")
fi
"${accelerate_bin}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${DUAL_AR_MAIN_PROCESS_PORT:-29672}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_path}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.gradient_accumulation_steps "${grad_accum}" \
  --datasets.vla_data.per_device_batch_size "${micro_batch}" \
  --trainer.num_warmup_steps "${warmup_steps}" \
  --trainer.eval_interval "$((max_train_steps + 1))" \
  --trainer.save_interval "${save_interval}" \
  "${extra_train_args[@]}" \
  --run_id "${run_id}"

checkpoint="results/Checkpoints/${run_id}/final_model/pytorch_model.pt"
if [[ ! -s "${checkpoint}" ]]; then
  echo "Training ended without a final checkpoint: ${checkpoint}" >&2
  exit 1
fi
if [[ "${DUAL_AR_SKIP_EVAL:-0}" == "1" ]]; then
  echo "Skipping held-out evaluation for smoke test"
  exit 0
elif [[ "${DUAL_AR_SKIP_EVAL:-0}" != "0" ]]; then
  echo "DUAL_AR_SKIP_EVAL must be 0 or 1" >&2
  exit 2
fi
eval_tasks=(adjust_bottle click_alarmclock)
if [[ "${DUAL_AR_EVAL_ALL_TASKS:-0}" == "1" ]]; then
  mapfile -t eval_tasks < docs/robotwin_50task_list.txt
  if [[ "${#eval_tasks[@]}" -ne 50 ]]; then
    echo "Expected exactly 50 evaluation tasks" >&2
    exit 1
  fi
fi
CUDA_VISIBLE_DEVICES="${DUAL_AR_EVAL_GPU:-0}" "${python_bin}" \
  scripts/physical_tokens/evaluate_dual_autoregressive_reasoner.py \
  --config "${config_path}" \
  --checkpoint "${checkpoint}" \
  --split docs/robotwin_tokenizer_episode_split.json \
  --tasks "${eval_tasks[@]}" \
  --samples-per-task "${DUAL_AR_EVAL_SAMPLES_PER_TASK:-2}" \
  --free-samples "${DUAL_AR_EVAL_FREE_SAMPLES:-2}" \
  "${extra_eval_args[@]}" \
  --output "results/Checkpoints/${run_id}/heldout_diagnostic.json"
