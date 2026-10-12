#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

require_idle_gpus() {
  if [[ "${D1_REQUIRE_IDLE_GPUS:-1}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "D1 preflight failed: nvidia-smi is unavailable" >&2
    exit 2
  fi
  memory_threshold="${D1_IDLE_MEMORY_THRESHOLD_MIB:-256}"
  if [[ ! "${memory_threshold}" =~ ^[0-9]+$ ]]; then
    echo "D1 preflight failed: memory threshold must be a non-negative integer" >&2
    exit 2
  fi
  compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  busy_memory="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v threshold="${memory_threshold}" '$2 + 0 > threshold {print}')"
  if [[ -n "${compute_apps}" || -n "${busy_memory}" ]]; then
    echo "D1 preflight refused to start: GPUs have active processes or retained memory." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    if [[ -n "${busy_memory}" ]]; then
      echo "GPUs above ${memory_threshold} MiB used memory:" >&2
      echo "${busy_memory}" >&2
    fi
    echo "Set D1_REQUIRE_IDLE_GPUS=0 only after manually confirming resource sharing is safe." >&2
    exit 2
  fi
}

require_idle_gpus

tasks="click_alarmclock,beat_block_hammer,open_microwave,shake_bottle,move_playingcard_away,pick_dual_bottles,handover_block,place_object_basket"
cache="${D1_FEATURE_CACHE:-artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz}"
output="${D1_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d1_8tasks_ema_256x128_steps1000}"
max_steps="${D1_MAX_STEPS:-1000}"
shared_pretrained_checkpoint="/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt"
node_local_pretrained_checkpoint="${NODE_LOCAL_PRETRAINED_CHECKPOINT:-/tmp/vlact_qwen3_pretrain_steps100000.pt}"
if [[ -f "${node_local_pretrained_checkpoint}" ]]; then
  pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-${node_local_pretrained_checkpoint}}"
else
  pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-${shared_pretrained_checkpoint}}"
fi
printf -v checkpoint_name 'step-%08d.pt' "${max_steps}"
final_checkpoint="${output}/${checkpoint_name}"
baseline_checkpoint="${D1_BASELINE_CHECKPOINT:-artifacts/physical_language_tokenizers/spatiotemporal_ema_256x128_steps1000/step-00001000.pt}"

if [[ ! -s "${cache}" || "${D1_REEXPORT:-0}" == "1" ]]; then
  SPATIOTEMPORAL_DATA_NAMES="${tasks}" \
  SPATIOTEMPORAL_MAX_EPISODES=8 \
  SPATIOTEMPORAL_EXCLUDE_EPISODE_IDS=5,20,25,39,49 \
  SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
  PRETRAINED_CHECKPOINT="${pretrained_checkpoint}" \
  bash scripts/physical_tokens/run_spatiotemporal_feature_export.sh
else
  echo "Reusing D1 feature cache: ${cache}"
fi

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/validate_d1_feature_cache.py \
  --input "${cache}" \
  --data-names "${tasks}" \
  --excluded-episode-ids 5,20,25,39,49 \
  --episodes-per-task 8 \
  --expected-episode-ids 0,1,2,3,4,6,7,8 \
  --validation-episodes-per-task 2 \
  --max-train-clips-per-task 300 \
  --expected-unique-frames 14077 \
  --expected-train-clips 8785 \
  --expected-validation-clips 3244 \
  --expected-capped-train-clips 2394

# Export uses one GPU, while tokenizer training immediately expands to all requested GPUs.
# Re-check here in case another workload started during the export.
require_idle_gpus

if [[ -s "${final_checkpoint}" ]]; then
  echo "Reusing completed D1 tokenizer checkpoint: ${final_checkpoint}"
elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "D1 refused to overwrite partial output directory: ${output}" >&2
  echo "Inspect it and choose a new D1_TOKENIZER_OUTPUT or implement an explicit resume." >&2
  exit 2
else
  SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=300 \
  SPATIOTEMPORAL_MAX_STEPS="${max_steps}" \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
fi

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" \
  --checkpoint "${final_checkpoint}" \
  --output "${output}/evaluation_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D1_EVAL_DEVICE:-cuda:0}"

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" \
  --checkpoint "${baseline_checkpoint}" \
  --output "${output}/evaluation_d0_on_d1_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D1_EVAL_DEVICE:-cuda:0}"

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/compare_spatiotemporal_tokenizers.py \
  --baseline "${output}/evaluation_d0_on_d1_per_task.json" \
  --candidate "${output}/evaluation_per_task.json" \
  --output "${output}/comparison_d0_vs_d1.json"

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/analyze_spatiotemporal_token_structure.py \
  --input "${cache}" \
  --checkpoint "${final_checkpoint}" \
  --output "${output}/token_structure_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D1_EVAL_DEVICE:-cuda:0}"

"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/analyze_spatiotemporal_token_structure.py \
  --input "${cache}" \
  --checkpoint "${baseline_checkpoint}" \
  --output "${output}/token_structure_d0_on_d1_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D1_EVAL_DEVICE:-cuda:0}"

structure_source_report="${output}/token_structure_per_task.json"
structure_gate_report="${output}/token_structure_gate.json"
set +e
"${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
  scripts/physical_tokens/require_physics_token_structure_gate.py \
  "${structure_source_report}" \
  --output "${structure_gate_report}"
structure_gate_status=$?
set -e

if [[ "${D1_AUTO_GATED_SMOKE:-1}" == "1" ]]; then
  gate_report="${output}/comparison_d0_vs_d1.json"
  if "${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}" \
    scripts/physical_tokens/require_d1_representation_gate.py "${gate_report}" \
    && [[ "${structure_gate_status}" -eq 0 ]]; then
    echo "D1 representation and token-structure gates passed; starting gated policy-cache and adapter smoke."
    require_idle_gpus
      D1_TOKENIZER_OUTPUT="${output}" \
      D1_COMPARISON_REPORT="${gate_report}" \
      D1_STRUCTURE_REPORT="${structure_source_report}" \
      bash scripts/physical_tokens/prepare_d1_policy_token_cache.sh
    require_idle_gpus
      D1_TOKENIZER_OUTPUT="${output}" \
      D1_COMPARISON_REPORT="${gate_report}" \
      D1_STRUCTURE_REPORT="${structure_source_report}" \
      bash scripts/physical_tokens/run_dit_physics_adapter_smoke.sh first 20 8
    require_idle_gpus
      D1_TOKENIZER_OUTPUT="${output}" \
      D1_COMPARISON_REPORT="${gate_report}" \
      D1_STRUCTURE_REPORT="${structure_source_report}" \
      bash scripts/physical_tokens/evaluate_dit_physics_adapter_pair.sh first 20 380 0
  else
    echo "D1 representation or token-structure gate failed; adapter training remains blocked."
  fi
fi
