#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

require_idle_gpus() {
  if [[ "${D2_REQUIRE_IDLE_GPUS:-1}" != "1" ]]; then
    return
  fi
  memory_threshold="${D2_IDLE_MEMORY_THRESHOLD_MIB:-256}"
  compute_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  busy_memory="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v threshold="${memory_threshold}" '$2 + 0 > threshold {print}')"
  if [[ -n "${compute_apps}" || -n "${busy_memory}" ]]; then
    echo "D2 preflight refused to start: GPUs have active processes or retained memory." >&2
    exit 2
  fi
}

python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
cache="${D1_FEATURE_CACHE:-artifacts/future_feature_caches/robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz}"
d1_output="${D1_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d1_8tasks_ema_256x128_steps1000}"
output="${D2_TOKENIZER_OUTPUT:-artifacts/physical_language_tokenizers/spatiotemporal_d2_8tasks_spatial_aligned_ema_256x128_steps1000}"
max_steps="${D2_MAX_STEPS:-1000}"
printf -v checkpoint_name 'step-%08d.pt' "${max_steps}"
final_checkpoint="${output}/${checkpoint_name}"
d1_evaluation="${d1_output}/evaluation_per_task.json"

for required in "${cache}" "${cache}.manifest.json" "${d1_output}/step-00001000.pt" "${d1_evaluation}"; do
  if [[ ! -s "${required}" ]]; then
    echo "D2 requires a completed D1 artifact: ${required}" >&2
    exit 2
  fi
done

"${python_bin}" scripts/physical_tokens/validate_d1_feature_cache.py \
  --input "${cache}" \
  --data-names click_alarmclock,beat_block_hammer,open_microwave,shake_bottle,move_playingcard_away,pick_dual_bottles,handover_block,place_object_basket \
  --excluded-episode-ids 5,20,25,39,49 \
  --episodes-per-task 8 \
  --expected-episode-ids 0,1,2,3,4,6,7,8 \
  --validation-episodes-per-task 2 \
  --max-train-clips-per-task 300 \
  --expected-unique-frames 14077 \
  --expected-train-clips 8785 \
  --expected-validation-clips 3244 \
  --expected-capped-train-clips 2394

require_idle_gpus
if [[ -s "${final_checkpoint}" ]]; then
  echo "Reusing completed D2 tokenizer checkpoint: ${final_checkpoint}"
elif [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "D2 refused to overwrite partial output directory: ${output}" >&2
  exit 2
else
  SPATIOTEMPORAL_FEATURE_CACHE="${cache}" \
  SPATIOTEMPORAL_OUTPUT_DIR="${output}" \
  SPATIOTEMPORAL_VALIDATION_EPISODES_PER_TASK=2 \
  SPATIOTEMPORAL_MAX_TRAIN_CLIPS_PER_TASK=300 \
  SPATIOTEMPORAL_MAX_STEPS="${max_steps}" \
  SPATIOTEMPORAL_TOKEN_ROUTING=spatial_aligned \
  bash scripts/physical_tokens/run_spatiotemporal_physical_tokenizer.sh
fi

"${python_bin}" scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py \
  --input "${cache}" \
  --checkpoint "${final_checkpoint}" \
  --output "${output}/evaluation_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D2_EVAL_DEVICE:-cuda:0}"

"${python_bin}" scripts/physical_tokens/analyze_spatiotemporal_token_structure.py \
  --input "${cache}" \
  --checkpoint "${final_checkpoint}" \
  --output "${output}/token_structure_per_task.json" \
  --validation-episodes-per-task 2 \
  --device "${D2_EVAL_DEVICE:-cuda:0}"

"${python_bin}" scripts/physical_tokens/compare_spatiotemporal_tokenizers.py \
  --baseline "${d1_evaluation}" \
  --candidate "${output}/evaluation_per_task.json" \
  --allowed-model-config-differences token_routing \
  --comparison-name paired_same_cache_d1_vs_d2_spatial_routing \
  --output "${output}/comparison_d1_vs_d2.json"

"${python_bin}" scripts/physical_tokens/require_d2_joint_gate.py \
  --comparison "${output}/comparison_d1_vs_d2.json" \
  --structure "${output}/token_structure_per_task.json" \
  --output "${output}/joint_gate.json"

echo "D2 joint gate passed. Audit comparison_d1_vs_d2.json and joint_gate.json before any policy run."
