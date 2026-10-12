#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

checkpoint="results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_b3_d42_predicted_k256_dit_first_actiononly_7gpu_adapteronly_seed42_steps20/checkpoints/steps_20_pytorch_model.pt"
aligned="artifacts/token_caches/d41_8tasks_predicted_dual_parts/click_alarmclock_predicted_dual.npz"
shuffled="artifacts/token_caches/d42_click_predicted_dual_episode_shuffled_seed42.npz"
split="artifacts/splits/click_alarmclock_train_probe_seed42.json"
eval_dir="artifacts/evaluations/d43_d42_checkpoint_train_episodes_0to4_190samples"

for required in "${checkpoint}" "${aligned}" "${shuffled}" "${split}"; do
  [[ -s "${required}" ]] || { echo "missing D43 input: ${required}" >&2; exit 2; }
done
mkdir -p "${eval_dir}"

run_direct() {
  local cache="$1"
  local enabled_flag="$2"
  local output="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python \
    scripts/physical_tokens/evaluate_qwenpi_oracle.py \
    --config examples/Robotwin/train_files/qwenpi_physical_tokens_click_alarmclock.yaml \
    --checkpoint "${checkpoint}" --mode b3 --token-cache "${cache}" \
    --action-vocab-size 128 --max-action-tokens 4 \
    --physics-vocab-size 256 --max-physics-tokens 256 \
    --physics-fusion-mode dit_multilayer_cross_attention --physics-fusion-heads 8 \
    --dit-injection-layers first --dit-attention-dropout 0.0 \
    --dit-action-tokens-only --dit-physics-adapter-type cross_attention \
    --dit-physics-memory-pooling none --physics-tokens-per-transition 16 \
    --physics-condition-dropout 0.2 "${enabled_flag}" \
    --pretrained-checkpoint /tmp/vlact_qwen3_pretrain_steps100000.pt \
    --action-init-checkpoint results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt \
    --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
    --split-manifest "${split}" --num-samples 190 --seed 42 --device cuda:0 \
    --output "${output}"
}

pids=()
run_direct "${aligned}" --physics-condition-enabled "${eval_dir}/aligned_enabled.json" 0 & pids+=("$!")
run_direct "${aligned}" --no-physics-condition-enabled "${eval_dir}/hard_disabled.json" 1 & pids+=("$!")
run_direct "${shuffled}" --physics-condition-enabled "${eval_dir}/episode_shuffled_enabled.json" 2 & pids+=("$!")
for pid in "${pids[@]}"; do
  wait "${pid}"
done

/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python \
  scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
  --control "${eval_dir}/hard_disabled.json" \
  --enabled "${eval_dir}/aligned_enabled.json" \
  --output "${eval_dir}/aligned_vs_disabled.json"
/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python \
  scripts/physical_tokens/compare_qwenpi_aligned_shuffled.py \
  --shuffled "${eval_dir}/episode_shuffled_enabled.json" \
  --aligned "${eval_dir}/aligned_enabled.json" \
  --output "${eval_dir}/aligned_vs_shuffled.json"
