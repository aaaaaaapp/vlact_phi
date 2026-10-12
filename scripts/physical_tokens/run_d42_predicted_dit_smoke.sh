#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
gate="artifacts/evaluations/d41_full_diversity_predicted_gate_k256_steps8000_256samples.json"
aligned="artifacts/token_caches/d41_8tasks_predicted_dual_parts/click_alarmclock_predicted_dual.npz"
shuffled="artifacts/token_caches/d42_click_predicted_dual_episode_shuffled_seed42.npz"
tokenizer="artifacts/physical_language_tokenizers/d29_full_diversity_ema_256x256_steps8000/step-00008000.pt"
tokenizer_sha="8625b3d7580625b1a4a33bce874537a2863cd18ddfab0e595fa20537a8f76b5c"
action_init="results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt"
steps=20
tag="d42_predicted_k256_dit_first_actiononly_7gpu_adapteronly"
run="qwenpi_physical_tokens_click_alarmclock_b3_${tag}_seed42_steps${steps}"
checkpoint="results/Checkpoints/${run}/checkpoints/steps_${steps}_pytorch_model.pt"
eval_dir="artifacts/evaluations/${run}_380samples"

"${python_bin}" - <<'PY'
import json
p=json.load(open('artifacts/evaluations/d41_full_diversity_predicted_gate_k256_steps8000_256samples.json'))
if not p.get('passed') or not p.get('dit_authorized') or p.get('physics_vocab_size') != 256:
    raise SystemExit('D42 requires the passed D41 K256 DiT authorization gate')
PY
for required in "${aligned}" "${tokenizer}" "${action_init}"; do
  [[ -s "${required}" ]] || { echo "missing D42 input: ${required}" >&2; exit 2; }
done
[[ "$(sha256sum "${tokenizer}" | awk '{print $1}')" == "${tokenizer_sha}" ]] || {
  echo "D42 tokenizer SHA mismatch" >&2; exit 2;
}
if [[ ! -s "${shuffled}" ]]; then
  "${python_bin}" scripts/physical_tokens/merge_discrete_token_caches.py \
    --action-cache "${aligned}" --physics-cache "${aligned}" \
    --shuffle-physics-seed 42 --output "${shuffled}"
fi

if [[ ! -s "${checkpoint}" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NUM_PROCESSES=7 \
  TOKEN_CACHE="${aligned}" ACTION_VOCAB_SIZE=128 MAX_ACTION_TOKENS=4 \
  PHYSICS_VOCAB_SIZE=256 MAX_PHYSICS_TOKENS=256 \
  PHYSICS_FUSION_MODE=dit_multilayer_cross_attention PHYSICS_FUSION_HEADS=8 \
  PHYSICS_CONDITION_DROPOUT=0.2 PHYSICS_CONDITION_ENABLED=true \
  DIT_INJECTION_LAYERS=first DIT_ATTENTION_DROPOUT=0.0 \
  DIT_FREEZE_BASE_ACTION_MODEL=true DIT_ACTION_TOKENS_ONLY=true \
  ACTION_INIT_CHECKPOINT="${action_init}" \
  PHYSICS_TOKENIZER_INIT_CHECKPOINT="${tokenizer}" \
  PHYSICS_TOKENIZER_INIT_SHA256="${tokenizer_sha}" \
  FREEZE_MODULES=qwen_vl_interface,discrete_conditioner \
  VARIANT_SUFFIX="_${tag}" MAIN_PROCESS_PORT="${D42_PORT:-29742}" WANDB_MODE=disabled \
    bash scripts/physical_tokens/run_qwenpi_oracle_ablation.sh b3 "${steps}" 8
fi

mkdir -p "${eval_dir}"
common=(
  MAX_ACTION_TOKENS=4 PHYSICS_VOCAB_SIZE=256 MAX_PHYSICS_TOKENS=256
  PHYSICS_FUSION_MODE=dit_multilayer_cross_attention PHYSICS_FUSION_HEADS=8
  DIT_INJECTION_LAYERS=first DIT_ATTENTION_DROPOUT=0.0
  DIT_ACTION_TOKENS_ONLY=true PHYSICS_CONDITION_DROPOUT=0.2
  ACTION_INIT_CHECKPOINT="${action_init}"
)
env "${common[@]}" TOKEN_CACHE="${aligned}" PHYSICS_CONDITION_ENABLED=true \
  EVAL_OUTPUT="${eval_dir}/aligned_enabled.json" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh b3 "${checkpoint}" 380 0
env "${common[@]}" TOKEN_CACHE="${aligned}" PHYSICS_CONDITION_ENABLED=false \
  EVAL_OUTPUT="${eval_dir}/hard_disabled.json" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh b3 "${checkpoint}" 380 0
env "${common[@]}" TOKEN_CACHE="${shuffled}" PHYSICS_CONDITION_ENABLED=true \
  EVAL_OUTPUT="${eval_dir}/episode_shuffled_enabled.json" \
  bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh b3 "${checkpoint}" 380 0

"${python_bin}" scripts/physical_tokens/compare_qwenpi_paired_evaluations.py \
  --control "${eval_dir}/hard_disabled.json" --enabled "${eval_dir}/aligned_enabled.json" \
  --output "${eval_dir}/aligned_vs_disabled.json"
"${python_bin}" scripts/physical_tokens/compare_qwenpi_aligned_shuffled.py \
  --shuffled "${eval_dir}/episode_shuffled_enabled.json" \
  --aligned "${eval_dir}/aligned_enabled.json" \
  --output "${eval_dir}/aligned_vs_shuffled.json"
