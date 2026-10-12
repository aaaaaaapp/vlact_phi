#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${STARVLA_PYTHON:-/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python}"
steps="${D31_STEPS:-2000}"
selection="artifacts/evaluations/d29_full_diversity_codebook_gate.json"
oracle_gate="${D31_ORACLE_GATE:-artifacts/evaluations/d30_action_conditioned_oracle_gate.json}"
dual_cache="artifacts/token_caches/robotwin_d30_8tasks_dual_action4_physics256_selected_ema.npz"
conditioned_checkpoint="${D31_CONDITIONED_CHECKPOINT:-artifacts/physical_language_tokenizers/d30_action_conditioned_selected_steps3000.pt}"
config="artifacts/configs/qwen_dual_token_probe_d31_full_diversity_selected_steps${steps}.yaml"

oracle_passed=$("${python_bin}" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["passed"])).lower())' "${oracle_gate}")
if [[ "${oracle_passed}" != "true" ]]; then
  echo "D31 not started: D30 oracle double gate failed" >&2
  exit 3
fi
"${python_bin}" -u scripts/physical_tokens/render_d31_full_diversity_config.py \
  --template examples/Robotwin/train_files/qwen_dual_token_probe_d23_8tasks.yaml \
  --selection "${selection}" --dual-cache "${dual_cache}" --output "${config}" --steps "${steps}"
codebook=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_codebook_size"])' "${selection}")
physics_checkpoint=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${selection}")
run_id="qwen_dual_token_probe_d31_full_diversity_k${codebook}_steps${steps}"
run_dir="results/Checkpoints/${run_id}"
checkpoint="${run_dir}/checkpoints/steps_${steps}_pytorch_model.pt"
evaluation="artifacts/evaluations/d31_full_diversity_dual_k${codebook}_steps${steps}.json"
gate="artifacts/evaluations/d31_full_diversity_predicted_gate_k${codebook}_steps${steps}.json"

if [[ ! -s "${checkpoint}" ]]; then
  if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "D31 refused partial output directory: ${run_dir}" >&2
    exit 2
  fi
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  pretrained="/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt"
  if [[ -s /tmp/vlact_qwen3_pretrain_steps100000.pt ]]; then
    pretrained="/tmp/vlact_qwen3_pretrain_steps100000.pt"
  fi
  "${python_bin}" -m accelerate.commands.launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "${D31_NPROC:-8}" --main_process_port "${MAIN_PROCESS_PORT:-29671}" \
    starVLA/training/train_starvla.py --config_yaml "${config}" \
    --trainer.pretrained_checkpoint "${pretrained}" --run_id "${run_id}"
fi

if [[ ! -s "${evaluation}" ]]; then
  CUDA_VISIBLE_DEVICES="${D31_EVAL_GPU:-5}" "${python_bin}" -u \
    scripts/physical_tokens/evaluate_multitask_dual_token_probe.py \
    --config "${config}" --checkpoint "${checkpoint}" \
    --baseline-config examples/Robotwin/train_files/qwen_action_token_probe_d16_8tasks.yaml \
    --baseline-checkpoint results/Checkpoints/qwen_action_token_probe_d16_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt \
    --token-cache "${dual_cache}" \
    --action-tokenizer-checkpoint artifacts/action_tokenizer_multitask/ema_4x128_steps5000/step-00005000.pt \
    --physics-tokenizer-checkpoint "${physics_checkpoint}" \
    --conditioned-decoder-checkpoint "${conditioned_checkpoint}" \
    --output "${evaluation}" --validation-episodes 5,20,25,39,49 \
    --samples-per-task "${D31_SAMPLES_PER_TASK:-64}" \
    --batch-size "${D31_EVAL_BATCH_SIZE:-2}" --device cuda:0
fi
"${python_bin}" -u scripts/physical_tokens/require_d31_predicted_physics_gate.py \
  --input "${evaluation}" --output "${gate}"
