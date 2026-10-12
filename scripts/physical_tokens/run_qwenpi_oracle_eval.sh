#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
checkpoint="${2:-}"
num_samples="${3:-32}"
gpu="${4:-0}"

if [[ "${mode}" != "e0" && "${mode}" != "e1" && "${mode}" != "e2" && "${mode}" != "b2" && "${mode}" != "b3" && "${mode}" != "b4" ]] || [[ -z "${checkpoint}" ]]; then
  echo "Usage: $0 {e0|e1|e2|b2|b3|b4} CHECKPOINT [num_samples] [gpu]" >&2
  exit 2
fi

checkpoint_name="$(basename "${checkpoint}" .pt)"
eval_variant_suffix="${EVAL_VARIANT_SUFFIX:-}"
output="${EVAL_OUTPUT:-artifacts/evaluations/qwenpi_${mode}${eval_variant_suffix}_${checkpoint_name}_${num_samples}samples.json}"
case "${mode}" in
  e2)
    default_token_cache="artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=0
    ;;
  b2)
    default_token_cache="artifacts/token_caches/click_alarmclock_oracle_physics_ema_4x128_step1000.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  b3)
    default_token_cache="artifacts/token_caches/click_alarmclock_pred_action4_oracle_physics4.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  b4)
    default_token_cache="artifacts/token_caches/click_alarmclock_pred_action4_shuffled_physics4_seed42.npz"
    default_action_vocab_size=128
    default_max_action_tokens=4
    default_physics_vocab_size=128
    ;;
  *)
    default_token_cache="artifacts/token_caches/click_alarmclock_clean_ema_step5000.npz"
    default_action_vocab_size=512
    default_max_action_tokens=8
    default_physics_vocab_size=0
    ;;
esac
token_cache="${TOKEN_CACHE:-${ACTION_TOKEN_CACHE:-${default_token_cache}}}"
action_vocab_size="${ACTION_VOCAB_SIZE:-${default_action_vocab_size}}"
max_action_tokens="${MAX_ACTION_TOKENS:-${default_max_action_tokens}}"
physics_vocab_size="${PHYSICS_VOCAB_SIZE:-${default_physics_vocab_size}}"
max_physics_tokens="${MAX_PHYSICS_TOKENS:-4}"
physics_fusion_mode="${PHYSICS_FUSION_MODE:-concat}"
physics_fusion_dim="${PHYSICS_FUSION_DIM:-256}"
physics_fusion_heads="${PHYSICS_FUSION_HEADS:-8}"
dit_injection_layers="${DIT_INJECTION_LAYERS:-all}"
dit_attention_dropout="${DIT_ATTENTION_DROPOUT:-0.0}"
physics_condition_dropout="${PHYSICS_CONDITION_DROPOUT:-0.0}"
physics_condition_enabled="${PHYSICS_CONDITION_ENABLED:-true}"
dit_action_tokens_only="${DIT_ACTION_TOKENS_ONLY:-false}"
dit_physics_adapter_type="${DIT_PHYSICS_ADAPTER_TYPE:-cross_attention}"
dit_physics_memory_pooling="${DIT_PHYSICS_MEMORY_POOLING:-none}"
physics_tokens_per_transition="${PHYSICS_TOKENS_PER_TRANSITION:-16}"
if [[ "${physics_condition_enabled}" == "true" ]]; then
  physics_condition_flag=(--physics-condition-enabled)
else
  physics_condition_flag=(--no-physics-condition-enabled)
fi
if [[ "${dit_action_tokens_only}" == "true" ]]; then
  dit_action_tokens_flag=(--dit-action-tokens-only)
else
  dit_action_tokens_flag=(--no-dit-action-tokens-only)
fi
shared_pretrained_checkpoint="/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt"
node_local_pretrained_checkpoint="${NODE_LOCAL_PRETRAINED_CHECKPOINT:-/tmp/vlact_qwen3_pretrain_steps100000.pt}"
if [[ -f "${node_local_pretrained_checkpoint}" ]]; then
  default_pretrained_checkpoint="${node_local_pretrained_checkpoint}"
else
  default_pretrained_checkpoint="${shared_pretrained_checkpoint}"
fi
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-${default_pretrained_checkpoint}}"
action_init_checkpoint="${ACTION_INIT_CHECKPOINT:-}"
action_init_args=()
if [[ -n "${action_init_checkpoint}" ]]; then
  action_init_args+=(--action-init-checkpoint "${action_init_checkpoint}")
fi

CUDA_VISIBLE_DEVICES="${gpu}" /mnt/jfzn/liujn/miniconda3/envs/vlact/bin/python \
  scripts/physical_tokens/evaluate_qwenpi_oracle.py \
  --config examples/Robotwin/train_files/qwenpi_physical_tokens_click_alarmclock.yaml \
  --checkpoint "${checkpoint}" \
  --mode "${mode}" \
  --token-cache "${token_cache}" \
  --action-vocab-size "${action_vocab_size}" \
  --max-action-tokens "${max_action_tokens}" \
  --physics-vocab-size "${physics_vocab_size}" \
  --max-physics-tokens "${max_physics_tokens}" \
  --physics-fusion-mode "${physics_fusion_mode}" \
  --physics-fusion-dim "${physics_fusion_dim}" \
  --physics-fusion-heads "${physics_fusion_heads}" \
  --dit-injection-layers "${dit_injection_layers}" \
  --dit-attention-dropout "${dit_attention_dropout}" \
  "${dit_action_tokens_flag[@]}" \
  --dit-physics-adapter-type "${dit_physics_adapter_type}" \
  --dit-physics-memory-pooling "${dit_physics_memory_pooling}" \
  --physics-tokens-per-transition "${physics_tokens_per_transition}" \
  --physics-condition-dropout "${physics_condition_dropout}" \
  "${physics_condition_flag[@]}" \
  --pretrained-checkpoint "${pretrained_checkpoint}" \
  "${action_init_args[@]}" \
  --pretrained-checkpoint-sha256 5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d \
  --split-manifest artifacts/splits/click_alarmclock_seed42.json \
  --num-samples "${num_samples}" \
  --seed 42 \
  --device cuda:0 \
  --output "${output}"

echo "Wrote ${output}"
