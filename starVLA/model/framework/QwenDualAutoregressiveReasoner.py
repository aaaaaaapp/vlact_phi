"""Autoregressive Qwen3-VL reasoner for action and physics token sequences."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.QwenActionTokenReasoner import (
    QwenActionTokenReasoner,
    format_action_token_solution,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


class LoRADelta(nn.Module):
    """Low-rank residual attached to an existing frozen Linear by a forward hook."""

    def __init__(self, in_features: int, out_features: int, *, rank: int,
                 alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires rank >= 1, alpha > 0 and dropout in [0, 1)")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scale


def attach_text_lora(language_model: nn.Module, *, rank: int, alpha: float,
                     dropout: float, target_names: tuple[str, ...]
                     ) -> tuple[nn.ModuleList, list]:
    """Attach trainable LoRA deltas without changing pretrained state-dict keys."""
    adapters = nn.ModuleList()
    handles = []
    for layer in language_model.layers:
        for name in target_names:
            target = layer
            for part in name.split("."):
                target = getattr(target, part)
            if not isinstance(target, nn.Linear):
                raise TypeError(f"LoRA target {name} is not nn.Linear")
            adapter = LoRADelta(target.in_features, target.out_features,
                                rank=rank, alpha=alpha, dropout=dropout)
            adapters.append(adapter)

            def add_delta(_module, inputs, output, *, _adapter=adapter):
                return output + _adapter(inputs[0]).to(output.dtype)

            handles.append(target.register_forward_hook(add_delta))
    if not adapters:
        raise ValueError("no LoRA target modules were found")
    return adapters, handles


def format_dual_solution(action_ids, physics_ids, *, action_vocab_size: int,
                         physics_vocab_size: int, action_count: int = 8,
                         physics_count: int = 64) -> str:
    actions = np.asarray(action_ids, dtype=np.int64)
    physics = np.asarray(physics_ids, dtype=np.int64)
    if actions.shape != (action_count,) or physics.shape != (physics_count,):
        raise ValueError(f"expected action {(action_count,)} and physics {(physics_count,)}, "
                         f"got {actions.shape} and {physics.shape}")
    if physics.min() < 0 or physics.max() >= physics_vocab_size:
        raise ValueError("physics token ID is outside the configured vocabulary")
    return format_action_token_solution(actions, vocab_size=action_vocab_size) + "".join(
        f"<robot_physics_{int(value)}>" for value in physics
    )


def extract_dual_generated_ids(generated_ids: torch.Tensor, *, prompt_width: int,
                               action_offset: int, action_vocab_size: int,
                               physics_offset: int, physics_vocab_size: int,
                               action_count: int = 8, physics_count: int = 64
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    if generated_ids.ndim != 2:
        raise ValueError("generated_ids must be [batch, length]")
    suffix = generated_ids[:, prompt_width:]
    if suffix.shape[1] != action_count + physics_count:
        raise ValueError("generated sequence length does not match action + physics counts")
    action = suffix[:, :action_count].long() - action_offset
    physics = suffix[:, action_count:].long() - physics_offset
    if ((action < 0) | (action >= action_vocab_size)).any():
        raise ValueError("generation contains an invalid action token")
    if ((physics < 0) | (physics >= physics_vocab_size)).any():
        raise ValueError("generation contains an invalid physics token")
    return action, physics


def aligned_dual_token_hidden(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    action_offset: int,
    action_vocab_size: int,
    physics_offset: int,
    physics_vocab_size: int,
    action_count: int = 32,
    physics_count: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the final-layer states that predict each supervised code ID.

    In a causal LM the hidden state at position i-1 predicts the token at i.
    Selecting states with the shifted label mask keeps that alignment even
    when prompts in a batch have different padded lengths.
    """
    if hidden.ndim != 3 or labels.ndim != 2 or hidden.shape[:2] != labels.shape:
        raise ValueError("hidden and labels must be [B, L, H] and [B, L]")
    shifted = labels[:, 1:]
    predictors = hidden[:, :-1]
    action_mask = (shifted >= action_offset) & (shifted < action_offset + action_vocab_size)
    physics_mask = (shifted >= physics_offset) & (shifted < physics_offset + physics_vocab_size)
    if not torch.all(action_mask.sum(dim=1) == action_count):
        raise ValueError(f"each sample must have {action_count} action targets")
    if not torch.all(physics_mask.sum(dim=1) == physics_count):
        raise ValueError(f"each sample must have {physics_count} physics targets")
    batch, width = hidden.shape[0], hidden.shape[-1]
    return (
        predictors[action_mask].reshape(batch, action_count, width),
        predictors[physics_mask].reshape(batch, physics_count, width),
    )


def keep_only_dual_target_labels(inputs: dict[str, torch.Tensor], *,
                                 action_offset: int, action_vocab_size: int,
                                 physics_offset: int, physics_vocab_size: int,
                                 action_count: int = 8, physics_count: int = 64
                                 ) -> dict[str, torch.Tensor]:
    """Exclude assistant terminators; supervise exactly the configured symbols."""
    labels = inputs["labels"].clone()
    action_mask = (labels >= action_offset) & (labels < action_offset + action_vocab_size)
    physics_mask = (labels >= physics_offset) & (labels < physics_offset + physics_vocab_size)
    for row in range(labels.shape[0]):
        symbols = labels[row][action_mask[row] | physics_mask[row]]
        if symbols.numel() != action_count + physics_count:
            raise ValueError(
                f"dual target must contain exactly {action_count} action and "
                f"{physics_count} physics symbols"
            )
        if not (action_mask[row][action_mask[row] | physics_mask[row]][:action_count].all()
                and physics_mask[row][action_mask[row] | physics_mask[row]][action_count:].all()):
            raise ValueError("dual target symbols are not ordered action then physics")
    labels[~(action_mask | physics_mask)] = -100
    inputs["labels"] = labels
    return inputs


def uniformize_physics_embedding_start(
    embedding: torch.Tensor, *, physics_offset: int, old_matrix_rows: int,
    seed: int,
) -> torch.Tensor:
    """Replace inherited padding rows using the new physics rows' distribution.

    Existing vocabulary rows and genuinely new rows are left unchanged, making
    this a controlled test of the old padded-row initialization alone.
    """
    if embedding.ndim != 2 or not 0 <= physics_offset < old_matrix_rows < embedding.shape[0]:
        raise ValueError("expected old padding rows followed by newly allocated physics rows")
    reference = embedding[old_matrix_rows:].float()
    mean = reference.mean(dim=0)
    std = reference.std(dim=0, unbiased=False)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    replacement = torch.randn(
        (old_matrix_rows - physics_offset, embedding.shape[1]),
        generator=generator, dtype=torch.float32,
    ) * std.cpu() + mean.cpu()
    result = embedding.clone()
    result[physics_offset:old_matrix_rows] = replacement.to(
        device=result.device, dtype=result.dtype
    )
    return result


def score_separate_heads(hidden: torch.Tensor, labels: torch.Tensor, *,
                         action_head: nn.Module, physics_head: nn.Module,
                         action_offset: int, action_vocab_size: int,
                         physics_offset: int, physics_vocab_size: int) -> dict[str, torch.Tensor]:
    """Next-token CE and logits on each modality's own code vocabulary."""
    if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
        raise ValueError("hidden must be [B, L, D] and labels [B, L]")
    targets = labels[:, 1:]
    previous_hidden = hidden[:, :-1]
    action_mask = (targets >= action_offset) & (targets < action_offset + action_vocab_size)
    physics_mask = (targets >= physics_offset) & (targets < physics_offset + physics_vocab_size)
    if not action_mask.any() or not physics_mask.any():
        raise ValueError("both action and physics targets are required")
    action_logits = action_head(previous_hidden[action_mask])
    physics_logits = physics_head(previous_hidden[physics_mask])
    action_targets = targets[action_mask] - action_offset
    physics_targets = targets[physics_mask] - physics_offset
    return {
        "action_logits": action_logits,
        "physics_logits": physics_logits,
        "action_targets": action_targets,
        "physics_targets": physics_targets,
        "action_ce": F.cross_entropy(action_logits.float(), action_targets),
        "physics_ce": F.cross_entropy(physics_logits.float(), physics_targets),
    }


@FRAMEWORK_REGISTRY.register("QwenDualAutoregressiveReasoner")
class QwenDualAutoregressiveReasoner(QwenActionTokenReasoner):
    """Teacher-forced causal LM SFT; constrained autoregressive generation."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        cfg = config.framework.dual_autoregressive_reasoner
        self.physics_vocab_size = int(cfg.get("physics_vocab_size", 2048))
        self.physics_token_count = int(cfg.get("physics_token_count", 64))
        self.separate_heads = bool(cfg.get("separate_heads", False))
        self.strict_modality_tying = bool(cfg.get("strict_modality_tying", False))
        if self.strict_modality_tying and not self.separate_heads:
            raise ValueError("strict_modality_tying requires separate_heads=true")
        lora_cfg = cfg.get("lora", {})
        self.use_lora = bool(lora_cfg.get("enabled", False))
        if self.use_lora and not self.strict_modality_tying:
            raise ValueError("LoRA mode requires strict_modality_tying=true")
        self.action_ce_weight = float(cfg.get("action_ce_weight", 1.0))
        self.physics_ce_weight = float(cfg.get("physics_ce_weight", 1.0))
        if self.action_ce_weight <= 0 or self.physics_ce_weight <= 0:
            raise ValueError("dual CE weights must be positive")
        if self.action_token_count not in (8, 32) or self.physics_token_count != 64:
            raise ValueError("this reasoner requires 8 or 32 action tokens and 64 physics tokens")
        if self.action_vocab_size != 525 or self.physics_vocab_size not in (2048, 25000):
            raise ValueError("checkpoint contract requires action K525 and physics K2048 or K25000")
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        symbols = [f"<robot_physics_{index}>" for index in range(self.physics_vocab_size)]
        existing = [tokenizer.convert_tokens_to_ids(symbol) for symbol in symbols]
        unknown = tokenizer.unk_token_id
        if all(index == unknown for index in existing):
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": symbols}, replace_additional_special_tokens=False
            )
            if added != self.physics_vocab_size:
                raise ValueError(f"expected {self.physics_vocab_size} new physics tokens, added {added}")
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        elif any(index == unknown for index in existing):
            raise ValueError("physics vocabulary is only partially installed")
        self.physics_token_offset = int(tokenizer.convert_tokens_to_ids(symbols[0]))
        if any(int(tokenizer.convert_tokens_to_ids(symbol)) != self.physics_token_offset + index
               for index, symbol in enumerate(symbols)):
            raise ValueError("physics token IDs must be contiguous")
        self.prompt_template = str(cfg.get(
            "prompt_template", f"Task: {{instruction}}\nPredict {self.action_token_count} robot action tokens, then 64 future physics tokens."
        ))
        # Resizing creates a new embedding parameter; reapply the trainable scope.
        self._configure_trainable_scope(int(config.framework.action_token_reasoner.get("train_last_n_layers", 2)))
        hidden_size = int(self.qwen_vl_interface.model.config.text_config.hidden_size)
        if self.strict_modality_tying:
            # Qwen's text embedding and text lm_head are tied. Keep both frozen;
            # action/physics input and output share only their own new tables.
            for parameter in self.qwen_vl_interface.model.get_input_embeddings().parameters():
                parameter.requires_grad = False
            self.action_embeddings = nn.Embedding(self.action_vocab_size, hidden_size)
            self.physics_embeddings = nn.Embedding(self.physics_vocab_size, hidden_size)
        elif self.separate_heads:
            self.action_head = nn.Linear(hidden_size, self.action_vocab_size, bias=False)
            self.physics_head = nn.Linear(hidden_size, self.physics_vocab_size, bias=False)
        if self.use_lora:
            # Freeze the base text model, including the previously trainable
            # final layers/norm. Only LoRA and the two tied code tables learn.
            for parameter in self.qwen_vl_interface.model.parameters():
                parameter.requires_grad = False
            targets = tuple(lora_cfg.get("target_modules", [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]))
            self.lora_adapters, self._lora_hook_handles = attach_text_lora(
                self.qwen_vl_interface.model.model.language_model,
                rank=int(lora_cfg.get("rank", 16)),
                alpha=float(lora_cfg.get("alpha", 32)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_names=targets,
            )

    def initialize_expanded_vocab_pretrained(self, checkpoint_path: str | Path) -> dict[str, int | str]:
        """Strictly warm-start Qwen; only new physics-vocabulary rows may be absent."""
        path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        prefix = "qwen_vl_interface."
        source = {key[len(prefix):]: value for key, value in checkpoint.items()
                  if key.startswith(prefix)}
        target = self.qwen_vl_interface.state_dict()
        if set(source) != set(target):
            raise ValueError(f"pretrained Qwen keys mismatch: missing={list(set(target)-set(source))[:5]}, "
                             f"extra={list(set(source)-set(target))[:5]}")
        expandable = {
            "model.model.language_model.embed_tokens.weight",
            "model.lm_head.weight",
        }
        adapted = {}
        expanded_rows = None
        old_matrix_rows = None
        for key, value in source.items():
            expected = target[key]
            if value.shape == expected.shape:
                adapted[key] = value
            elif (key in expandable and value.ndim == 2 and expected.ndim == 2
                  and value.shape[1] == expected.shape[1]
                  and value.shape[0] >= self.physics_token_offset
                  and expected.shape[0] == self.physics_token_offset + self.physics_vocab_size):
                full = expected.detach().clone()
                # The old model has padding rows beyond the tokenizer's true
                # vocabulary. New physics IDs reuse those numerical positions;
                # do not overwrite their fresh initialization with old padding.
                full[:self.physics_token_offset].copy_(value[:self.physics_token_offset])
                adapted[key] = full
                expanded_rows = self.physics_vocab_size
                old_matrix_rows = value.shape[0]
            else:
                raise ValueError(f"unexpected pretrained tensor shape for {key}: "
                                 f"{tuple(value.shape)} -> {tuple(expected.shape)}")
        if expanded_rows is None:
            raise ValueError("pretrained checkpoint did not require physics-vocabulary expansion")
        uniform_init = bool(self.config.framework.dual_autoregressive_reasoner.get(
            "uniform_physics_init", False
        ))
        if uniform_init:
            if self.config.trainer.get("dual_ar_trainable_checkpoint", None):
                raise ValueError("uniform_physics_init must start from the base model, not trained rows")
            if old_matrix_rows is None:
                raise ValueError("old embedding matrix size was not recorded")
            embed_key = "model.model.language_model.embed_tokens.weight"
            head_key = "model.lm_head.weight"
            equal_start = uniformize_physics_embedding_start(
                adapted[embed_key], physics_offset=self.physics_token_offset,
                old_matrix_rows=old_matrix_rows, seed=int(self.config.seed) + 271828,
            )
            adapted[embed_key] = equal_start
            adapted[head_key] = equal_start.clone()
        self.qwen_vl_interface.load_state_dict(adapted, strict=True)
        if self.separate_heads:
            weight = self.qwen_vl_interface.model.lm_head.weight.detach()
            with torch.no_grad():
                action_start = weight[self.action_token_offset:self.action_token_offset + self.action_vocab_size]
                physics_start = weight[self.physics_token_offset:self.physics_token_offset + self.physics_vocab_size]
                if self.strict_modality_tying:
                    self.action_embeddings.weight.copy_(action_start)
                    self.physics_embeddings.weight.copy_(physics_start)
                else:
                    self.action_head.weight.copy_(action_start)
                    self.physics_head.weight.copy_(physics_start)
        return {"checkpoint": str(path), "loaded_tensors": len(adapted),
                "new_vocab_rows": expanded_rows,
                "separate_heads": self.separate_heads,
                "strict_modality_tying": self.strict_modality_tying,
                "lora_enabled": self.use_lora,
                "uniform_physics_init": uniform_init,
                "reinitialized_old_padding_rows":
                    old_matrix_rows - self.physics_token_offset if uniform_init else 0}

    def initialize_trainable_checkpoint(self, checkpoint_path: str | Path) -> dict[str, int | str]:
        """Restore a trainable-only post-training checkpoint after the base warm-start."""
        path = Path(checkpoint_path).resolve()
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError(f"trainable checkpoint keys mismatch: "
                             f"missing={list(expected-set(state))[:5]}, "
                             f"extra={list(set(state)-expected)[:5]}")
        self.load_state_dict(state, strict=False)
        return {"checkpoint": str(path), "loaded_tensors": len(state)}

    def action_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.strict_modality_tying:
            return F.linear(hidden, self.action_embeddings.weight)
        return self.action_head(hidden)

    def physics_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.strict_modality_tying:
            return F.linear(hidden, self.physics_embeddings.weight)
        return self.physics_head(hidden)

    def _routed_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Use Qwen only for text/vision; code IDs use their own tied tables."""
        embeddings = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)
        action_mask = ((input_ids >= self.action_token_offset) &
                       (input_ids < self.action_token_offset + self.action_vocab_size))
        physics_mask = ((input_ids >= self.physics_token_offset) &
                        (input_ids < self.physics_token_offset + self.physics_vocab_size))
        if action_mask.any() or physics_mask.any():
            embeddings = embeddings.clone()
            if action_mask.any():
                embeddings[action_mask] = self.action_embeddings(
                    input_ids[action_mask] - self.action_token_offset
                )
            if physics_mask.any():
                embeddings[physics_mask] = self.physics_embeddings(
                    input_ids[physics_mask] - self.physics_token_offset
                )
        return embeddings

    def _strict_backbone_forward(self, inputs, *, past_key_values=None,
                                 cache_position=None, rope_deltas=None, use_cache=False):
        """Retain Qwen's image-aware RoPE while routing code input embeddings."""
        input_ids = inputs["input_ids"]
        backbone = self.qwen_vl_interface.model.model
        attention_mask = inputs.get("attention_mask")
        if past_key_values is None:
            position_ids, rope_deltas = backbone.get_rope_index(
                input_ids, inputs.get("image_grid_thw"), inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
            )
        else:
            if cache_position is None or rope_deltas is None:
                raise ValueError("cached strict decoding requires cache_position and rope_deltas")
            positions = cache_position.view(1, -1) + rope_deltas.to(input_ids.device)
            position_ids = positions.unsqueeze(0).expand(3, -1, -1)
        model_inputs = {key: value for key, value in inputs.items()
                        if key not in {"input_ids", "labels"}}
        model_inputs["inputs_embeds"] = self._routed_input_embeddings(input_ids)
        model_inputs["position_ids"] = position_ids
        model_inputs["use_cache"] = use_cache
        model_inputs["return_dict"] = True
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if cache_position is not None:
            model_inputs["cache_position"] = cache_position
        return backbone(**model_inputs), rope_deltas

    def forward(self, examples: List[dict] = None, **kwargs):
        if not examples:
            raise ValueError("examples must not be empty")
        solutions = [format_dual_solution(
            example["action_token_ids"], example["physics_token_ids"],
            action_vocab_size=self.action_vocab_size,
            physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        ) for example in examples]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[example["image"] for example in examples],
            instructions=self._reasoner_instructions(examples), solutions=solutions,
        )
        inputs = keep_only_dual_target_labels(
            inputs, action_offset=self.action_token_offset,
            action_vocab_size=self.action_vocab_size,
            physics_offset=self.physics_token_offset,
            physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        )
        if self.separate_heads:
            labels = inputs.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=labels.is_cuda):
                if self.strict_modality_tying:
                    hidden = self._strict_backbone_forward(inputs)[0].last_hidden_state
                else:
                    hidden = self.qwen_vl_interface.model.model(
                        **inputs, use_cache=False, return_dict=True,
                    ).last_hidden_state
                scores = score_separate_heads(
                    hidden, labels, action_head=self.action_logits, physics_head=self.physics_logits,
                    action_offset=self.action_token_offset, action_vocab_size=self.action_vocab_size,
                    physics_offset=self.physics_token_offset, physics_vocab_size=self.physics_vocab_size,
                )
            total = self.action_ce_weight * scores["action_ce"] + self.physics_ce_weight * scores["physics_ce"]
            return {"action_loss": total, "dual_token_ce_loss": total,
                    "action_token_ce_loss": scores["action_ce"],
                    "physics_token_ce_loss": scores["physics_ce"]}
        outputs = self.qwen_vl_interface(
            **inputs, output_attentions=False, output_hidden_states=False, return_dict=True
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise RuntimeError("dual-token causal LM loss is missing or non-finite")
        shifted_labels = inputs["labels"][:, 1:]
        shifted_logits = outputs.logits[:, :-1]
        action_mask = ((shifted_labels >= self.action_token_offset) &
                       (shifted_labels < self.action_token_offset + self.action_vocab_size))
        physics_mask = ((shifted_labels >= self.physics_token_offset) &
                        (shifted_labels < self.physics_token_offset + self.physics_vocab_size))
        action_ce = F.cross_entropy(shifted_logits[action_mask].float(), shifted_labels[action_mask])
        physics_ce = F.cross_entropy(shifted_logits[physics_mask].float(), shifted_labels[physics_mask])
        total = self.action_ce_weight * action_ce + self.physics_ce_weight * physics_ce
        return {"action_loss": total, "dual_token_ce_loss": total,
                "action_token_ce_loss": action_ce, "physics_token_ce_loss": physics_ce}

    @torch.inference_mode()
    def predict_token_ids(self, examples: List[dict] = None) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("examples must not be empty")
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[example["image"] for example in examples],
            instructions=self._reasoner_instructions(examples),
        )
        if self.separate_heads:
            return self._predict_with_separate_heads(inputs)
        prompt_width = int(inputs["input_ids"].shape[1])
        action_allowed = list(range(self.action_token_offset,
                                    self.action_token_offset + self.action_vocab_size))
        physics_allowed = list(range(self.physics_token_offset,
                                     self.physics_token_offset + self.physics_vocab_size))

        def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            generated = input_ids.shape[-1] - prompt_width
            return action_allowed if generated < self.action_token_count else physics_allowed

        generated = self.qwen_vl_interface.model.generate(
            **inputs, do_sample=False,
            max_new_tokens=self.action_token_count + self.physics_token_count,
            min_new_tokens=self.action_token_count + self.physics_token_count,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        action, physics = extract_dual_generated_ids(
            generated, prompt_width=prompt_width,
            action_offset=self.action_token_offset, action_vocab_size=self.action_vocab_size,
            physics_offset=self.physics_token_offset, physics_vocab_size=self.physics_vocab_size,
            action_count=self.action_token_count, physics_count=self.physics_token_count,
        )
        return {"action_token_ids": action.cpu().numpy(),
                "physics_token_ids": physics.cpu().numpy()}

    def _predict_with_separate_heads(self, inputs) -> dict[str, np.ndarray]:
        """Greedy AR decoding; feed predicted IDs back through Qwen's input embeddings."""
        prompt_length = int(inputs["input_ids"].shape[1])
        attention_mask = inputs["attention_mask"]
        backbone = self.qwen_vl_interface.model.model
        prefill_position = torch.arange(prompt_length, device=inputs["input_ids"].device)
        if self.strict_modality_tying:
            outputs, rope_deltas = self._strict_backbone_forward(
                inputs, cache_position=prefill_position, use_cache=True,
            )
        else:
            outputs = backbone(
                **inputs, use_cache=True, return_dict=True, cache_position=prefill_position,
            )
        action_ids, physics_ids = [], []
        total_steps = self.action_token_count + self.physics_token_count
        for step in range(total_steps):
            hidden = outputs.last_hidden_state[:, -1, :]
            if step < self.action_token_count:
                code = self.action_logits(hidden).argmax(dim=-1)
                next_id = code + self.action_token_offset
                action_ids.append(code)
            else:
                code = self.physics_logits(hidden).argmax(dim=-1)
                next_id = code + self.physics_token_offset
                physics_ids.append(code)
            if step + 1 == total_steps:
                break
            attention_mask = torch.cat((
                attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))
            ), dim=1)
            next_inputs = {"input_ids": next_id[:, None], "attention_mask": attention_mask}
            next_position = torch.tensor([prompt_length + step], device=next_id.device)
            if self.strict_modality_tying:
                outputs, _ = self._strict_backbone_forward(
                    next_inputs, past_key_values=outputs.past_key_values,
                    cache_position=next_position, rope_deltas=rope_deltas, use_cache=True,
                )
            else:
                outputs = backbone(
                    **next_inputs, past_key_values=outputs.past_key_values,
                    cache_position=next_position, use_cache=True, return_dict=True,
                )
        return {"action_token_ids": torch.stack(action_ids, dim=1).cpu().numpy(),
                "physics_token_ids": torch.stack(physics_ids, dim=1).cpu().numpy()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        return self.predict_token_ids(examples)
