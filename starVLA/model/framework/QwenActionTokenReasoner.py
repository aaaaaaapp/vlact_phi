"""Autoregressive Qwen3-VL reasoner for cached EMA-VQ action tokens."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def format_action_token_solution(token_ids, *, vocab_size: int) -> str:
    """Serialize one code sequence using the action tokens already in Qwen's vocabulary."""
    values = np.asarray(token_ids, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"action_token_ids must be one-dimensional, got {values.shape}")
    if values.size == 0:
        raise ValueError("action_token_ids must not be empty")
    if values.min() < 0 or values.max() >= vocab_size:
        raise ValueError(f"action token ID is outside [0, {vocab_size})")
    return "".join(f"<robot_action_{int(value)}>" for value in values)


def extract_generated_action_ids(
    generated_ids: torch.Tensor,
    *,
    prompt_width: int,
    token_offset: int,
    vocab_size: int,
    token_count: int,
) -> torch.Tensor:
    """Convert a constrained generation suffix back to EMA-VQ code IDs."""
    if generated_ids.ndim != 2:
        raise ValueError(f"generated_ids must be [B, L], got {tuple(generated_ids.shape)}")
    suffix = generated_ids[:, prompt_width:]
    if suffix.shape[1] != token_count:
        raise ValueError(f"expected {token_count} generated tokens, got {suffix.shape[1]}")
    result = suffix.long() - int(token_offset)
    if result.numel() and (result.min() < 0 or result.max() >= vocab_size):
        raise ValueError("generation contains tokens outside the configured action vocabulary")
    return result


@FRAMEWORK_REGISTRY.register("QwenActionTokenReasoner")
class QwenActionTokenReasoner(baseframework):
    """Predict a short sequence of EMA-VQ action codes with causal LM loss."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=config)

        reasoner_cfg = config.framework.action_token_reasoner
        self.action_vocab_size = int(reasoner_cfg.get("vocab_size", 512))
        self.action_token_count = int(reasoner_cfg.get("token_count", 8))
        self.action_token_offset = int(reasoner_cfg.get("token_offset", 151669))
        self.prompt_template = str(
            reasoner_cfg.get(
                "prompt_template",
                "Task: {instruction}\nPredict the next robot action tokens.",
            )
        )
        self._validate_special_token_vocabulary()
        self._configure_trainable_scope(int(reasoner_cfg.get("train_last_n_layers", 2)))

    def _validate_special_token_vocabulary(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        probes = (0, self.action_vocab_size - 1)
        for code_id in probes:
            token = f"<robot_action_{code_id}>"
            actual = int(tokenizer.convert_tokens_to_ids(token))
            expected = self.action_token_offset + code_id
            if actual != expected:
                raise ValueError(f"{token} maps to {actual}, expected {expected}")

    def _configure_trainable_scope(self, last_n_layers: int) -> None:
        if last_n_layers < 0:
            raise ValueError("train_last_n_layers must be non-negative")
        model = self.qwen_vl_interface.model
        for parameter in model.parameters():
            parameter.requires_grad = False

        language_model = model.model.language_model
        if last_n_layers > len(language_model.layers):
            raise ValueError(
                f"requested {last_n_layers} trainable layers, model has {len(language_model.layers)}"
            )
        if last_n_layers:
            for layer in language_model.layers[-last_n_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        for parameter in language_model.norm.parameters():
            parameter.requires_grad = True
        # Qwen3 ties input embeddings and lm_head, so this trains both token input and output rows.
        for parameter in model.get_input_embeddings().parameters():
            parameter.requires_grad = True

    def _reasoner_instructions(self, examples: List[dict]) -> list[str]:
        return [self.prompt_template.format(instruction=example["lang"]) for example in examples]

    def forward(self, examples: List[dict] = None, **kwargs):
        if not examples:
            raise ValueError("examples must not be empty")
        images = [example["image"] for example in examples]
        instructions = self._reasoner_instructions(examples)
        solutions = [
            format_action_token_solution(
                example["action_token_ids"], vocab_size=self.action_vocab_size
            )
            for example in examples
        ]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=solutions,
        )
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise RuntimeError("action-token causal LM loss is missing or non-finite")
        return {"action_loss": outputs.loss, "action_token_ce_loss": outputs.loss}

    @torch.inference_mode()
    def predict_token_ids(self, examples: List[dict] = None) -> np.ndarray:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("examples must not be empty")
        images = [example["image"] for example in examples]
        instructions = self._reasoner_instructions(examples)
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
        )
        prompt_width = int(inputs["input_ids"].shape[1])
        allowed_ids = list(
            range(self.action_token_offset, self.action_token_offset + self.action_vocab_size)
        )

        def allowed_tokens(_batch_id: int, _input_ids: torch.Tensor) -> list[int]:
            return allowed_ids

        generated = self.qwen_vl_interface.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.action_token_count,
            min_new_tokens=self.action_token_count,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        token_ids = extract_generated_action_ids(
            generated,
            prompt_width=prompt_width,
            token_offset=self.action_token_offset,
            vocab_size=self.action_vocab_size,
            token_count=self.action_token_count,
        )
        return token_ids.cpu().numpy()

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        return {"action_token_ids": self.predict_token_ids(examples)}
