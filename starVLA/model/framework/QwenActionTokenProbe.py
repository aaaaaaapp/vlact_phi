"""Parallel diagnostic probe for predicting cached action-token sequences."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def last_valid_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the final non-padding position for either left or right padding."""
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden and attention_mask must have shapes [B,L,H] and [B,L]")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden and attention_mask batch/sequence dimensions do not match")
    positions = torch.arange(attention_mask.shape[1], device=hidden.device).unsqueeze(0)
    positions = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1)
    final_positions = positions.max(dim=1).values
    if (final_positions < 0).any():
        raise ValueError("each sample must contain at least one non-padding token")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, final_positions]


@FRAMEWORK_REGISTRY.register("QwenActionTokenProbe")
class QwenActionTokenProbe(baseframework):
    """Predict all action-code positions in parallel from one VLM prompt state."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=config)
        probe_cfg = config.framework.action_token_probe
        self.action_vocab_size = int(probe_cfg.get("vocab_size", 512))
        self.action_token_count = int(probe_cfg.get("token_count", 8))
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        probe_hidden_size = int(probe_cfg.get("hidden_size", hidden_size))
        dropout = float(probe_cfg.get("dropout", 0.0))
        self.prompt_template = str(
            probe_cfg.get(
                "prompt_template",
                "Task: {instruction}\nPredict the next robot action tokens.",
            )
        )
        self.token_probe = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, probe_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden_size, self.action_token_count * self.action_vocab_size),
        )
        self._configure_trainable_scope(int(probe_cfg.get("train_last_n_layers", 2)))

    def _configure_trainable_scope(self, last_n_layers: int) -> None:
        model = self.qwen_vl_interface.model
        for parameter in model.parameters():
            parameter.requires_grad = False
        language_model = model.model.language_model
        if not 0 <= last_n_layers <= len(language_model.layers):
            raise ValueError(
                f"train_last_n_layers must be in [0,{len(language_model.layers)}]"
            )
        for layer in language_model.layers[-last_n_layers:] if last_n_layers else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in language_model.norm.parameters():
            parameter.requires_grad = True
        for parameter in self.token_probe.parameters():
            parameter.requires_grad = True

    def _instructions(self, examples: List[dict]) -> list[str]:
        return [self.prompt_template.format(instruction=example["lang"]) for example in examples]

    def _logits(self, examples: List[dict]) -> torch.Tensor:
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=[example["image"] for example in examples],
            instructions=self._instructions(examples),
        )
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = last_valid_hidden(outputs.hidden_states[-1], inputs["attention_mask"])
        logits = self.token_probe(pooled)
        return logits.view(-1, self.action_token_count, self.action_vocab_size)

    def forward(self, examples: List[dict] = None, **kwargs):
        if not examples:
            raise ValueError("examples must not be empty")
        logits = self._logits(examples)
        targets = torch.as_tensor(
            np.stack([example["action_token_ids"] for example in examples]),
            device=logits.device,
            dtype=torch.long,
        )
        if targets.shape != logits.shape[:2]:
            raise ValueError(
                f"expected target shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}"
            )
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.action_vocab_size):
            raise ValueError("action token target is outside the configured vocabulary")
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return {"action_loss": loss, "action_token_ce_loss": loss}

    @torch.inference_mode()
    def predict_token_ids(self, examples: List[dict] = None) -> np.ndarray:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("examples must not be empty")
        return self._logits(examples).argmax(dim=-1).cpu().numpy()

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        return {"action_token_ids": self.predict_token_ids(examples)}
