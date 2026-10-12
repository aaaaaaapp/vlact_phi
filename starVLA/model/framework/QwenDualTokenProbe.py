"""Joint parallel probe for action and action-conditioned physics tokens."""

from __future__ import annotations

import hashlib
import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.QwenActionTokenProbe import last_valid_hidden
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def soft_action_context(logits: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    """Flatten differentiable expected action-code embeddings for all positions."""
    if logits.ndim != 3:
        raise ValueError(f"action logits must be [B,N,V], got {tuple(logits.shape)}")
    if logits.shape[-1] != embedding.num_embeddings:
        raise ValueError("action logits vocabulary does not match action embedding")
    probabilities = logits.float().softmax(dim=-1).to(embedding.weight.dtype)
    context = probabilities @ embedding.weight
    return context.flatten(start_dim=1)


def action_distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Token-averaged forward KL from a frozen action teacher."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher action logits must have equal shapes")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return per_token.mean() * temperature**2


@FRAMEWORK_REGISTRY.register("QwenDualTokenProbe")
class QwenDualTokenProbe(baseframework):
    """Predict action IDs and physics IDs with explicit soft action conditioning."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=config)
        probe_cfg = config.framework.dual_token_probe
        self.action_vocab_size = int(probe_cfg.get("action_vocab_size", 128))
        self.action_token_count = int(probe_cfg.get("action_token_count", 4))
        self.physics_vocab_size = int(probe_cfg.get("physics_vocab_size", 64))
        self.physics_token_count = int(probe_cfg.get("physics_token_count", 4))
        self.physics_loss_weight = float(probe_cfg.get("physics_loss_weight", 1.0))
        self.action_loss_weight = float(probe_cfg.get("action_loss_weight", 1.0))
        self.action_distill_weight = float(probe_cfg.get("action_distill_weight", 0.0))
        self.action_distill_temperature = float(
            probe_cfg.get("action_distill_temperature", 1.0)
        )
        self.enable_action_teacher = bool(probe_cfg.get("enable_action_teacher", False))
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        probe_hidden_size = int(probe_cfg.get("hidden_size", hidden_size))
        condition_dim = int(probe_cfg.get("action_condition_dim", 256))
        dropout = float(probe_cfg.get("dropout", 0.0))
        self.prompt_template = str(
            probe_cfg.get(
                "prompt_template",
                "Task: {instruction}\nPredict action and physical dynamics tokens.",
            )
        )
        self.action_token_probe = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, probe_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden_size, self.action_token_count * self.action_vocab_size),
        )
        self.action_condition_embedding = nn.Embedding(self.action_vocab_size, condition_dim)
        physics_input_size = hidden_size + self.action_token_count * condition_dim
        self.physics_token_probe = nn.Sequential(
            nn.LayerNorm(physics_input_size),
            nn.Linear(physics_input_size, probe_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden_size, self.physics_token_count * self.physics_vocab_size),
        )
        if self.enable_action_teacher:
            self.teacher_qwen_vl_interface = copy.deepcopy(self.qwen_vl_interface)
            self.teacher_action_token_probe = copy.deepcopy(self.action_token_probe)
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
        for module in (
            self.action_token_probe,
            self.action_condition_embedding,
            self.physics_token_probe,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if self.enable_action_teacher:
            for module in (
                self.teacher_qwen_vl_interface,
                self.teacher_action_token_probe,
            ):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def initialize_from_action_probe(
        self, checkpoint_path: str | Path, expected_sha256: str | None = None
    ) -> dict[str, int | str]:
        """Load the mature action probe and its tuned VLM layers by exact name mapping."""
        path = Path(checkpoint_path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"action probe checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}"
            )
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        parameters = dict(self.named_parameters())
        loaded = []
        for source_name, value in state.items():
            target_name = source_name
            if source_name.startswith("token_probe."):
                target_name = "action_token_probe." + source_name[len("token_probe.") :]
            if target_name not in parameters:
                raise ValueError(f"action probe initialization has unexpected key: {source_name}")
            target = parameters[target_name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for {source_name}: {tuple(value.shape)} vs {tuple(target.shape)}"
                )
            target.data.copy_(value.to(dtype=target.dtype))
            loaded.append(target_name)
        expected = {
            name
            for name, parameter in self.named_parameters()
            if (
                name.startswith("qwen_vl_interface.") and parameter.requires_grad
            )
            or name.startswith("action_token_probe.")
        }
        if set(loaded) != expected:
            raise ValueError(
                "action probe initialization key mismatch: "
                f"missing={sorted(expected - set(loaded))[:8]}, "
                f"extra={sorted(set(loaded) - expected)[:8]}"
            )
        if self.enable_action_teacher:
            self.teacher_qwen_vl_interface.load_state_dict(
                self.qwen_vl_interface.state_dict(), strict=True
            )
            self.teacher_action_token_probe.load_state_dict(
                self.action_token_probe.state_dict(), strict=True
            )
        return {"path": str(path), "sha256": digest, "loaded_tensors": len(loaded)}

    def _instructions(self, examples: List[dict]) -> list[str]:
        return [self.prompt_template.format(instruction=example["lang"]) for example in examples]

    def _logits(self, examples: List[dict]) -> tuple[torch.Tensor, torch.Tensor]:
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
        action_logits = self.action_token_probe(pooled).view(
            -1, self.action_token_count, self.action_vocab_size
        )
        action_context = soft_action_context(action_logits, self.action_condition_embedding)
        physics_logits = self.physics_token_probe(
            torch.cat((pooled, action_context.to(pooled.dtype)), dim=-1)
        ).view(-1, self.physics_token_count, self.physics_vocab_size)
        return action_logits, physics_logits

    @torch.no_grad()
    def _teacher_action_logits(self, examples: List[dict]) -> torch.Tensor:
        if not self.enable_action_teacher:
            raise RuntimeError("action teacher is disabled")
        self.teacher_qwen_vl_interface.eval()
        self.teacher_action_token_probe.eval()
        inputs = self.teacher_qwen_vl_interface.build_qwenvl_inputs(
            images=[example["image"] for example in examples],
            instructions=self._instructions(examples),
        )
        outputs = self.teacher_qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = last_valid_hidden(outputs.hidden_states[-1], inputs["attention_mask"])
        return self.teacher_action_token_probe(pooled).view(
            -1, self.action_token_count, self.action_vocab_size
        )

    @staticmethod
    def _targets(
        examples: List[dict], key: str, shape: tuple[int, int], vocab_size: int, device
    ) -> torch.Tensor:
        targets = torch.as_tensor(
            np.stack([example[key] for example in examples]),
            device=device,
            dtype=torch.long,
        )
        if tuple(targets.shape) != shape:
            raise ValueError(f"expected {key} shape {shape}, got {tuple(targets.shape)}")
        if targets.numel() and (targets.min() < 0 or targets.max() >= vocab_size):
            raise ValueError(f"{key} target is outside the configured vocabulary")
        return targets

    def forward(self, examples: List[dict] = None, **kwargs):
        if not examples:
            raise ValueError("examples must not be empty")
        action_logits, physics_logits = self._logits(examples)
        action_targets = self._targets(
            examples,
            "action_token_ids",
            tuple(action_logits.shape[:2]),
            self.action_vocab_size,
            action_logits.device,
        )
        physics_targets = self._targets(
            examples,
            "physics_token_ids",
            tuple(physics_logits.shape[:2]),
            self.physics_vocab_size,
            physics_logits.device,
        )
        action_loss = F.cross_entropy(action_logits.flatten(0, 1), action_targets.flatten())
        physics_loss = F.cross_entropy(physics_logits.flatten(0, 1), physics_targets.flatten())
        distill_loss = action_logits.new_zeros(())
        if self.action_distill_weight:
            if not self.enable_action_teacher:
                raise RuntimeError("action_distill_weight requires enable_action_teacher=true")
            distill_loss = action_distillation_loss(
                action_logits,
                self._teacher_action_logits(examples),
                self.action_distill_temperature,
            )
        total = (
            self.action_loss_weight * action_loss
            + self.physics_loss_weight * physics_loss
            + self.action_distill_weight * distill_loss
        )
        result = {
            "action_loss": total,
            "action_token_ce_loss": action_loss,
            "physics_token_ce_loss": physics_loss,
        }
        if self.action_distill_weight:
            result["action_distill_loss"] = distill_loss
        return result

    @torch.inference_mode()
    def predict_token_ids(
        self, examples: List[dict] = None, return_physics_logits: bool = False
    ) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("examples must not be empty")
        action_logits, physics_logits = self._logits(examples)
        result = {
            "action_token_ids": action_logits.argmax(dim=-1).cpu().numpy(),
            "physics_token_ids": physics_logits.argmax(dim=-1).cpu().numpy(),
        }
        if return_physics_logits:
            result["physics_logits"] = physics_logits.float().cpu().numpy()
        return result

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        return self.predict_token_ids(examples)
