"""Joint action-token, physics-token, and continuous Pi action framework."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.QwenActionTokenProbe import last_valid_hidden
from starVLA.model.framework.QwenDualTokenProbe import soft_action_context
from starVLA.model.framework.QwenPIv4PhysicalTokens import QwenPIv4PhysicalTokens
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPIv4JointTokens")
class QwenPIv4JointTokens(QwenPIv4PhysicalTokens):
    """Predict both discrete streams and train the continuous Pi head jointly."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        joint = config.framework.joint_token_prediction
        self.joint_action_vocab_size = int(joint.get("action_vocab_size", 128))
        self.joint_action_token_count = int(joint.get("action_token_count", 4))
        self.joint_physics_vocab_size = int(joint.get("physics_vocab_size", 256))
        self.joint_physics_token_count = int(joint.get("physics_token_count", 256))
        self.joint_condition_source = str(joint.get("condition_source", "cached"))
        valid_condition_sources = {
            "cached",
            "predicted",
            "soft",
            "predicted_action_cached_physics",
            "cached_action_predicted_physics",
        }
        if self.joint_condition_source not in valid_condition_sources:
            raise ValueError(
                "joint condition_source must be one of "
                f"{sorted(valid_condition_sources)}, got {self.joint_condition_source!r}"
            )
        self.joint_soft_temperature = float(joint.get("soft_temperature", 1.0))
        if self.joint_soft_temperature <= 0.0:
            raise ValueError("joint soft_temperature must be positive")
        self.action_token_loss_weight = float(joint.get("action_loss_weight", 1e-3))
        self.physics_token_loss_weight = float(joint.get("physics_loss_weight", 1e-4))
        # Paired Pure-PI teacher loss; the teacher uses the same backbone with
        # discrete conditioning disabled and is evaluated without gradients.
        self.action_distill_weight_start = float(joint.get("action_distill_weight_start", 0.0))
        self.action_distill_weight_end = float(joint.get("action_distill_weight_end", 0.0))
        self.action_distill_warmup_steps = int(joint.get("action_distill_warmup_steps", 20000))
        self._distill_step = 0
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        probe_hidden = int(joint.get("hidden_size", 1024))
        condition_dim = int(joint.get("action_condition_dim", 256))
        dropout = float(joint.get("dropout", 0.0))
        self.action_token_probe = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, probe_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                probe_hidden,
                self.joint_action_token_count * self.joint_action_vocab_size,
            ),
        )
        self.action_condition_embedding = nn.Embedding(
            self.joint_action_vocab_size, condition_dim
        )
        physics_input = hidden_size + self.joint_action_token_count * condition_dim
        self.physics_token_probe = nn.Sequential(
            nn.LayerNorm(physics_input),
            nn.Linear(physics_input, probe_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                probe_hidden,
                self.joint_physics_token_count * self.joint_physics_vocab_size,
            ),
        )
        self._joint_token_losses: tuple[torch.Tensor, torch.Tensor] | None = None

    def initialize_joint_token_predictor(
        self, checkpoint_path: str | Path, expected_sha256: str | None = None
    ) -> dict[str, int | str]:
        """Load only D31 predictor heads, preserving the mature policy VLM."""
        path = Path(checkpoint_path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"joint token predictor SHA256 mismatch: expected {expected_sha256}, got {digest}"
            )
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        prefixes = (
            "action_token_probe.",
            "action_condition_embedding.",
            "physics_token_probe.",
        )
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        expected = {
            key for key in self.state_dict() if key.startswith(prefixes)
        }
        if set(selected) != expected:
            raise ValueError(
                "joint predictor key mismatch: "
                f"missing={sorted(expected - set(selected))[:8]}, "
                f"extra={sorted(set(selected) - expected)[:8]}"
            )
        result = self.load_state_dict(selected, strict=False)
        unexpected = list(result.unexpected_keys)
        missing_outside_predictor = [
            key for key in result.missing_keys if key.startswith(prefixes)
        ]
        if unexpected or missing_outside_predictor:
            raise ValueError(
                "failed to load joint predictor exactly: "
                f"missing={missing_outside_predictor[:8]}, unexpected={unexpected[:8]}"
            )
        return {"path": str(path), "sha256": digest, "loaded_tensors": len(selected)}

    def initialize_policy_extension(
        self, checkpoint_path: str | Path, expected_sha256: str | None = None
    ) -> dict[str, int | str]:
        """Warm-start the conditioner and independent residual gates."""
        path = Path(checkpoint_path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(
                f"policy extension SHA256 mismatch: expected {expected_sha256}, got {digest}"
            )
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        prefixes = (
            "pi_discrete_conditioner.",
            "pi_action_layer_gates.",
            "pi_physics_layer_gates.",
        )
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        expected = {key for key in self.state_dict() if key.startswith(prefixes)}
        if set(selected) != expected:
            raise ValueError(
                "policy extension key mismatch: "
                f"missing={sorted(expected - set(selected))[:8]}, "
                f"extra={sorted(set(selected) - expected)[:8]}"
            )
        self.load_state_dict(selected, strict=False)
        return {"path": str(path), "sha256": digest, "loaded_tensors": len(selected)}

    def _joint_logits(
        self, last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = last_valid_hidden(last_hidden, attention_mask)
        probe_dtype = next(self.action_token_probe.parameters()).dtype
        pooled = pooled.to(dtype=probe_dtype)
        action_logits = self.action_token_probe(pooled).view(
            -1, self.joint_action_token_count, self.joint_action_vocab_size
        )
        action_context = soft_action_context(
            action_logits, self.action_condition_embedding
        )
        physics_logits = self.physics_token_probe(
            torch.cat((pooled, action_context.to(pooled.dtype)), dim=-1)
        ).view(
            -1, self.joint_physics_token_count, self.joint_physics_vocab_size
        )
        return action_logits, physics_logits

    @staticmethod
    def _targets(examples: List[dict], key: str, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([example[key] for example in examples]),
            device=device,
            dtype=torch.long,
        )

    def _conditioning_token_ids(
        self,
        examples: List[dict],
        action_logits: torch.Tensor,
        physics_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Select cached or predicted IDs independently for diagnostic inference."""
        predicted_action = self.joint_condition_source in {
            "predicted",
            "predicted_action_cached_physics",
        }
        predicted_physics = self.joint_condition_source in {
            "predicted",
            "cached_action_predicted_physics",
        }
        action_ids = (
            action_logits.argmax(dim=-1)
            if predicted_action
            else self._targets(examples, "action_token_ids", action_logits.device)
        )
        if not self.physics_condition_enabled:
            return action_ids, None
        physics_ids = (
            physics_logits.argmax(dim=-1)
            if predicted_physics
            else self._targets(examples, "physics_token_ids", physics_logits.device)
        )
        return action_ids, physics_ids

    def _condition_prefix_hidden_soft(
        self,
        prefix_hidden: torch.Tensor,
        action_logits: torch.Tensor,
        physics_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not getattr(self, "pi_separate_residual_fusion_enabled", False):
            raise ValueError(
                "soft joint conditioning currently requires "
                "pi_separate_multilayer_residual"
            )
        temperature = self.joint_soft_temperature
        action_probabilities = torch.softmax(
            action_logits.float() / temperature, dim=-1
        )
        physics_probabilities = (
            torch.softmax(physics_logits.float() / temperature, dim=-1)
            if self.physics_condition_enabled
            else None
        )
        action_tokens, physics_tokens = (
            self.pi_discrete_conditioner.build_soft_token_groups(
                action_probabilities, physics_probabilities
            )
        )
        self._pi_action_residual_token_count = (
            0 if action_tokens is None else action_tokens.shape[1]
        )
        self._pi_physics_residual_token_count = (
            0 if physics_tokens is None else physics_tokens.shape[1]
        )
        token_groups = [
            tokens.to(prefix_hidden.dtype)
            for tokens in (action_tokens, physics_tokens)
            if tokens is not None
        ]
        return torch.cat((prefix_hidden, *token_groups), dim=1)

    def _encode_qwen_hidden(self, batch_images, instructions, examples=None):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        prefix_dtype = next(
            (parameter.dtype for parameter in self.prefix_proj.parameters()),
            last_hidden.dtype,
        )
        last_hidden = last_hidden.to(dtype=prefix_dtype)
        prefix_hidden = self.prefix_proj(last_hidden)
        self._last_pure_prefix_hidden = prefix_hidden
        if not self.training and self.joint_condition_source == "cached":
            self._joint_token_losses = None
            prefix_hidden = self._condition_prefix_hidden(prefix_hidden, examples)
            return last_hidden, prefix_hidden, qwen_inputs

        action_logits, physics_logits = self._joint_logits(
            last_hidden, qwen_inputs["attention_mask"]
        )
        self._joint_token_losses = None
        if examples and all(
            "action_token_ids" in example and "physics_token_ids" in example
            for example in examples
        ):
            action_targets = self._targets(examples, "action_token_ids", action_logits.device)
            physics_targets = self._targets(
                examples, "physics_token_ids", physics_logits.device
            )
            if tuple(action_targets.shape) != tuple(action_logits.shape[:2]):
                raise ValueError("joint action-token target shape does not match logits")
            if tuple(physics_targets.shape) != tuple(physics_logits.shape[:2]):
                raise ValueError("joint physics-token target shape does not match logits")
            self._joint_token_losses = (
                F.cross_entropy(action_logits.flatten(0, 1), action_targets.flatten()),
                F.cross_entropy(physics_logits.flatten(0, 1), physics_targets.flatten()),
            )

        if not self.discrete_conditioning_enabled:
            return last_hidden, prefix_hidden, qwen_inputs
        if self.joint_condition_source == "soft":
            prefix_hidden = self._condition_prefix_hidden_soft(
                prefix_hidden, action_logits, physics_logits
            )
        else:
            action_ids, physics_ids = self._conditioning_token_ids(
                examples, action_logits, physics_logits
            )
            prefix_hidden = self._condition_prefix_hidden_from_ids(
                prefix_hidden, action_ids, physics_ids
            )
        return last_hidden, prefix_hidden, qwen_inputs

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        output = super().forward(examples=examples, **kwargs)
        if self._joint_token_losses is None:
            raise ValueError("joint training requires cached action and physics targets")
        action_ce, physics_ce = self._joint_token_losses
        flow_loss = output["action_loss"]
        output.update(
            {
                "action_loss": flow_loss
                + self.action_token_loss_weight * action_ce
                + self.physics_token_loss_weight * physics_ce,
                "action_dit_loss": flow_loss.detach(),
                "action_token_ce_loss": action_ce,
                "physics_token_ce_loss": physics_ce,
            }
        )
        return output

    @torch.inference_mode()
    def predict_token_ids(self, examples: List[dict]) -> dict[str, np.ndarray]:
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = self._inject_state_into_instruction(instructions, states)
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        outputs = self.qwen_vl_interface(
            **inputs, output_attentions=False, output_hidden_states=True, return_dict=True
        )
        action_logits, physics_logits = self._joint_logits(
            outputs.hidden_states[-1], inputs["attention_mask"]
        )
        return {
            "action_token_ids": action_logits.argmax(dim=-1).cpu().numpy(),
            "physics_token_ids": physics_logits.argmax(dim=-1).cpu().numpy(),
        }
