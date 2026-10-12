"""Pi0.5-style action expert conditioned on discrete action and physics IDs."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import nn

from starVLA.model.framework.QwenPI_v4 import Qwen_PI_v4
from starVLA.model.modules.conditioning import DiscreteTokenConditioner
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _config_value(config, name: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    return default if value is None else value


@FRAMEWORK_REGISTRY.register("QwenPIv4PhysicalTokens")
class QwenPIv4PhysicalTokens(Qwen_PI_v4):
    """Append cached or predicted discrete conditions to the Pi expert prefix."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        condition_config = getattr(config.framework, "discrete_conditioning", None)
        if condition_config is None:
            raise ValueError("framework.discrete_conditioning is required")
        self.pi_discrete_conditioner = DiscreteTokenConditioner(
            hidden_dim=self.expert_hidden_dim,
            action_vocab_size=int(_config_value(condition_config, "action_vocab_size")),
            physics_vocab_size=int(_config_value(condition_config, "physics_vocab_size")),
            max_action_tokens=int(_config_value(condition_config, "max_action_tokens", 4)),
            max_physics_tokens=int(_config_value(condition_config, "max_physics_tokens", 256)),
            gate_init=float(_config_value(condition_config, "gate_init", 0.1)),
            action_gate_init=_config_value(condition_config, "action_gate_init"),
            physics_gate_init=_config_value(condition_config, "physics_gate_init"),
            action_condition_dropout=float(
                _config_value(condition_config, "action_condition_dropout", 0.0)
            ),
            physics_condition_dropout=float(
                _config_value(condition_config, "physics_condition_dropout", 0.2)
            ),
            physics_fusion_mode=str(
                _config_value(condition_config, "physics_fusion_mode", "concat")
            ),
            physics_group_size=int(
                _config_value(condition_config, "physics_group_size", 16)
            ),
            physics_condition_enabled=bool(
                _config_value(condition_config, "physics_condition_enabled", True)
            ),
            action_condition_enabled=bool(
                _config_value(condition_config, "action_condition_enabled", True)
            ),
        )
        self.discrete_conditioning_enabled = bool(
            _config_value(condition_config, "enabled", True)
        )
        self.physics_condition_enabled = bool(
            _config_value(condition_config, "physics_condition_enabled", True)
        )
        self.action_condition_enabled = bool(
            _config_value(condition_config, "action_condition_enabled", True)
        )
        self.pi_gate_bound_mode = str(_config_value(condition_config, "pi_gate_bound_mode", "none"))
        self.pi_gate_alpha_max = float(_config_value(condition_config, "pi_gate_alpha_max", 0.2))
        self.pi_residual_fusion_enabled = (
            self.pi_discrete_conditioner.physics_fusion_mode
            == "pi_multilayer_residual"
        )
        self.pi_separate_residual_fusion_enabled = (
            self.pi_discrete_conditioner.physics_fusion_mode
            == "pi_separate_multilayer_residual"
        )
        if getattr(self, "pi_residual_fusion_enabled", False):
            self.pi_discrete_layer_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(())) for _ in self.expert_layers]
            )
        if getattr(self, "pi_separate_residual_fusion_enabled", False):
            self.pi_action_layer_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(())) for _ in self.expert_layers]
            )
            self.pi_physics_layer_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(())) for _ in self.expert_layers]
            )
        self._pi_residual_token_count = 0
        self._pi_action_residual_token_count = 0
        self._pi_physics_residual_token_count = 0

    def set_discrete_conditioning_enabled(self, enabled: bool) -> None:
        self.discrete_conditioning_enabled = bool(enabled)

    def set_physics_condition_enabled(self, enabled: bool) -> None:
        self.physics_condition_enabled = bool(enabled)

    @staticmethod
    def _batch_token_ids(
        examples: List[dict], key: str, device: torch.device
    ) -> torch.Tensor | None:
        present = [key in example and example[key] is not None for example in examples]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"{key} must be present for every sample or none")
        values = np.stack([np.asarray(example[key], dtype=np.int64) for example in examples])
        if values.ndim != 2:
            raise ValueError(f"{key} must contain one-dimensional token sequences")
        return torch.from_numpy(values).to(device=device)

    def _condition_prefix_hidden(
        self,
        prefix_hidden: torch.Tensor,
        examples: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        self._pi_residual_token_count = 0
        self._pi_action_residual_token_count = 0
        self._pi_physics_residual_token_count = 0
        if not self.discrete_conditioning_enabled:
            return prefix_hidden
        if not examples:
            raise ValueError("discrete conditioning requires examples")
        action_ids = None
        if self.action_condition_enabled:
            action_ids = self._batch_token_ids(
                examples, "action_token_ids", prefix_hidden.device
            )
        if self.action_condition_enabled and action_ids is None:
            raise ValueError("Pi discrete conditioning requires action_token_ids")
        physics_ids = None
        if self.physics_condition_enabled:
            physics_ids = self._batch_token_ids(
                examples, "physics_token_ids", prefix_hidden.device
            )
            if physics_ids is None:
                raise ValueError(
                    "enabled Pi physics conditioning requires physics_token_ids"
                )
        return self._condition_prefix_hidden_from_ids(
            prefix_hidden, action_ids, physics_ids
        )

    def _condition_prefix_hidden_from_ids(
        self,
        prefix_hidden: torch.Tensor,
        action_ids: torch.Tensor | None,
        physics_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Condition an expert prefix from already materialized token IDs."""
        self._pi_residual_token_count = 0
        self._pi_action_residual_token_count = 0
        self._pi_physics_residual_token_count = 0
        if not self.discrete_conditioning_enabled:
            return prefix_hidden
        if not self.action_condition_enabled and not self.physics_condition_enabled:
            return prefix_hidden
        if getattr(self, "pi_separate_residual_fusion_enabled", False):
            action_tokens, physics_tokens = (
                self.pi_discrete_conditioner.build_token_groups(
                    action_ids, physics_ids
                )
            )
            token_groups = []
            if action_tokens is not None:
                self._pi_action_residual_token_count = action_tokens.shape[1]
                token_groups.append(action_tokens.to(prefix_hidden.dtype))
            if physics_tokens is not None:
                self._pi_physics_residual_token_count = physics_tokens.shape[1]
                token_groups.append(physics_tokens.to(prefix_hidden.dtype))
            conditioned = torch.cat((prefix_hidden, *token_groups), dim=1)
        else:
            conditioned = self.pi_discrete_conditioner(
                [prefix_hidden],
                action_token_ids=action_ids,
                physics_token_ids=physics_ids,
                enabled=True,
            )[0]
        if getattr(self, "pi_residual_fusion_enabled", False):
            self._pi_residual_token_count = conditioned.shape[1] - prefix_hidden.shape[1]
            if self._pi_residual_token_count <= 0:
                raise RuntimeError("Pi residual fusion produced no discrete memory")
        return conditioned

    def _split_expert_memories(
        self, prefix_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not getattr(self, "pi_separate_residual_fusion_enabled", False):
            return super()._split_expert_memories(prefix_hidden)
        action_count = self._pi_action_residual_token_count
        physics_count = self._pi_physics_residual_token_count
        residual_count = action_count + physics_count
        if residual_count == 0:
            return prefix_hidden, None, None
        if prefix_hidden.shape[1] <= residual_count:
            raise RuntimeError("separate discrete memories require a non-empty Pi prefix")
        split_at = prefix_hidden.shape[1] - residual_count
        action_end = split_at + action_count
        base = prefix_hidden[:, :split_at]
        action = prefix_hidden[:, split_at:action_end] if action_count else None
        physics = prefix_hidden[:, action_end:] if physics_count else None
        return base, action, physics

    def _split_expert_prefix(
        self, prefix_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not getattr(self, "pi_residual_fusion_enabled", False)
            or getattr(self, "_pi_residual_token_count", 0) == 0
        ):
            return prefix_hidden, None
        if prefix_hidden.shape[1] <= self._pi_residual_token_count:
            raise RuntimeError("discrete memory cannot consume the entire Pi prefix")
        split_at = prefix_hidden.shape[1] - self._pi_residual_token_count
        return prefix_hidden[:, :split_at], prefix_hidden[:, split_at:]

    def _expert_residual_gate(self, layer_index: int) -> torch.Tensor | None:
        gate = None
        if getattr(self, "pi_separate_residual_fusion_enabled", False):
            gate = self.pi_action_layer_gates[layer_index]
        elif getattr(self, "pi_residual_fusion_enabled", False):
            gate = self.pi_discrete_layer_gates[layer_index]
        if gate is None:
            return None
        return self._bound_pi_gate(gate)

    def _expert_physics_residual_gate(
        self, layer_index: int
    ) -> torch.Tensor | None:
        if not getattr(self, "pi_separate_residual_fusion_enabled", False):
            return None
        return self._bound_pi_gate(self.pi_physics_layer_gates[layer_index])

    def _bound_pi_gate(self, gate: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "pi_gate_bound_mode", "none")
        if mode == "tanh":
            return float(getattr(self, "pi_gate_alpha_max", 0.2)) * torch.tanh(gate)
        if mode == "sigmoid":
            return float(getattr(self, "pi_gate_alpha_max", 0.2)) * torch.sigmoid(gate)
        return gate
