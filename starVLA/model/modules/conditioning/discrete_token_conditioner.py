"""Discrete action/physics token conditioning for VLM-to-action prefixes."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.distributed as dist
import torch.nn as nn


class DiscreteTokenConditioner(nn.Module):
    """Embed token IDs and append them to each layer-wise VLM prefix.

    Disabling conditioning returns the original prefix tensors unchanged. This
    explicit bypass is required for an exact baseline: appending zero-valued
    tokens would still alter a cross-attention softmax denominator.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_vocab_size: int,
        physics_vocab_size: int = 0,
        max_action_tokens: int = 32,
        max_physics_tokens: int = 32,
        gate_init: float = 0.1,
        action_gate_init: float | None = None,
        physics_gate_init: float | None = None,
        action_condition_dropout: float = 0.0,
        physics_condition_dropout: float = 0.0,
        physics_fusion_mode: str = "concat",
        physics_fusion_dim: int = 256,
        physics_fusion_heads: int = 8,
        physics_condition_enabled: bool = True,
        action_condition_enabled: bool = True,
        physics_group_size: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or action_vocab_size < 2:
            raise ValueError("hidden_dim must be positive and action_vocab_size at least 2")
        if physics_vocab_size == 1 or physics_vocab_size < 0:
            raise ValueError("physics_vocab_size must be zero or at least 2")
        self.hidden_dim = int(hidden_dim)
        self.action_vocab_size = int(action_vocab_size)
        self.physics_vocab_size = int(physics_vocab_size)
        self.max_action_tokens = int(max_action_tokens)
        self.max_physics_tokens = int(max_physics_tokens)
        for name, probability in (
            ("action_condition_dropout", action_condition_dropout),
            ("physics_condition_dropout", physics_condition_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        self.action_condition_dropout = float(action_condition_dropout)
        self.physics_condition_dropout = float(physics_condition_dropout)
        if physics_fusion_mode not in (
            "concat",
            "grouped_concat",
            "pi_multilayer_residual",
            "pi_separate_multilayer_residual",
            "residual_cross_attention",
            "dit_multilayer_cross_attention",
        ):
            raise ValueError(
                "physics_fusion_mode must be 'concat', 'grouped_concat', "
                "'pi_multilayer_residual', 'pi_separate_multilayer_residual', "
                "'residual_cross_attention', or 'dit_multilayer_cross_attention'"
            )
        if physics_fusion_mode != "concat" and not physics_vocab_size:
            raise ValueError(f"{physics_fusion_mode} requires physics_vocab_size")
        if physics_fusion_dim < 1 or physics_fusion_dim % physics_fusion_heads:
            raise ValueError("physics_fusion_dim must be positive and divisible by its heads")
        self.physics_fusion_mode = physics_fusion_mode
        if int(physics_group_size) < 1:
            raise ValueError("physics_group_size must be positive")
        self.physics_group_size = int(physics_group_size)
        self.physics_condition_enabled = bool(physics_condition_enabled)
        self.action_condition_enabled = bool(action_condition_enabled)
        action_gate_init = gate_init if action_gate_init is None else action_gate_init
        physics_gate_init = gate_init if physics_gate_init is None else physics_gate_init

        self.action_embedding = nn.Embedding(action_vocab_size, hidden_dim)
        self.action_position = nn.Embedding(max_action_tokens, hidden_dim)
        self.action_type = nn.Parameter(torch.zeros(hidden_dim))
        self.action_gate = nn.Parameter(torch.tensor(float(action_gate_init)))
        prefix_uses_physics = (
            physics_vocab_size
            and physics_fusion_mode != "dit_multilayer_cross_attention"
        )
        if prefix_uses_physics:
            self.physics_embedding = nn.Embedding(physics_vocab_size, hidden_dim)
            self.physics_position = nn.Embedding(max_physics_tokens, hidden_dim)
            self.physics_type = nn.Parameter(torch.zeros(hidden_dim))
            self.physics_gate = nn.Parameter(torch.tensor(float(physics_gate_init)))
            if physics_fusion_mode == "residual_cross_attention":
                self.physics_query_norm = nn.LayerNorm(hidden_dim)
                self.physics_query_proj = nn.Linear(hidden_dim, physics_fusion_dim)
                self.physics_key_value_proj = nn.Linear(hidden_dim, physics_fusion_dim)
                self.physics_cross_attention = nn.MultiheadAttention(
                    physics_fusion_dim,
                    physics_fusion_heads,
                    batch_first=True,
                )
                self.physics_output_proj = nn.Linear(physics_fusion_dim, hidden_dim)
            else:
                self.physics_query_norm = None
                self.physics_query_proj = None
                self.physics_key_value_proj = None
                self.physics_cross_attention = None
                self.physics_output_proj = None
        else:
            self.physics_embedding = None
            self.physics_position = None
            self.register_parameter("physics_type", None)
            self.register_parameter("physics_gate", None)
            self.physics_query_norm = None
            self.physics_query_proj = None
            self.physics_key_value_proj = None
            self.physics_cross_attention = None
            self.physics_output_proj = None
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_embedding.weight, std=self.hidden_dim**-0.5)
        nn.init.normal_(self.action_position.weight, std=0.02)
        if self.physics_embedding is not None:
            nn.init.normal_(self.physics_embedding.weight, std=self.hidden_dim**-0.5)
            nn.init.normal_(self.physics_position.weight, std=0.02)
        if self.physics_output_proj is not None:
            nn.init.zeros_(self.physics_output_proj.weight)
            nn.init.zeros_(self.physics_output_proj.bias)

    @torch.no_grad()
    def _initialize_code_embeddings(
        self,
        embedding: nn.Embedding,
        codebook: torch.Tensor,
        *,
        name: str,
    ) -> None:
        """Initialize an ID embedding while preserving code geometry."""
        codebook = torch.as_tensor(codebook).detach().float().cpu()
        expected_rows = embedding.num_embeddings
        if codebook.ndim != 2 or codebook.shape[0] != expected_rows:
            raise ValueError(
                f"{name} codebook must have shape "
                f"[{expected_rows}, D], got {tuple(codebook.shape)}"
            )
        if not torch.isfinite(codebook).all():
            raise ValueError(f"{name} codebook contains non-finite values")
        code_dim = int(codebook.shape[1])
        if code_dim < 1 or self.hidden_dim % code_dim:
            raise ValueError(
                f"condition hidden dim {self.hidden_dim} must be divisible by "
                f"{name} code dim {code_dim}"
            )
        repeats = self.hidden_dim // code_dim
        expanded = codebook.repeat(1, repeats) / math.sqrt(repeats)
        embedding.weight.copy_(
            expanded.to(
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
        )

    def initialize_action_code_embeddings(self, codebook: torch.Tensor) -> None:
        self._initialize_code_embeddings(
            self.action_embedding, codebook, name="action"
        )

    def initialize_physics_code_embeddings(self, codebook: torch.Tensor) -> None:
        if self.physics_embedding is None:
            raise ValueError("physics embeddings are not enabled")
        self._initialize_code_embeddings(
            self.physics_embedding, codebook, name="physics"
        )

    @staticmethod
    def _validate_ids(ids: torch.Tensor, *, name: str, vocab_size: int, maximum: int) -> None:
        if ids.ndim != 2:
            raise ValueError(f"{name} token IDs must be [B, N], got {tuple(ids.shape)}")
        if ids.shape[1] > maximum:
            raise ValueError(f"{name} token count {ids.shape[1]} exceeds configured maximum {maximum}")
        if ids.numel() and (ids.min() < 0 or ids.max() >= vocab_size):
            raise ValueError(f"{name} token ID is outside [0, {vocab_size})")

    def _embed_action(self, ids: torch.Tensor) -> torch.Tensor:
        self._validate_ids(
            ids,
            name="action",
            vocab_size=self.action_vocab_size,
            maximum=self.max_action_tokens,
        )
        positions = torch.arange(ids.shape[1], device=ids.device)
        tokens = self.action_embedding(ids) + self.action_position(positions) + self.action_type
        return self.output_norm(tokens) * self.action_gate.to(tokens.dtype)

    @staticmethod
    def _validate_probabilities(
        probabilities: torch.Tensor,
        *,
        name: str,
        vocab_size: int,
        maximum: int,
    ) -> None:
        if probabilities.ndim != 3:
            raise ValueError(
                f"{name} token probabilities must be [B, N, V], "
                f"got {tuple(probabilities.shape)}"
            )
        if probabilities.shape[1] > maximum:
            raise ValueError(
                f"{name} token count {probabilities.shape[1]} exceeds "
                f"configured maximum {maximum}"
            )
        if probabilities.shape[2] != vocab_size:
            raise ValueError(
                f"{name} probability vocabulary must be {vocab_size}, "
                f"got {probabilities.shape[2]}"
            )
        if not torch.isfinite(probabilities).all():
            raise ValueError(f"{name} token probabilities contain non-finite values")

    def _embed_action_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        self._validate_probabilities(
            probabilities,
            name="action",
            vocab_size=self.action_vocab_size,
            maximum=self.max_action_tokens,
        )
        weights = self.action_embedding.weight
        expected = probabilities.to(weights.dtype) @ weights
        positions = torch.arange(probabilities.shape[1], device=probabilities.device)
        tokens = expected + self.action_position(positions) + self.action_type
        return self.output_norm(tokens) * self.action_gate.to(tokens.dtype)

    def _embed_physics(self, ids: torch.Tensor, *, apply_gate: bool = True) -> torch.Tensor:
        if self.physics_embedding is None or self.physics_position is None:
            raise ValueError("physics token IDs were supplied but physics conditioning is disabled")
        self._validate_ids(
            ids,
            name="physics",
            vocab_size=self.physics_vocab_size,
            maximum=self.max_physics_tokens,
        )
        positions = torch.arange(ids.shape[1], device=ids.device)
        tokens = self.physics_embedding(ids) + self.physics_position(positions) + self.physics_type
        tokens = self.output_norm(tokens)
        return tokens * self.physics_gate.to(tokens.dtype) if apply_gate else tokens

    def _embed_physics_probabilities(
        self, probabilities: torch.Tensor, *, apply_gate: bool = True
    ) -> torch.Tensor:
        if self.physics_embedding is None or self.physics_position is None:
            raise ValueError(
                "physics token probabilities were supplied but physics conditioning is disabled"
            )
        self._validate_probabilities(
            probabilities,
            name="physics",
            vocab_size=self.physics_vocab_size,
            maximum=self.max_physics_tokens,
        )
        weights = self.physics_embedding.weight
        expected = probabilities.to(weights.dtype) @ weights
        positions = torch.arange(probabilities.shape[1], device=probabilities.device)
        tokens = expected + self.physics_position(positions) + self.physics_type
        tokens = self.output_norm(tokens)
        return tokens * self.physics_gate.to(tokens.dtype) if apply_gate else tokens

    def _pool_physics_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.physics_fusion_mode not in (
            "grouped_concat",
            "pi_multilayer_residual",
            "pi_separate_multilayer_residual",
        ):
            return tokens
        token_count = tokens.shape[1]
        if token_count % self.physics_group_size:
            raise ValueError(
                f"physics token count {token_count} must be divisible by "
                f"physics_group_size={self.physics_group_size}"
            )
        return tokens.reshape(
            tokens.shape[0],
            token_count // self.physics_group_size,
            self.physics_group_size,
            self.hidden_dim,
        ).mean(dim=2)

    def _drop_group(self, probability: float) -> bool:
        if not self.training or probability <= 0.0:
            return False
        draw = torch.rand((), device=self.action_gate.device)
        if dist.is_available() and dist.is_initialized():
            # Entire token groups change the set of parameters participating in
            # backward. All ranks must therefore take the same branch.
            dist.broadcast(draw, src=0)
        return bool(draw < probability)

    def _drop_decisions(self) -> tuple[bool, bool]:
        drop_action = self._drop_group(self.action_condition_dropout)
        drop_physics = self._drop_group(self.physics_condition_dropout)
        return bool(drop_action), bool(drop_physics)

    def build_token_groups(
        self,
        action_token_ids: torch.Tensor | None,
        physics_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        drop_action, drop_physics = self._drop_decisions()
        action_tokens = None
        physics_tokens = None
        if self.action_condition_enabled and action_token_ids is not None and not drop_action:
            action_tokens = self._embed_action(action_token_ids.long())
        if self.physics_condition_enabled and physics_token_ids is not None and not drop_physics:
            physics_tokens = self._pool_physics_tokens(
                self._embed_physics(physics_token_ids.long())
            )
        if action_tokens is None and physics_tokens is None:
            # When both independent draws drop their group, retain action tokens.
            # This also preserves the action-token pathway in every training batch.
            if self.action_condition_enabled and action_token_ids is not None:
                action_tokens = self._embed_action(action_token_ids.long())
            elif self.physics_condition_enabled and physics_token_ids is not None:
                physics_tokens = self._pool_physics_tokens(
                    self._embed_physics(physics_token_ids.long())
                )
            else:
                raise ValueError("at least one action or physics token sequence is required")
        groups = [tokens for tokens in (action_tokens, physics_tokens) if tokens is not None]
        batch_sizes = {tokens.shape[0] for tokens in groups}
        if len(batch_sizes) != 1:
            raise ValueError("action and physics token batch sizes must match")
        return action_tokens, physics_tokens

    def build_soft_token_groups(
        self,
        action_token_probabilities: torch.Tensor | None,
        physics_token_probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build condition memories from categorical probability distributions."""
        drop_action, drop_physics = self._drop_decisions()
        action_tokens = None
        physics_tokens = None
        if self.action_condition_enabled and action_token_probabilities is not None and not drop_action:
            action_tokens = self._embed_action_probabilities(
                action_token_probabilities
            )
        if self.physics_condition_enabled and physics_token_probabilities is not None and not drop_physics:
            physics_tokens = self._pool_physics_tokens(
                self._embed_physics_probabilities(physics_token_probabilities)
            )
        if action_tokens is None and physics_tokens is None:
            if self.action_condition_enabled and action_token_probabilities is not None:
                action_tokens = self._embed_action_probabilities(
                    action_token_probabilities
                )
            elif self.physics_condition_enabled and physics_token_probabilities is not None:
                physics_tokens = self._pool_physics_tokens(
                    self._embed_physics_probabilities(physics_token_probabilities)
                )
            else:
                raise ValueError(
                    "at least one action or physics token distribution is required"
                )
        groups = [tokens for tokens in (action_tokens, physics_tokens) if tokens is not None]
        if len({tokens.shape[0] for tokens in groups}) != 1:
            raise ValueError("action and physics token batch sizes must match")
        return action_tokens, physics_tokens

    def build_tokens(
        self,
        action_token_ids: torch.Tensor | None,
        physics_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_tokens, physics_tokens = self.build_token_groups(
            action_token_ids, physics_token_ids
        )
        token_groups = [
            tokens for tokens in (action_tokens, physics_tokens) if tokens is not None
        ]
        return torch.cat(token_groups, dim=1)

    def _residual_cross_attention(
        self,
        prefix: torch.Tensor,
        physics_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if any(
            module is None
            for module in (
                self.physics_query_norm,
                self.physics_query_proj,
                self.physics_key_value_proj,
                self.physics_cross_attention,
                self.physics_output_proj,
            )
        ):
            raise RuntimeError("residual physics fusion modules are not initialized")
        physics_tokens = self._embed_physics(physics_token_ids.long(), apply_gate=False)
        query = self.physics_query_proj(self.physics_query_norm(prefix))
        key_value = self.physics_key_value_proj(physics_tokens)
        residual, _ = self.physics_cross_attention(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        residual = self.physics_output_proj(residual).to(prefix.dtype)
        return prefix + residual * self.physics_gate.to(prefix.dtype)

    def forward(
        self,
        vl_embs_list: Sequence[torch.Tensor],
        *,
        action_token_ids: torch.Tensor | None = None,
        physics_token_ids: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> list[torch.Tensor]:
        if not vl_embs_list:
            raise ValueError("vl_embs_list must contain at least one layer")
        if not enabled:
            return list(vl_embs_list)
        if self.physics_fusion_mode == "dit_multilayer_cross_attention":
            if action_token_ids is None:
                raise ValueError(
                    "DiT physics fusion keeps action tokens in the VLM prefix and "
                    "therefore requires action_token_ids"
                )
            drop_action = self._drop_group(self.action_condition_dropout)
            if drop_action:
                return list(vl_embs_list)
            action_tokens = self._embed_action(action_token_ids.long())
            return [
                torch.cat((prefix, action_tokens.to(prefix.dtype)), dim=1)
                for prefix in vl_embs_list
            ]
        if self.physics_fusion_mode == "residual_cross_attention":
            drop_action, drop_physics = self._drop_decisions()
            if drop_action and drop_physics and action_token_ids is not None:
                drop_action = False
            action_tokens = None
            if action_token_ids is not None and not drop_action:
                action_tokens = self._embed_action(action_token_ids.long())
            conditioned = []
            for layer_index, prefix in enumerate(vl_embs_list):
                if prefix.ndim != 3 or prefix.shape[-1] != self.hidden_dim:
                    raise ValueError(
                        f"layer {layer_index} prefix must be [B, S, {self.hidden_dim}], "
                        f"got {tuple(prefix.shape)}"
                    )
                base = (
                    torch.cat((prefix, action_tokens.to(prefix.dtype)), dim=1)
                    if action_tokens is not None
                    else prefix
                )
                if (
                    self.physics_condition_enabled
                    and physics_token_ids is not None
                    and not drop_physics
                ):
                    base = self._residual_cross_attention(base, physics_token_ids)
                conditioned.append(base)
            return conditioned
        tokens = self.build_tokens(action_token_ids, physics_token_ids)
        conditioned = []
        for layer_index, prefix in enumerate(vl_embs_list):
            if prefix.ndim != 3 or prefix.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"layer {layer_index} prefix must be [B, S, {self.hidden_dim}], "
                    f"got {tuple(prefix.shape)}"
                )
            if prefix.shape[0] != tokens.shape[0]:
                raise ValueError("prefix and discrete-token batch sizes must match")
            conditioned.append(torch.cat((prefix, tokens.to(prefix.dtype)), dim=1))
        return conditioned
