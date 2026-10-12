"""Physics-token cross-attention adapters for noisy action DiT states."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn


def resolve_injection_layers(
    selection: str | Sequence[int], num_layers: int
) -> tuple[int, ...]:
    """Resolve named or explicit DiT layer selections to validated indices."""
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if isinstance(selection, str):
        normalized = selection.strip().lower()
        if normalized == "all":
            indices = tuple(range(num_layers))
        elif normalized in ("first", "first-block"):
            indices = (0,)
        elif normalized in ("early4", "early-4"):
            indices = tuple(range(min(4, num_layers)))
        else:
            try:
                indices = tuple(
                    int(value.strip())
                    for value in normalized.split(",")
                    if value.strip()
                )
            except ValueError as error:
                raise ValueError(f"invalid physics injection layers: {selection}") from error
    else:
        indices = tuple(int(value) for value in selection)
    if not indices:
        raise ValueError("at least one physics injection layer is required")
    if len(set(indices)) != len(indices):
        raise ValueError("physics injection layers must be unique")
    if min(indices) < 0 or max(indices) >= num_layers:
        raise ValueError(
            f"physics injection layer outside [0, {num_layers}): {indices}"
        )
    return indices


class PhysicsCrossAttentionAdapter(nn.Module):
    """One residual cross-attention branch with a zero-initialized output."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        residual, _ = self.attention(
            self.query_norm(hidden_states),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        return hidden_states + self.output_proj(residual).to(hidden_states.dtype)


class PhysicsTemporalFiLMAdapter(nn.Module):
    """Zero-initialized FiLM with deterministic temporal slot alignment."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[1] % memory.shape[1]:
            raise ValueError(
                "action-query count must be divisible by physics transition count"
            )
        repeats = hidden_states.shape[1] // memory.shape[1]
        aligned_memory = memory.repeat_interleave(repeats, dim=1)
        scale, shift = self.modulation(aligned_memory).chunk(2, dim=-1)
        residual = scale * self.hidden_norm(hidden_states) + shift
        return hidden_states + residual.to(hidden_states.dtype)


class MultiLayerPhysicsTokenAdapter(nn.Module):
    """Route shared discrete physics tokens into selected action-DiT blocks.

    The caller prepares physics memory once per denoising pass and invokes
    :meth:`apply_layer` immediately after each original DiT block. A disabled or
    dropped branch returns the exact input tensor without entering attention.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        physics_vocab_size: int,
        num_layers: int,
        injection_layers: str | Sequence[int] = "all",
        max_physics_tokens: int = 256,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        condition_dropout: float = 0.2,
        memory_pooling: str = "none",
        tokens_per_transition: int = 16,
        adapter_type: str = "cross_attention",
    ) -> None:
        super().__init__()
        if physics_vocab_size < 2:
            raise ValueError("physics_vocab_size must be at least 2")
        if max_physics_tokens < 1:
            raise ValueError("max_physics_tokens must be positive")
        if not 0.0 <= condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if memory_pooling not in {"none", "transition_mean"}:
            raise ValueError("memory_pooling must be 'none' or 'transition_mean'")
        if tokens_per_transition < 1:
            raise ValueError("tokens_per_transition must be positive")
        if adapter_type not in {"cross_attention", "temporal_film"}:
            raise ValueError(
                "adapter_type must be 'cross_attention' or 'temporal_film'"
            )
        self.hidden_dim = int(hidden_dim)
        self.physics_vocab_size = int(physics_vocab_size)
        self.max_physics_tokens = int(max_physics_tokens)
        self.condition_dropout = float(condition_dropout)
        self.memory_pooling = str(memory_pooling)
        self.tokens_per_transition = int(tokens_per_transition)
        self.adapter_type = str(adapter_type)
        self.injection_layers = resolve_injection_layers(injection_layers, num_layers)

        self.physics_embedding = nn.Embedding(physics_vocab_size, hidden_dim)
        self.physics_position = nn.Embedding(max_physics_tokens, hidden_dim)
        self.physics_type = nn.Parameter(torch.zeros(hidden_dim))
        self.memory_norm = nn.LayerNorm(hidden_dim)
        if self.adapter_type == "cross_attention":
            adapter_factory = lambda: PhysicsCrossAttentionAdapter(
                hidden_dim, num_heads, dropout=attention_dropout
            )
        else:
            adapter_factory = lambda: PhysicsTemporalFiLMAdapter(hidden_dim)
        self.adapters = nn.ModuleDict(
            {str(index): adapter_factory() for index in self.injection_layers}
        )
        nn.init.normal_(self.physics_embedding.weight, std=hidden_dim**-0.5)
        nn.init.normal_(self.physics_position.weight, std=0.02)

    @torch.no_grad()
    def initialize_code_embeddings(self, codebook: torch.Tensor) -> None:
        """Initialize ID embeddings from an EMA tokenizer codebook.

        Repeating a code vector preserves all pairwise dot products up to one
        shared scale while matching the wider DiT hidden dimension.
        """
        codebook = torch.as_tensor(codebook).detach().float().cpu()
        expected_rows = self.physics_embedding.num_embeddings
        if codebook.ndim != 2 or codebook.shape[0] != expected_rows:
            raise ValueError(
                "physics codebook must have shape "
                f"[{expected_rows}, D], got {tuple(codebook.shape)}"
            )
        if not torch.isfinite(codebook).all():
            raise ValueError("physics codebook contains non-finite values")
        code_dim = int(codebook.shape[1])
        if code_dim < 1 or self.hidden_dim % code_dim:
            raise ValueError(
                f"DiT hidden dim {self.hidden_dim} must be divisible by "
                f"physics code dim {code_dim}"
            )
        repeats = self.hidden_dim // code_dim
        expanded = codebook.repeat(1, repeats) / math.sqrt(repeats)
        self.physics_embedding.weight.copy_(
            expanded.to(
                device=self.physics_embedding.weight.device,
                dtype=self.physics_embedding.weight.dtype,
            )
        )

    def _validate_ids(self, physics_token_ids: torch.Tensor) -> None:
        if physics_token_ids.ndim != 2:
            raise ValueError(
                f"physics token IDs must be [B, N], got {tuple(physics_token_ids.shape)}"
            )
        if physics_token_ids.shape[1] > self.max_physics_tokens:
            raise ValueError(
                f"physics token count {physics_token_ids.shape[1]} exceeds configured "
                f"maximum {self.max_physics_tokens}"
            )
        if physics_token_ids.numel() and (
            physics_token_ids.min() < 0
            or physics_token_ids.max() >= self.physics_vocab_size
        ):
            raise ValueError(
                f"physics token ID is outside [0, {self.physics_vocab_size})"
            )

    def prepare_memory(
        self,
        physics_token_ids: torch.Tensor | None,
        *,
        enabled: bool = True,
        drop_condition: bool | None = None,
    ) -> torch.Tensor | None:
        if not enabled or physics_token_ids is None:
            return None
        self._validate_ids(physics_token_ids)
        if drop_condition is None:
            drop_condition = bool(
                self.training
                and torch.rand((), device=physics_token_ids.device) < self.condition_dropout
            )
        if drop_condition:
            return None
        ids = physics_token_ids.long()
        positions = torch.arange(ids.shape[1], device=ids.device)
        memory = (
            self.physics_embedding(ids)
            + self.physics_position(positions)
            + self.physics_type
        )
        memory = self.memory_norm(memory)
        if self.memory_pooling == "transition_mean":
            if ids.shape[1] % self.tokens_per_transition:
                raise ValueError(
                    "physics token count must be divisible by tokens_per_transition"
                )
            memory = memory.reshape(
                ids.shape[0],
                ids.shape[1] // self.tokens_per_transition,
                self.tokens_per_transition,
                self.hidden_dim,
            ).mean(dim=2)
        return memory

    def apply_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        memory: torch.Tensor | None,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled or memory is None or str(layer_index) not in self.adapters:
            return hidden_states
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"DiT hidden states must be [B, T, {self.hidden_dim}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if memory.shape[0] != hidden_states.shape[0]:
            raise ValueError("physics memory and DiT hidden-state batch sizes must match")
        return self.adapters[str(layer_index)](
            hidden_states,
            memory.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

    def attach_zero_gradient(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep a dropped adapter-only batch differentiable without changing values."""
        anchor = hidden_states.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.reshape(-1)[0].to(hidden_states.dtype) * 0.0
        return hidden_states + anchor
