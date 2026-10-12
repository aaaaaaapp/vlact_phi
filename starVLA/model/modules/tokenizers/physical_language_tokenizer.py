"""Compact discrete tokenizer for future visual-feature dynamics."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import EMAVectorQuantizer, FiniteScalarQuantizer


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = torch.broadcast_to(weight, value.shape)
    denominator = weight.sum()
    return (value * weight).sum() / denominator.clamp_min(1.0)


class PhysicalLanguageTokenizer(nn.Module):
    """Encode future feature deltas into a fixed number of physical-language IDs."""

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        num_tokens: int = 4,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        quantizer: str = "ema_vq",
        codebook_size: int = 128,
        fsq_levels: Iterable[int] = (4, 4, 8),
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_code_steps: int = 200,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or horizon < 2 or num_tokens < 1:
            raise ValueError("feature_dim, num_tokens, and horizon must be positive; horizon >= 2")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.feature_dim = int(feature_dim)
        self.horizon = int(horizon)
        self.num_tokens = int(num_tokens)
        self.quantizer_type = str(quantizer)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_position = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.token_queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.token_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.to_latent = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

        if quantizer == "ema_vq":
            self.quantizer = EMAVectorQuantizer(
                latent_dim,
                codebook_size=codebook_size,
                decay=ema_decay,
                commitment_weight=commitment_weight,
                dead_code_steps=dead_code_steps,
            )
        elif quantizer == "fsq":
            self.quantizer = FiniteScalarQuantizer(latent_dim, fsq_levels)
        else:
            raise ValueError(f"unsupported quantizer: {quantizer}")

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.output_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        self.decoder_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        )

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.feature_dim,) or std.shape != (self.feature_dim,):
            raise ValueError(f"normalization tensors must have shape [{self.feature_dim}]")
        self.feature_mean.copy_(mean.float())
        self.feature_std.copy_(std.float().clamp_min(1e-5))

    def _validate(self, features: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (self.horizon, self.feature_dim):
            raise ValueError(
                f"expected features [B,{self.horizon},{self.feature_dim}], got {tuple(features.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)
            if valid_mask.shape != features.shape[:2]:
                raise ValueError("valid_mask shape must match features[:2]")
        if not valid_mask[:, 0].all():
            raise ValueError("the current frame at offset 0 must always be valid")
        return valid_mask

    def _standardize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def _unstandardize(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.feature_std + self.feature_mean

    def encode(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        valid_mask = self._validate(features, valid_mask)
        hidden = self.input_proj(self._standardize(features)) + self.input_position
        hidden = self.temporal_encoder(hidden, src_key_padding_mask=~valid_mask)
        hidden = self.encoder_norm(hidden)
        queries = self.token_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        tokens, _ = self.token_cross_attention(
            queries, hidden, hidden, key_padding_mask=~valid_mask, need_weights=False
        )
        quantized, ids, stats = self.quantizer(self.to_latent(tokens))
        return ids, quantized, stats

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        memory = self.from_latent(latents)
        queries = self.output_queries.unsqueeze(0).expand(latents.shape[0], -1, -1)
        decoded, _ = self.decoder_cross_attention(
            queries, memory, memory, need_weights=False
        )
        return self._unstandardize(self.output_head(decoded))

    def decode_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.num_tokens:
            raise ValueError(f"token ids must have shape [B,{self.num_tokens}]")
        return self._decode_latents(self.quantizer.decode(ids))

    def _forward_impl(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ids, quantized, stats = self.encode(features, valid_mask)
        return {
            "reconstruction": self._decode_latents(quantized),
            "token_ids": ids,
            **stats,
        }

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        loss_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        if loss_kwargs is not None:
            return self.loss(features, valid_mask=valid_mask, **loss_kwargs)
        return self._forward_impl(features, valid_mask)

    def loss(
        self,
        features: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        cosine_weight: float = 0.25,
        temporal_weight: float = 0.25,
        current_frame_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        valid_mask = self._validate(features, valid_mask)
        output = self._forward_impl(features, valid_mask)
        reconstruction = output["reconstruction"]
        weights = valid_mask.float()
        weights[:, 0] *= float(current_frame_weight)

        standardized_target = self._standardize(features)
        standardized_reconstruction = self._standardize(reconstruction)
        reconstruction_loss = _weighted_mean(
            F.smooth_l1_loss(
                standardized_reconstruction, standardized_target, reduction="none"
            ),
            weights,
        )
        cosine_loss = _weighted_mean(
            1.0
            - F.cosine_similarity(
                standardized_reconstruction, standardized_target, dim=-1, eps=1e-6
            ),
            weights,
        )
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        temporal_loss = _weighted_mean(
            F.smooth_l1_loss(
                standardized_reconstruction[:, 1:] - standardized_reconstruction[:, :-1],
                standardized_target[:, 1:] - standardized_target[:, :-1],
                reduction="none",
            ),
            pair_mask,
        )
        total = (
            reconstruction_loss
            + float(cosine_weight) * cosine_loss
            + float(temporal_weight) * temporal_loss
            + output["commitment_loss"]
        )
        return {
            **output,
            "loss": total,
            "reconstruction_loss": reconstruction_loss,
            "cosine_loss": cosine_loss,
            "temporal_loss": temporal_loss,
        }
