"""PhiZero-style tokenizer for adjacent spatial-feature transitions."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import EMAVectorQuantizer, FiniteScalarQuantizer


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    weight = torch.broadcast_to(weight, value.shape)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class SpatiotemporalPhysicalLanguageTokenizer(nn.Module):
    """Quantize each adjacent-frame transition with a fixed query group."""

    def __init__(
        self,
        feature_dim: int,
        num_frames: int = 17,
        spatial_tokens: int = 16,
        queries_per_transition: int = 16,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_heads: int = 8,
        quantizer: str = "ema_vq",
        codebook_size: int = 128,
        fsq_levels: Iterable[int] = (4, 4, 8),
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_code_steps: int = 200,
        token_routing: str = "global_qformer",
        action_aux_dim: int = 0,
        action_aux_weight: float = 0.0,
        action_velocity_weight: float = 0.25,
        action_gripper_weight: float = 0.1,
        action_aux_target_mode: str = "absolute",
        action_aux_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if min(feature_dim, spatial_tokens, queries_per_transition) < 1 or num_frames < 2:
            raise ValueError("feature/token sizes must be positive and num_frames at least 2")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if token_routing not in {"global_qformer", "spatial_aligned"}:
            raise ValueError(f"unsupported token routing: {token_routing}")
        if token_routing == "spatial_aligned":
            spatial_side = math.isqrt(spatial_tokens)
            query_side = math.isqrt(queries_per_transition)
            if (
                spatial_side * spatial_side != spatial_tokens
                or query_side * query_side != queries_per_transition
                or spatial_side % query_side
            ):
                raise ValueError(
                    "spatial_aligned routing requires square spatial/query grids and "
                    "the spatial grid side divisible by the query grid side"
                )
        self.feature_dim = int(feature_dim)
        self.num_frames = int(num_frames)
        self.num_transitions = self.num_frames - 1
        self.spatial_tokens = int(spatial_tokens)
        self.queries_per_transition = int(queries_per_transition)
        self.num_tokens = self.num_transitions * self.queries_per_transition
        self.quantizer_type = str(quantizer)
        self.token_routing = str(token_routing)
        self.action_aux_dim = int(action_aux_dim)
        self.action_aux_weight = float(action_aux_weight)
        self.action_velocity_weight = float(action_velocity_weight)
        self.action_gripper_weight = float(action_gripper_weight)
        self.action_aux_target_mode = str(action_aux_target_mode)
        self.action_aux_pooling = str(action_aux_pooling)
        if self.action_aux_dim not in {0, 14}:
            raise ValueError("action_aux_dim must be 0 or 14")
        if self.action_aux_target_mode not in {"absolute", "action_token_residual"}:
            raise ValueError(f"unsupported action auxiliary target mode: {self.action_aux_target_mode}")
        if self.action_aux_pooling not in {"mean", "flatten"}:
            raise ValueError(f"unsupported action auxiliary pooling: {self.action_aux_pooling}")
        if min(
            self.action_aux_weight,
            self.action_velocity_weight,
            self.action_gripper_weight,
        ) < 0:
            raise ValueError("action auxiliary loss weights must be non-negative")
        if token_routing == "spatial_aligned":
            encoder_attention_mask = torch.full(
                (queries_per_transition, spatial_tokens), float("-inf")
            )
            group_side = spatial_side // query_side
            for spatial_index in range(spatial_tokens):
                row, column = divmod(spatial_index, spatial_side)
                query_index = (row // group_side) * query_side + column // group_side
                encoder_attention_mask[query_index, spatial_index] = 0.0
            decoder_attention_mask = encoder_attention_mask.transpose(0, 1).contiguous()
        else:
            encoder_attention_mask = torch.empty(0)
            decoder_attention_mask = torch.empty(0)
        self.register_buffer(
            "spatial_alignment_mask", encoder_attention_mask, persistent=False
        )
        self.register_buffer(
            "spatial_decode_alignment_mask", decoder_attention_mask, persistent=False
        )

        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.spatial_position = nn.Parameter(torch.randn(spatial_tokens, hidden_dim) * 0.02)
        self.transition_position = nn.Parameter(
            torch.randn(self.num_transitions, hidden_dim) * 0.02
        )
        self.motion_queries = nn.Parameter(
            torch.randn(queries_per_transition, hidden_dim) * 0.02
        )
        self.encoder_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

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
        self.output_queries = nn.Parameter(torch.randn(spatial_tokens, hidden_dim) * 0.02)
        self.decoder_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, feature_dim),
        )
        action_aux_input_dim = (
            latent_dim * queries_per_transition
            if self.action_aux_pooling == "flatten"
            else latent_dim
        )
        self.action_aux_head = (
            nn.Sequential(
                nn.LayerNorm(action_aux_input_dim),
                nn.Linear(action_aux_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.action_aux_dim),
            )
            if self.action_aux_dim
            else None
        )

    def action_aux_features(self, quantized: torch.Tensor) -> torch.Tensor:
        """Pool quantized spatial latents without silently changing decoder semantics."""
        if quantized.ndim != 4 or quantized.shape[1:3] != (
            self.num_transitions,
            self.queries_per_transition,
        ):
            raise ValueError(
                "quantized latents must have shape "
                f"[B,{self.num_transitions},{self.queries_per_transition},D]"
            )
        if self.action_aux_pooling == "flatten":
            return quantized.flatten(start_dim=2)
        return quantized.mean(dim=2)

    def predict_action_aux(self, quantized: torch.Tensor) -> torch.Tensor:
        if self.action_aux_head is None:
            raise RuntimeError("action auxiliary decoder is disabled")
        return self.action_aux_head(self.action_aux_features(quantized))

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
        expected = (self.num_frames, self.spatial_tokens, self.feature_dim)
        if features.ndim != 4 or features.shape[1:] != expected:
            raise ValueError(f"expected features [B,{','.join(map(str, expected))}], got {tuple(features.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)
            if valid_mask.shape != features.shape[:2]:
                raise ValueError("valid_mask shape must match features[:2]")
        if not valid_mask[:, 0].all():
            raise ValueError("the current frame must always be valid")
        return valid_mask

    def _transition_targets(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 1:] - features[:, :-1]

    def _standardize(self, deltas: torch.Tensor) -> torch.Tensor:
        return (deltas - self.feature_mean) / self.feature_std

    def _unstandardize(self, deltas: torch.Tensor) -> torch.Tensor:
        return deltas * self.feature_std + self.feature_mean

    def encode_continuous(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode adjacent transitions without applying the quantizer."""
        valid_mask = self._validate(features, valid_mask)
        deltas = self._standardize(self._transition_targets(features))
        batch = features.shape[0]
        hidden = self.input_proj(deltas)
        hidden = hidden + self.spatial_position[None, None]
        hidden = hidden + self.transition_position[None, :, None]
        hidden = hidden.reshape(batch * self.num_transitions, self.spatial_tokens, -1)
        transition_position = self.transition_position[:, None, :].expand(
            -1, self.queries_per_transition, -1
        )
        queries = self.motion_queries[None] + transition_position
        queries = queries[None].expand(batch, -1, -1, -1).reshape(
            batch * self.num_transitions, self.queries_per_transition, -1
        )
        attention_mask = (
            self.spatial_alignment_mask
            if self.token_routing == "spatial_aligned"
            else None
        )
        tokens, _ = self.encoder_attention(
            queries, hidden, hidden, attn_mask=attention_mask, need_weights=False
        )
        tokens = self.encoder_norm(tokens + queries).reshape(
            batch, self.num_transitions, self.queries_per_transition, -1
        )
        return self.to_latent(tokens)

    def encode(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        latents = self.encode_continuous(features, valid_mask)
        quantized, ids, stats = self.quantizer(latents)
        return ids.reshape(features.shape[0], self.num_tokens), quantized, stats

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 3:
            latents = latents.reshape(
                latents.shape[0], self.num_transitions, self.queries_per_transition, -1
            )
        batch = latents.shape[0]
        memory = self.from_latent(latents).reshape(
            batch * self.num_transitions, self.queries_per_transition, -1
        )
        queries = self.output_queries[None, None] + self.spatial_position[None, None]
        queries = queries + self.transition_position[None, :, None]
        queries = queries.expand(batch, -1, -1, -1).reshape(
            batch * self.num_transitions, self.spatial_tokens, -1
        )
        attention_mask = (
            self.spatial_decode_alignment_mask
            if self.token_routing == "spatial_aligned"
            else None
        )
        decoded, _ = self.decoder_attention(
            queries, memory, memory, attn_mask=attention_mask, need_weights=False
        )
        decoded = self.output_head(decoded + queries).reshape(
            batch, self.num_transitions, self.spatial_tokens, self.feature_dim
        )
        return self._unstandardize(decoded)

    def decode_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.num_tokens:
            raise ValueError(f"token ids must have shape [B,{self.num_tokens}]")
        return self._decode_latents(self.quantizer.decode(ids))

    def reconstruct_continuous(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct transitions through the encoder/decoder without quantization."""
        latents = self.encode_continuous(features, valid_mask)
        return self._decode_latents(self.quantizer.project_continuous(latents))

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        action_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid_mask = self._validate(features, valid_mask)
        ids, quantized, stats = self.encode(features, valid_mask)
        reconstruction = self._decode_latents(quantized)
        target = self._transition_targets(features)
        transition_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        standardized_reconstruction = self._standardize(reconstruction)
        standardized_target = self._standardize(target)
        reconstruction_loss = _masked_mean(
            F.smooth_l1_loss(
                standardized_reconstruction, standardized_target, reduction="none"
            ),
            transition_mask,
        )
        cosine_loss = _masked_mean(
            1.0
            - F.cosine_similarity(
                standardized_reconstruction, standardized_target, dim=-1, eps=1e-6
            ),
            transition_mask,
        )
        loss = reconstruction_loss + 0.25 * cosine_loss + stats["commitment_loss"]
        result = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "cosine_loss": cosine_loss,
            "reconstruction": reconstruction,
            "token_ids": ids,
            **stats,
        }
        if self.action_aux_head is not None and action_targets is not None:
            expected = (features.shape[0], self.num_frames, self.action_aux_dim)
            if action_targets.shape != expected:
                raise ValueError(
                    f"action_targets must have shape {expected}, got {tuple(action_targets.shape)}"
                )
            action_targets = action_targets.to(quantized.dtype)
            action_prediction = self.predict_action_aux(quantized)
            period = action_targets.new_tensor(2.0 * torch.pi)
            if self.action_aux_target_mode == "absolute":
                joint_target = torch.remainder(
                    action_targets[:, 1:, :12] - action_targets[:, :1, :12] + torch.pi,
                    period,
                ) - torch.pi
            else:
                joint_target = action_targets[:, 1:, :12]
            joint_prediction = action_prediction[..., :12]
            joint_loss = _masked_mean(
                F.smooth_l1_loss(joint_prediction, joint_target, reduction="none"),
                transition_mask,
            )
            velocity_mask = transition_mask[:, 1:] & transition_mask[:, :-1]
            velocity_loss = _masked_mean(
                F.smooth_l1_loss(
                    joint_prediction[:, 1:] - joint_prediction[:, :-1],
                    joint_target[:, 1:] - joint_target[:, :-1],
                    reduction="none",
                ),
                velocity_mask,
            )
            gripper_target = action_targets[:, 1:, 12:14]
            if self.action_aux_target_mode == "absolute":
                gripper_error = F.binary_cross_entropy_with_logits(
                    action_prediction[..., 12:14], gripper_target, reduction="none"
                )
            else:
                gripper_error = F.smooth_l1_loss(
                    action_prediction[..., 12:14], gripper_target, reduction="none"
                )
            gripper_loss = _masked_mean(gripper_error, transition_mask)
            action_loss = (
                joint_loss
                + self.action_velocity_weight * velocity_loss
                + self.action_gripper_weight * gripper_loss
            )
            result.update(
                loss=loss + self.action_aux_weight * action_loss,
                action_aux_loss=action_loss,
                action_joint_loss=joint_loss,
                action_velocity_loss=velocity_loss,
                action_gripper_loss=gripper_loss,
                action_prediction=action_prediction,
            )
        return result
