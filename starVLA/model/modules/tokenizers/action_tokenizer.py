"""Compact discrete tokenizers for normalized robot action chunks."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, value.shape)
    denominator = mask.sum()
    result = (value * mask).sum() / denominator.clamp_min(1)
    return torch.where(denominator > 0, result, value.sum() * 0.0)


class EMAVectorQuantizer(nn.Module):
    """Nearest-neighbor VQ with distributed EMA updates and dead-code revival."""

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_weight: float = 0.25,
        dead_code_steps: int = 200,
        dead_code_count: float = 0.1,
    ) -> None:
        super().__init__()
        if dim < 1 or codebook_size < 2:
            raise ValueError("dim must be positive and codebook_size must be at least 2")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.decay = float(decay)
        self.eps = float(eps)
        self.commitment_weight = float(commitment_weight)
        self.dead_code_steps = int(dead_code_steps)
        self.dead_code_count = float(dead_code_count)

        codebook = torch.empty(codebook_size, dim)
        nn.init.normal_(codebook, std=dim**-0.5)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_sum", codebook.clone())
        self.register_buffer("inactive_steps", torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, flat: torch.Tensor) -> None:
        if bool(self.initialized) or flat.numel() == 0:
            return
        candidates = flat.detach().float()
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.linspace(
                    0, max(0, len(candidates) - 1), self.codebook_size, device=flat.device
                ).long()
                initial = candidates[indices % len(candidates)]
            else:
                initial = torch.empty_like(self.codebook, dtype=torch.float32)
            dist.broadcast(initial, src=0)
        else:
            indices = torch.linspace(
                0, max(0, len(candidates) - 1), self.codebook_size, device=flat.device
            ).long()
            initial = candidates[indices % len(candidates)]
        self.codebook.copy_(initial.to(self.codebook.dtype))
        self.ema_sum.copy_(initial.to(self.ema_sum.dtype))
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, ids: torch.Tensor) -> None:
        counts = torch.bincount(ids, minlength=self.codebook_size).to(torch.float32)
        sums = torch.zeros(
            self.codebook_size, self.dim, device=flat.device, dtype=torch.float32
        )
        sums.index_add_(0, ids, flat.float())
        _all_reduce_sum(counts)
        _all_reduce_sum(sums)

        self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        total = self.ema_count.sum()
        smoothed = (
            (self.ema_count + self.eps)
            / (total + self.codebook_size * self.eps)
            * total
        )
        self.codebook.copy_((self.ema_sum / smoothed.unsqueeze(-1)).to(self.codebook.dtype))

        used = counts > 0
        self.inactive_steps.add_(1)
        self.inactive_steps[used] = 0
        if self.dead_code_steps <= 0:
            return
        dead = (self.inactive_steps >= self.dead_code_steps) & (
            self.ema_count < self.dead_code_count
        )
        if not dead.any() or flat.numel() == 0:
            return
        dead_indices = dead.nonzero(as_tuple=False).flatten()
        replacements = flat.detach().float()[
            torch.arange(len(dead_indices), device=flat.device) % len(flat)
        ]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(replacements, src=0)
        self.codebook[dead_indices] = replacements.to(self.codebook.dtype)
        self.ema_sum[dead_indices] = replacements.to(self.ema_sum.dtype)
        self.ema_count[dead_indices] = 1.0
        self.inactive_steps[dead_indices] = 0

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {tokens.shape[-1]}")
        flat = tokens.reshape(-1, self.dim)
        if self.training:
            self._initialize(flat)
        distance = (
            flat.float().square().sum(-1, keepdim=True)
            - 2.0 * flat.float() @ self.codebook.float().t()
            + self.codebook.float().square().sum(-1).unsqueeze(0)
        )
        ids = distance.argmin(dim=-1)
        quantized = F.embedding(ids, self.codebook).view_as(tokens).to(tokens.dtype)
        if self.training:
            self._ema_update(flat.detach(), ids.detach())
        straight_through = tokens + (quantized - tokens).detach()
        commitment = self.commitment_weight * F.mse_loss(tokens.float(), quantized.detach().float())

        counts = torch.bincount(ids, minlength=self.codebook_size).float()
        probabilities = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probabilities * (probabilities + 1e-10).log()).sum())
        stats = {
            "commitment_loss": commitment,
            "perplexity": perplexity,
            "active_codes": (counts > 0).sum(),
            "active_ratio": (counts > 0).float().mean(),
        }
        return straight_through, ids.view(tokens.shape[:-1]), stats

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.min() < 0 or ids.max() >= self.codebook_size:
            raise ValueError("EMA-VQ token id is outside the codebook")
        return F.embedding(ids.long(), self.codebook)

    def project_continuous(self, tokens: torch.Tensor) -> torch.Tensor:
        """EMA-VQ has no learned bottleneck outside nearest-neighbor lookup."""
        return tokens


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization with learned projections but no learned codebook."""

    def __init__(self, dim: int, levels: Iterable[int] = (8, 8, 8)) -> None:
        super().__init__()
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < 2 for level in levels):
            raise ValueError("every FSQ level must be at least 2")
        self.dim = int(dim)
        self.level_values = levels
        self.codebook_size = math.prod(levels)
        self.project_in = nn.Linear(dim, len(levels))
        self.project_out = nn.Linear(len(levels), dim)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    def _normalized_values(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_max = self.levels.to(projected.dtype) - 1.0
        bounded_digits = (torch.tanh(projected) + 1.0) * 0.5 * level_max
        digits = bounded_digits.round().long().clamp_min(0)
        digits = torch.minimum(digits, self.levels - 1)
        ids = (digits * self.basis).sum(dim=-1)
        hard_normalized = digits.to(projected.dtype) / level_max * 2.0 - 1.0
        soft_normalized = bounded_digits / level_max * 2.0 - 1.0
        return hard_normalized, soft_normalized, ids

    def _quantize(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hard_normalized, soft_normalized, ids = self._normalized_values(projected)
        normalized = soft_normalized + (hard_normalized - soft_normalized).detach()
        return normalized, ids

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        normalized, ids = self._quantize(self.project_in(tokens))
        quantized = self.project_out(normalized)
        counts = torch.bincount(ids.reshape(-1), minlength=self.codebook_size).float()
        probabilities = counts / counts.sum().clamp_min(1)
        stats = {
            "commitment_loss": tokens.new_zeros(()),
            "perplexity": torch.exp(-(probabilities * (probabilities + 1e-10).log()).sum()),
            "active_codes": (counts > 0).sum(),
            "active_ratio": (counts > 0).float().mean(),
        }
        return quantized, ids, stats

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.min() < 0 or ids.max() >= self.codebook_size:
            raise ValueError("FSQ token id is outside the finite code space")
        digits = (ids.long().unsqueeze(-1) // self.basis) % self.levels
        level_max = self.levels.to(dtype=self.project_out.weight.dtype) - 1.0
        normalized = digits.to(level_max.dtype) / level_max * 2.0 - 1.0
        return self.project_out(normalized)

    def project_continuous(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the learned FSQ bottleneck while omitting only scalar rounding."""
        _, soft_normalized, _ = self._normalized_values(self.project_in(tokens))
        return self.project_out(soft_normalized)


class _TemporalEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.input_norm = nn.LayerNorm(action_dim)
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, 1),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(self.input_norm(actions)).transpose(1, 2)
        hidden = self.network(hidden)
        return F.adaptive_avg_pool1d(hidden, self.num_tokens).transpose(1, 2)


class _TemporalDecoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.network = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, action_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(
            latents.transpose(1, 2), size=self.horizon, mode="linear", align_corners=False
        )
        return self.network(hidden).transpose(1, 2)


class ActionChunkTokenizer(nn.Module):
    """Encode one normalized action chunk into a short sequence of discrete IDs."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        num_tokens: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        quantizer: str = "ema_vq",
        codebook_size: int = 512,
        fsq_levels: Iterable[int] = (8, 8, 8),
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_code_steps: int = 200,
    ) -> None:
        super().__init__()
        if horizon < 2 or not 1 <= num_tokens <= horizon:
            raise ValueError("num_tokens must be in [1, horizon] and horizon must be at least 2")
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_tokens = int(num_tokens)
        self.quantizer_type = str(quantizer)
        self.encoder = _TemporalEncoder(action_dim, hidden_dim, latent_dim, num_tokens)
        self.decoder = _TemporalDecoder(action_dim, hidden_dim, latent_dim, horizon)
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

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    def _validate_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, T, D], got {tuple(actions.shape)}")
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"expected action shape [B, {self.horizon}, {self.action_dim}], got {tuple(actions.shape)}"
            )

    def encode(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_actions(actions)
        latents = self.encoder(actions)
        quantized, ids, stats = self.quantizer(latents)
        return ids, quantized, stats

    def _decode_action_values(
        self,
        decoder_output: torch.Tensor,
        gripper_dims: Iterable[int],
    ) -> torch.Tensor:
        gripper_indices = [int(index) % self.action_dim for index in gripper_dims]
        if not gripper_indices:
            return decoder_output
        reconstruction = decoder_output.clone()
        reconstruction[..., gripper_indices] = torch.sigmoid(
            decoder_output[..., gripper_indices]
        )
        return reconstruction

    def decode_tokens(
        self,
        ids: torch.Tensor,
        gripper_dims: Iterable[int] = (),
    ) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.num_tokens:
            raise ValueError(f"token ids must have shape [B, {self.num_tokens}]")
        decoder_output = self.decoder(self.quantizer.decode(ids))
        return self._decode_action_values(decoder_output, gripper_dims)

    def forward(
        self,
        actions: torch.Tensor,
        gripper_dims: Iterable[int] = (),
    ) -> dict[str, torch.Tensor]:
        ids, quantized, stats = self.encode(actions)
        decoder_output = self.decoder(quantized)
        reconstruction = self._decode_action_values(decoder_output, gripper_dims)
        return {
            "reconstruction": reconstruction,
            "decoder_output": decoder_output,
            "token_ids": ids,
            **stats,
        }

    def loss(
        self,
        actions: torch.Tensor,
        *,
        action_valid_mask: torch.Tensor | None = None,
        action_dim_mask: torch.Tensor | None = None,
        angular_dim_mask: torch.Tensor | None = None,
        gripper_dims: Iterable[int] = (),
        velocity_weight: float = 0.25,
        gripper_weight: float = 0.25,
        wrap_weight: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        gripper_indices = [int(index) % self.action_dim for index in gripper_dims]
        output = self(actions, gripper_dims=gripper_indices)
        reconstruction = output["reconstruction"]
        mask = None
        if action_valid_mask is not None:
            mask = action_valid_mask.to(dtype=torch.bool, device=actions.device).unsqueeze(-1)
        if action_dim_mask is not None:
            dim_mask = action_dim_mask.to(dtype=torch.bool, device=actions.device)
            if dim_mask.ndim == 1:
                dim_mask = dim_mask.unsqueeze(0)
            dim_mask = dim_mask.unsqueeze(1)
            mask = dim_mask if mask is None else mask & dim_mask

        raw_difference = reconstruction - actions
        reconstruction_loss = _masked_mean(F.smooth_l1_loss(reconstruction, actions, reduction="none"), mask)
        velocity_mask = None if mask is None else mask[:, 1:] & mask[:, :-1]
        velocity_loss = _masked_mean(
            F.smooth_l1_loss(
                reconstruction[:, 1:] - reconstruction[:, :-1],
                actions[:, 1:] - actions[:, :-1],
                reduction="none",
            ),
            velocity_mask,
        )

        wrap_loss = actions.new_zeros(())
        if angular_dim_mask is not None:
            angular = angular_dim_mask.to(dtype=torch.bool, device=actions.device)
            if angular.ndim == 1:
                angular = angular.unsqueeze(0)
            angular_mask = angular.unsqueeze(1)
            if mask is not None:
                angular_mask = angular_mask & mask
            wrapped = torch.remainder(raw_difference + math.pi, 2 * math.pi) - math.pi
            wrap_loss = _masked_mean(F.smooth_l1_loss(wrapped, torch.zeros_like(wrapped), reduction="none"), angular_mask)

        gripper_loss = actions.new_zeros(())
        if gripper_indices:
            logits = output["decoder_output"][..., gripper_indices]
            targets = actions[..., gripper_indices].clamp(0, 1)
            gripper_mask = None if mask is None else mask[..., gripper_indices]
            gripper_loss = _masked_mean(
                F.binary_cross_entropy_with_logits(logits, targets, reduction="none"),
                gripper_mask,
            )

        total = (
            reconstruction_loss
            + float(velocity_weight) * velocity_loss
            + float(gripper_weight) * gripper_loss
            + float(wrap_weight) * wrap_loss
            + output["commitment_loss"]
        )
        return {
            **output,
            "loss": total,
            "reconstruction_loss": reconstruction_loss,
            "velocity_loss": velocity_loss,
            "gripper_loss": gripper_loss,
            "wrap_loss": wrap_loss,
        }
