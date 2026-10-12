#!/usr/bin/env python3
"""Measure whether a nominal 256-token physics sequence is structurally diverse."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer
from train_spatiotemporal_physical_tokenizer import (
    SpatiotemporalFeatureDataset,
    parse_ints,
    split_complete_indices,
)


class TokenStructureAccumulator:
    def __init__(self, num_transitions: int, queries_per_transition: int, codebook_size: int):
        self.num_transitions = int(num_transitions)
        self.queries_per_transition = int(queries_per_transition)
        self.num_tokens = self.num_transitions * self.queries_per_transition
        self.codebook_size = int(codebook_size)
        self.position_counts = torch.zeros(
            self.num_tokens, self.codebook_size, dtype=torch.int64
        )
        self.num_samples = 0
        self.within_unique_sum = 0.0
        self.within_pair_change_sum = 0.0
        self.adjacent_slot_change_sum = 0.0
        self.adjacent_signature_change_sum = 0.0
        self.sample_unique_ratio_sum = 0.0
        self.continuous_centered_square_sum = 0.0
        self.continuous_square_sum = 0.0
        self.continuous_pairwise_cosine_sum = 0.0
        self.continuous_pairwise_cosine_count = 0
        self.continuous_effective_rank_sum = 0.0
        self.continuous_effective_rank_count = 0

    def update(
        self, token_ids: torch.Tensor, continuous_latents: torch.Tensor | None = None
    ) -> None:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.num_tokens:
            raise ValueError(
                f"expected token IDs [B,{self.num_tokens}], got {tuple(token_ids.shape)}"
            )
        ids = token_ids.detach().long().cpu()
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.codebook_size):
            raise ValueError("token ID is outside the codebook")
        batch = ids.shape[0]
        if batch == 0:
            return
        position_offsets = (
            torch.arange(self.num_tokens, dtype=torch.long) * self.codebook_size
        )
        flat_position_ids = ids + position_offsets.unsqueeze(0)
        self.position_counts += torch.bincount(
            flat_position_ids.reshape(-1),
            minlength=self.num_tokens * self.codebook_size,
        ).reshape(self.num_tokens, self.codebook_size)

        grouped = ids.reshape(batch, self.num_transitions, self.queries_per_transition)
        counts = F.one_hot(grouped, num_classes=self.codebook_size).sum(dim=2)
        unique_per_transition = (counts > 0).sum(dim=-1).float()
        self.within_unique_sum += float(
            (unique_per_transition / self.queries_per_transition).sum()
        )
        if self.queries_per_transition > 1:
            matching_pairs = (counts * (counts - 1) // 2).sum(dim=-1).float()
            total_pairs = self.queries_per_transition * (
                self.queries_per_transition - 1
            ) / 2
            self.within_pair_change_sum += float((1.0 - matching_pairs / total_pairs).sum())
        if self.num_transitions > 1:
            adjacent_changed = grouped[:, 1:] != grouped[:, :-1]
            self.adjacent_slot_change_sum += float(adjacent_changed.float().sum())
            self.adjacent_signature_change_sum += float(
                adjacent_changed.any(dim=-1).float().sum()
            )
        sample_unique = F.one_hot(ids, num_classes=self.codebook_size).any(dim=1).sum(dim=-1)
        self.sample_unique_ratio_sum += float(
            (sample_unique.float() / self.codebook_size).sum()
        )
        if continuous_latents is not None:
            expected = (
                batch,
                self.num_transitions,
                self.queries_per_transition,
            )
            if continuous_latents.ndim != 4 or continuous_latents.shape[:3] != expected:
                raise ValueError(
                    "continuous latents must have shape "
                    f"[B,{self.num_transitions},{self.queries_per_transition},D]"
                )
            latents = continuous_latents.detach().float().cpu()
            centered = latents - latents.mean(dim=2, keepdim=True)
            self.continuous_centered_square_sum += float(centered.square().sum())
            self.continuous_square_sum += float(latents.square().sum())
            if self.queries_per_transition > 1:
                normalized = F.normalize(latents, dim=-1, eps=1e-8)
                similarities = normalized @ normalized.transpose(-2, -1)
                off_diagonal = ~torch.eye(
                    self.queries_per_transition, dtype=torch.bool
                ).reshape(1, 1, self.queries_per_transition, self.queries_per_transition)
                values = similarities.masked_select(off_diagonal)
                self.continuous_pairwise_cosine_sum += float(values.sum())
                self.continuous_pairwise_cosine_count += values.numel()
                singular_values = torch.linalg.svdvals(centered)
                energy = singular_values.square()
                effective_rank = energy.sum(dim=-1).square() / energy.square().sum(
                    dim=-1
                ).clamp_min(1e-12)
                self.continuous_effective_rank_sum += float(effective_rank.sum())
                self.continuous_effective_rank_count += effective_rank.numel()
        self.num_samples += batch

    def compute(self) -> dict[str, float | int | list[float] | list[int]]:
        if self.num_samples == 0:
            raise ValueError("cannot summarize an empty token set")
        counts = self.position_counts.float()
        global_counts = counts.sum(dim=0)
        global_probabilities = global_counts / global_counts.sum().clamp_min(1)
        global_entropy = float(
            -(global_probabilities * (global_probabilities + 1e-12).log()).sum()
        )
        position_probabilities = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
        position_entropy = -(
            position_probabilities * (position_probabilities + 1e-12).log()
        ).sum(dim=1)
        position_perplexity = position_entropy.exp()
        position_active = (counts > 0).sum(dim=1)

        joint = counts / counts.sum().clamp_min(1)
        position_marginal = joint.sum(dim=1, keepdim=True)
        code_marginal = joint.sum(dim=0, keepdim=True)
        expected = position_marginal * code_marginal
        nonzero = joint > 0
        mutual_information = float(
            (joint[nonzero] * (joint[nonzero] / expected[nonzero]).log()).sum()
        )
        transition_count = self.num_samples * self.num_transitions
        adjacent_slot_count = (
            self.num_samples
            * max(self.num_transitions - 1, 0)
            * self.queries_per_transition
        )
        adjacent_transition_count = self.num_samples * max(self.num_transitions - 1, 0)
        result = {
            "num_samples": self.num_samples,
            "num_tokens_per_sample": self.num_tokens,
            "codebook_size": self.codebook_size,
            "global_active_codes": int((global_counts > 0).sum()),
            "global_active_ratio": float((global_counts > 0).float().mean()),
            "global_perplexity": math.exp(global_entropy),
            "mean_position_active_codes": float(position_active.float().mean()),
            "min_position_active_codes": int(position_active.min()),
            "max_position_active_codes": int(position_active.max()),
            "mean_position_perplexity": float(position_perplexity.mean()),
            "min_position_perplexity": float(position_perplexity.min()),
            "max_position_perplexity": float(position_perplexity.max()),
            "within_transition_unique_ratio": self.within_unique_sum / transition_count,
            "within_transition_pairwise_token_change": (
                self.within_pair_change_sum / transition_count
                if self.queries_per_transition > 1
                else 0.0
            ),
            "adjacent_transition_same_slot_change": (
                self.adjacent_slot_change_sum / adjacent_slot_count
                if adjacent_slot_count
                else 0.0
            ),
            "adjacent_transition_signature_change": (
                self.adjacent_signature_change_sum / adjacent_transition_count
                if adjacent_transition_count
                else 0.0
            ),
            "per_sample_unique_code_ratio": self.sample_unique_ratio_sum / self.num_samples,
            "position_code_mutual_information_nats": mutual_information,
            "position_code_mi_fraction_of_code_entropy": (
                mutual_information / global_entropy if global_entropy > 0 else 0.0
            ),
            "position_active_codes": position_active.tolist(),
            "position_perplexity": position_perplexity.tolist(),
        }
        if self.continuous_square_sum > 0:
            result.update(
                continuous_within_transition_relative_dispersion=math.sqrt(
                    self.continuous_centered_square_sum / self.continuous_square_sum
                ),
                continuous_within_transition_pairwise_cosine=(
                    self.continuous_pairwise_cosine_sum
                    / self.continuous_pairwise_cosine_count
                    if self.continuous_pairwise_cosine_count
                    else None
                ),
                continuous_within_transition_effective_rank=(
                    self.continuous_effective_rank_sum
                    / self.continuous_effective_rank_count
                    if self.continuous_effective_rank_count
                    else None
                ),
            )
        return result


def analyze_indices(
    model: SpatiotemporalPhysicalLanguageTokenizer,
    dataset: SpatiotemporalFeatureDataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    accumulator = TokenStructureAccumulator(
        model.num_transitions, model.queries_per_transition, model.codebook_size
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for batch in loader:
            continuous_latents = model.encode_continuous(
                batch["features"].to(device, non_blocking=True),
                batch["valid_mask"].to(device, non_blocking=True),
            )
            _, token_ids, _ = model.quantizer(continuous_latents)
            accumulator.update(
                token_ids.reshape(token_ids.shape[0], -1), continuous_latents
            )
    return accumulator.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument("--validation-episodes-per-task", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = SpatiotemporalFeatureDataset(args.input)
    _, validation_indices = split_complete_indices(
        dataset,
        parse_ints(args.validation_episodes),
        args.validation_episodes_per_task,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    config["fsq_levels"] = tuple(config["fsq_levels"])
    model = SpatiotemporalPhysicalLanguageTokenizer(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    aggregate = analyze_indices(
        model,
        dataset,
        validation_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    task_metrics = {}
    for task_id, task_name in enumerate(dataset.task_names):
        task_indices = [
            index for index in validation_indices if int(dataset.task_ids[index]) == task_id
        ]
        if task_indices:
            task_metrics[task_name] = analyze_indices(
                model,
                dataset,
                task_indices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
    report = {
        "schema_version": 1,
        "kind": "spatiotemporal_physics_token_structure",
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_complete_clips": len(validation_indices),
        "task_names": list(dataset.task_names),
        "metrics": aggregate,
        "task_metrics": task_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
