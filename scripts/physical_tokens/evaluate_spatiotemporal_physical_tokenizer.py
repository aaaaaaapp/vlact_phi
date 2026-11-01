#!/usr/bin/env python3
"""Evaluate quantized, continuous, and no-motion tokenizer reconstructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer
from train_spatiotemporal_physical_tokenizer import (
    SpatiotemporalFeatureDataset,
    evaluate,
    parse_ints,
    split_complete_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-array-cache", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-target-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-episodes", default="5,20,25,39,49")
    parser.add_argument("--validation-episodes-per-task", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_indices_by_task(
    dataset: SpatiotemporalFeatureDataset, indices: list[int]
) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for index in indices:
        task_id = int(dataset.task_ids[index])
        grouped.setdefault(task_id, []).append(index)
    return grouped


def make_loader(
    dataset: SpatiotemporalFeatureDataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def collect_position_code_usage(
    model: SpatiotemporalPhysicalLanguageTokenizer,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Count IDs across the full held-out set at each token position."""
    codebook_size = model.codebook_size
    position_counts = torch.zeros(model.num_tokens, codebook_size, device=device, dtype=torch.long)
    offsets = torch.arange(model.num_tokens, device=device) * codebook_size
    for batch in loader:
        ids, _, _ = model.encode(
            batch["features"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
        position_counts += torch.bincount(
            (ids.long() + offsets).flatten(),
            minlength=model.num_tokens * codebook_size,
        ).reshape(model.num_tokens, codebook_size)
    result = []
    for counts in position_counts:
        probabilities = counts.float() / counts.sum().clamp_min(1)
        entropy = -(probabilities * (probabilities + 1e-10).log()).sum()
        result.append({
            "active_codes": int((counts > 0).sum().cpu()),
            "entropy_nats": float(entropy.cpu()),
            "normalized_entropy": float((entropy / math.log(codebook_size)).cpu()),
            "perplexity": float(entropy.exp().cpu()),
            "top10_fraction": float(
                (counts.topk(min(10, codebook_size)).values.sum() / counts.sum().clamp_min(1)).cpu()
            ),
            "total_tokens": int(counts.sum().cpu()),
        })
    return result


def main() -> None:
    args = parse_args()
    dataset = SpatiotemporalFeatureDataset(
        args.input, args.action_target_cache, args.feature_array_cache
    )
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
    loader = make_loader(
        dataset,
        validation_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, loader, device, world_size=1)
    task_metrics = {}
    for task_id, task_indices in group_indices_by_task(dataset, validation_indices).items():
        task_loader = make_loader(
            dataset,
            task_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        task_metrics[dataset.task_names[task_id]] = {
            "task_id": task_id,
            "validation_complete_clips": len(task_indices),
            "metrics": evaluate(model, task_loader, device, world_size=1),
        }
    macro_metrics = {
        key: float(np.mean([entry["metrics"][key] for entry in task_metrics.values()]))
        for key in metrics
    }
    report = {
        "metric_schema_version": 3,
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.resolve().stat().st_size,
        "model_config": config,
        "num_tasks": len(dataset.task_names),
        "task_names": list(dataset.task_names),
        "validation_complete_clips": len(validation_indices),
        "metrics": metrics,
        "macro_metrics": macro_metrics,
        "task_metrics": task_metrics,
        "position_code_usage": collect_position_code_usage(model, loader, device),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
