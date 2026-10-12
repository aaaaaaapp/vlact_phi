#!/usr/bin/env python3
"""Export observation-predicted action tokens as a keyed dataset cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_action_token_reasoner import load_reasoner
from evaluate_qwenpi_oracle import build_validation_dataset, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-token-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def export(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    source_cache_path = args.source_token_cache.resolve()
    config = OmegaConf.load(config_path)
    config.framework.action_token_probe.token_count = args.token_count
    config.framework.action_token_probe.vocab_size = args.vocab_size
    config.datasets.vla_data.discrete_token_cache = str(source_cache_path)
    config.datasets.vla_data.exclude_episode_ids = None
    config.datasets.vla_data.include_episode_ids = list(range(50))

    dataset = build_validation_dataset(config, list(range(50)))
    transforms = getattr(dataset, "transforms", None)
    if transforms is not None:
        transforms.eval()
    model = load_reasoner(config, checkpoint_path, torch.device(args.device))

    predictions = []
    for start in range(0, len(dataset), args.batch_size):
        batch = [dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))]
        predictions.append(model.predict_token_ids(batch).astype(np.int32))
    token_ids = np.concatenate(predictions, axis=0)
    if token_ids.shape != (len(dataset), args.token_count):
        raise RuntimeError(f"unexpected prediction shape: {token_ids.shape}")
    keys = np.asarray(dataset.all_steps, dtype=np.int64)
    arrays = {
        "action_token_ids": token_ids,
        "episode_ids": keys[:, 0],
        "base_indices": keys[:, 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": 1,
        "kind": "observation_predicted_action_tokens",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_token_cache": str(source_cache_path),
        "source_token_cache_sha256": sha256_file(source_cache_path),
        "num_samples": len(dataset),
        "num_tokens": args.token_count,
        "vocab_size": args.vocab_size,
        "augmentation_mode": "eval",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    print(json.dumps(export(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
