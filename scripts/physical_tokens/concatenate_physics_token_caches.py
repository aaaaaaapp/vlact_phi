#!/usr/bin/env python3
"""Concatenate key-aligned visual-physics and residual token caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--residual-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, token_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data[token_key], dtype=np.int32),
            np.asarray(data["episode_ids"], dtype=np.int64),
            np.asarray(data["base_indices"], dtype=np.int64),
        )


def main() -> None:
    args = parse_args()
    visual_path = args.visual_cache.resolve()
    residual_path = args.residual_cache.resolve()
    visual, episodes, bases = load(visual_path, "physics_token_ids")
    residual, residual_episodes, residual_bases = load(residual_path, "action_token_ids")
    if not np.array_equal(episodes, residual_episodes) or not np.array_equal(bases, residual_bases):
        raise ValueError("visual and residual token caches are not key-aligned")
    combined = np.concatenate((visual, residual), axis=1)
    arrays = {
        "physics_token_ids": combined,
        "episode_ids": episodes,
        "base_indices": bases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "visual_and_action_residual_physics_tokens",
        "visual_cache": str(visual_path),
        "visual_cache_sha256": sha256_file(visual_path),
        "residual_cache": str(residual_path),
        "residual_cache_sha256": sha256_file(residual_path),
        "num_samples": len(combined),
        "num_visual_tokens": visual.shape[1],
        "num_residual_tokens": residual.shape[1],
        "num_tokens": combined.shape[1],
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
