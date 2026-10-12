#!/usr/bin/env python3
"""Shuffle physics-token packages within each task and episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shuffle_physics(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    required = {
        "action_token_ids",
        "physics_token_ids",
        "task_ids",
        "task_names",
        "episode_ids",
        "base_indices",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"multitask token cache is missing keys: {sorted(missing)}")
    row_count = len(arrays["physics_token_ids"])
    for name in required - {"task_names"}:
        if len(arrays[name]) != row_count:
            raise ValueError(f"cache field {name} has inconsistent row count")

    result = {name: np.array(value, copy=True) for name, value in arrays.items()}
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = {}
    for row, (task_id, episode_id) in enumerate(
        zip(arrays["task_ids"], arrays["episode_ids"])
    ):
        groups.setdefault((int(task_id), int(episode_id)), []).append(row)
    for rows in groups.values():
        if len(rows) > 1:
            shift = int(rng.integers(1, len(rows)))
            result["physics_token_ids"][rows] = arrays["physics_token_ids"][
                np.roll(rows, shift)
            ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input.resolve()
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    shuffled = shuffle_physics(arrays, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **shuffled)
    os.replace(temporary, args.output)

    manifest = {
        "schema_version": 1,
        "kind": "task_episode_local_shuffled_physics_packages",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "seed": args.seed,
        "num_samples": len(shuffled["physics_token_ids"]),
        "physics_token_shape": list(shuffled["physics_token_ids"].shape),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
