#!/usr/bin/env python3
"""Join action and physics token targets by exact task-qualified sample key."""

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


def load(path: Path, token_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            token_key,
            "task_ids",
            "task_names",
            "episode_ids",
            "base_indices",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def qualified_keys(arrays: dict[str, np.ndarray]) -> list[tuple[int, int, int]]:
    return [
        (int(task), int(episode), int(base))
        for task, episode, base in zip(
            arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"]
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    action_path = args.action_cache.resolve()
    physics_path = args.physics_cache.resolve()
    action = load(action_path, "action_token_ids")
    physics = load(physics_path, "physics_token_ids")
    if not np.array_equal(action["task_names"], physics["task_names"]):
        raise ValueError("action and physics caches have different task_names")

    action_keys = qualified_keys(action)
    physics_keys = qualified_keys(physics)
    action_rows = {key: row for row, key in enumerate(action_keys)}
    if len(action_rows) != len(action_keys):
        raise ValueError("action cache contains duplicate qualified keys")
    if len(set(physics_keys)) != len(physics_keys):
        raise ValueError("physics cache contains duplicate qualified keys")
    missing = [key for key in physics_keys if key not in action_rows]
    if missing:
        raise ValueError(f"action cache is missing {len(missing)} physics keys; first={missing[0]}")
    indices = np.asarray([action_rows[key] for key in physics_keys], dtype=np.int64)

    arrays = {
        "action_token_ids": np.asarray(action["action_token_ids"][indices], dtype=np.int32),
        "physics_token_ids": np.asarray(physics["physics_token_ids"], dtype=np.int32),
        "task_ids": np.asarray(physics["task_ids"], dtype=np.int64),
        "task_names": np.asarray(physics["task_names"]),
        "episode_ids": np.asarray(physics["episode_ids"], dtype=np.int64),
        "base_indices": np.asarray(physics["base_indices"], dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)

    report = {
        "schema_version": 1,
        "kind": "multitask_dual_action_physics_token_targets",
        "action_cache": str(action_path),
        "action_cache_sha256": sha256_file(action_path),
        "physics_cache": str(physics_path),
        "physics_cache_sha256": sha256_file(physics_path),
        "num_samples": len(physics_keys),
        "num_tasks": len(arrays["task_names"]),
        "action_token_shape": list(arrays["action_token_ids"].shape),
        "physics_token_shape": list(arrays["physics_token_ids"].shape),
        "exact_qualified_key_join": True,
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
