#!/usr/bin/env python3
"""Join action and physics token caches by source episode/frame key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--physics-token-key", default="physics_token_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffle-physics-seed", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path, token_name: str) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    with np.load(path, allow_pickle=False) as data:
        if token_name not in data:
            raise ValueError(f"{path} has no {token_name}")
        token_ids = np.asarray(data[token_name], dtype=np.int32)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
        base_indices = np.asarray(data["base_indices"], dtype=np.int64)
        task_ids = np.asarray(data["task_ids"], dtype=np.int64) if "task_ids" in data else None
    if task_ids is not None and len(task_ids) != len(token_ids):
        raise ValueError(f"{path} task/key lengths differ")
    keys = [
        (int(task), int(episode), int(base)) if task_ids is not None else (int(episode), int(base))
        for row, (episode, base) in enumerate(zip(episode_ids, base_indices))
        for task in ([task_ids[row]] if task_ids is not None else [None])
    ]
    if len(keys) != len(token_ids):
        raise ValueError(f"{path} token/key lengths differ")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} contains duplicate keys")
    return token_ids, keys


def shuffle_within_episodes(
    token_ids: np.ndarray, keys: list[tuple[int, int]], seed: int
) -> np.ndarray:
    result = token_ids.copy()
    rng = np.random.default_rng(seed)
    episode_rows: dict[int, list[int]] = {}
    for row, (episode_id, _) in enumerate(keys):
        episode_rows.setdefault(episode_id, []).append(row)
    for rows in episode_rows.values():
        if len(rows) > 1:
            shift = int(rng.integers(1, len(rows)))
            result[rows] = token_ids[np.roll(rows, shift)]
    return result


def merge(
    action_cache: Path,
    physics_cache: Path,
    shuffle_physics_seed: int | None = None,
    physics_token_key: str = "physics_token_ids",
) -> tuple[dict, dict[str, np.ndarray]]:
    action_ids, action_keys = load_cache(action_cache, "action_token_ids")
    physics_ids, physics_keys = load_cache(physics_cache, physics_token_key)
    if set(action_keys) != set(physics_keys):
        missing_action = sorted(set(physics_keys).difference(action_keys))[:5]
        missing_physics = sorted(set(action_keys).difference(physics_keys))[:5]
        raise ValueError(
            f"cache key sets differ; missing_action={missing_action}, missing_physics={missing_physics}"
        )
    physics_row = {key: row for row, key in enumerate(physics_keys)}
    aligned_physics = np.stack([physics_ids[physics_row[key]] for key in action_keys])
    if shuffle_physics_seed is not None:
        aligned_physics = shuffle_within_episodes(
            aligned_physics, action_keys, shuffle_physics_seed
        )
    keys = np.asarray(action_keys, dtype=np.int64)
    arrays = {
        "action_token_ids": action_ids,
        "physics_token_ids": aligned_physics,
        "episode_ids": keys[:, 0],
        "base_indices": keys[:, 1],
    }
    # Preserve task-qualified identity for full multi-task caches.  The first
    # key column is task_id when task_ids are present; this keeps the merged
    # cache joinable by the LeRobot dataset after episode IDs repeat per task.
    with np.load(action_cache, allow_pickle=False) as action_archive:
        if "task_ids" in action_archive:
            arrays["task_ids"] = np.asarray(action_archive["task_ids"], dtype=np.int64)
            if "task_names" in action_archive:
                arrays["task_names"] = np.asarray(action_archive["task_names"])
            arrays["episode_ids"] = keys[:, 1]
            arrays["base_indices"] = keys[:, 2]
    manifest = {
        "schema_version": 1,
        "kind": "action_and_physics_token_conditions",
        "action_cache": str(action_cache.resolve()),
        "action_cache_sha256": sha256_file(action_cache),
        "physics_cache": str(physics_cache.resolve()),
        "physics_cache_sha256": sha256_file(physics_cache),
        "physics_token_key": physics_token_key,
        "num_samples": len(keys),
        "num_action_tokens": action_ids.shape[1],
        "num_physics_tokens": aligned_physics.shape[1],
        "shuffle_physics_seed": shuffle_physics_seed,
    }
    return manifest, arrays


def main() -> None:
    args = parse_args()
    manifest, arrays = merge(
        args.action_cache.resolve(),
        args.physics_cache.resolve(),
        args.shuffle_physics_seed,
        args.physics_token_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
