#!/usr/bin/env python3
"""Merge task-local action caches with explicit task-qualified keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.task_names):
        raise ValueError("inputs and task names must have equal length")
    parts = []
    manifests = []
    expected_shape = None
    for task_id, (path, task_name) in enumerate(zip(args.inputs, args.task_names)):
        path = path.resolve()
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["data_name"] != task_name:
            raise ValueError(f"task name mismatch for {path}: {manifest['data_name']}")
        with np.load(path, allow_pickle=False) as archive:
            part = {key: np.asarray(archive[key]) for key in archive.files}
        shape = tuple(part["actions"].shape[1:])
        if expected_shape is None:
            expected_shape = shape
        if shape != expected_shape:
            raise ValueError(f"action shape mismatch for {task_name}: {shape}")
        count = len(part["actions"])
        row_keys = ("episode_ids", "base_indices", "action_valid_mask")
        if any(len(part[key]) != count for key in row_keys):
            raise ValueError(f"row count mismatch for {task_name}")
        parts.append(part)
        manifests.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "path": str(path),
                "sha256": sha256_file(path),
                "num_samples": count,
                "num_episodes": int(np.unique(part["episode_ids"]).size),
                "source_manifest_sha256": sha256_file(manifest_path),
            }
        )
    arrays = {
        "actions": np.concatenate([part["actions"] for part in parts]),
        "action_valid_mask": np.concatenate([part["action_valid_mask"] for part in parts]),
        "action_dim_mask": parts[0]["action_dim_mask"],
        "task_ids": np.concatenate(
            [
                np.full(len(part["actions"]), task_id, dtype=np.int64)
                for task_id, part in enumerate(parts)
            ]
        ),
        "task_names": np.asarray(args.task_names),
        "episode_ids": np.concatenate([part["episode_ids"] for part in parts]),
        "base_indices": np.concatenate([part["base_indices"] for part in parts]),
    }
    qualified_keys = list(
        zip(
            arrays["task_ids"].tolist(),
            arrays["episode_ids"].tolist(),
            arrays["base_indices"].tolist(),
        )
    )
    if len(set(qualified_keys)) != len(qualified_keys):
        raise ValueError("merged cache has duplicate task-qualified keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "robotwin_multitask_action_chunks",
        "tasks": manifests,
        "num_tasks": len(parts),
        "num_samples": len(arrays["actions"]),
        "action_shape": list(arrays["actions"].shape),
        "key": ["task_id", "episode_id", "base_index"],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
