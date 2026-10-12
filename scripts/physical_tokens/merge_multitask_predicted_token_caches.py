#!/usr/bin/env python3
"""Merge single-task predicted-token parts and verify oracle-cache key alignment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from export_robotwin_future_features import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-names", nargs="+", required=True)
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.task_names):
        raise ValueError("input/task counts differ")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    ids, physics_ids, task_ids, episodes, bases, records = [], [], [], [], [], []
    has_physics = None
    reference = None
    for task_id, (path, name) in enumerate(zip(args.inputs, args.task_names)):
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        with np.load(path, allow_pickle=False) as cache:
            if cache["task_names"].tolist() != [name] or manifest["data_name"] != name:
                raise ValueError(f"task mismatch in {path}")
            current = (manifest["checkpoint_sha256"], manifest["source_token_cache_sha256"])
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError(f"model/source mismatch in {path}")
            part_ids = cache["action_token_ids"].copy()
            part_has_physics = "physics_token_ids" in cache.files
            if has_physics is None:
                has_physics = part_has_physics
            elif has_physics != part_has_physics:
                raise ValueError("predicted-token parts disagree on physics-token presence")
            if part_has_physics:
                physics_ids.append(cache["physics_token_ids"].copy())
            part_episodes = cache["episode_ids"].copy()
            part_bases = cache["base_indices"].copy()
        ids.append(part_ids)
        task_ids.append(np.full(len(part_ids), task_id, dtype=np.int64))
        episodes.append(part_episodes)
        bases.append(part_bases)
        records.append({
            "task_id": task_id,
            "task_name": name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "manifest_sha256": sha256_file(manifest_path),
            "num_samples": len(part_ids),
        })
    arrays = {
        "action_token_ids": np.concatenate(ids),
        "task_ids": np.concatenate(task_ids),
        "task_names": np.asarray(args.task_names),
        "episode_ids": np.concatenate(episodes),
        "base_indices": np.concatenate(bases),
    }
    if has_physics:
        arrays["physics_token_ids"] = np.concatenate(physics_ids)
    keys = np.stack([arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"]], axis=1)
    with np.load(args.oracle_cache, allow_pickle=False) as oracle:
        oracle_keys = np.stack(
            [oracle["task_ids"], oracle["episode_ids"], oracle["base_indices"]], axis=1
        )
        oracle_names = oracle["task_names"].tolist()
    if oracle_names != args.task_names or not np.array_equal(keys, oracle_keys):
        raise ValueError("predicted-token keys are not exactly aligned with oracle cache")
    manifest = {
        "schema_version": 1,
        "kind": (
            "multitask_observation_predicted_dual_tokens"
            if has_physics
            else "multitask_observation_predicted_action_tokens"
        ),
        "task_names": args.task_names,
        "num_samples": len(keys),
        "token_shape": list(arrays["action_token_ids"].shape),
        "physics_token_shape": (
            list(arrays["physics_token_ids"].shape) if has_physics else None
        ),
        "checkpoint_sha256": reference[0],
        "oracle_cache": str(args.oracle_cache.resolve()),
        "oracle_cache_sha256": sha256_file(args.oracle_cache),
        "exact_oracle_key_alignment": True,
        "inputs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
