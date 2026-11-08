#!/usr/bin/env python3
"""Join per-task Randomized targets, then combine them with the Clean cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in names}


def write(path: Path, arrays: dict[str, np.ndarray], report: dict) -> None:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() and manifest.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    # A killed export can leave only one side of the pair. Both outputs are
    # derived, so a restart may replace that incomplete pair after recomputing.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    manifest_temporary = manifest.with_name(manifest.name + ".tmp")
    manifest_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest)


def join_task(args: argparse.Namespace) -> None:
    action = read(args.action, ("action_token_ids", "episode_ids", "base_indices"))
    physics = read(args.physics, ("physics_token_ids", "episode_ids", "base_indices", "task_names"))
    feature = read(args.feature, ("future_valid_mask", "episode_ids", "base_indices", "task_names"))
    name = f"Randomized/{args.task}"
    if physics["task_names"].tolist() != [name] or feature["task_names"].tolist() != [name]:
        raise ValueError(f"physics task name does not match {name}")
    for key in ("episode_ids", "base_indices"):
        if not np.array_equal(physics[key], feature[key]):
            raise ValueError(f"physics IDs and features disagree on {key}")
    if action["action_token_ids"].ndim != 2 or action["action_token_ids"].shape[1] != 32:
        raise ValueError("action token width must be 32")
    if physics["physics_token_ids"].ndim != 2 or physics["physics_token_ids"].shape[1] != 64:
        raise ValueError("physics token width must be 64")
    action_keys = list(zip(action["episode_ids"].tolist(), action["base_indices"].tolist()))
    physics_keys = list(zip(physics["episode_ids"].tolist(), physics["base_indices"].tolist()))
    if len(set(action_keys)) != len(action_keys) or len(set(physics_keys)) != len(physics_keys):
        raise ValueError("duplicate sample keys")
    action_row = {key: row for row, key in enumerate(action_keys)}
    complete = np.asarray(feature["future_valid_mask"], dtype=bool).all(axis=1)
    selected = np.flatnonzero(complete)
    missing = [physics_keys[row] for row in selected if physics_keys[row] not in action_row]
    if missing:
        raise ValueError(f"action IDs missing {len(missing)} complete physics clips; first={missing[0]}")
    action_rows = np.asarray([action_row[physics_keys[row]] for row in selected], dtype=np.int64)
    arrays = {
        "action_token_ids": action["action_token_ids"][action_rows].astype(np.int32),
        "physics_token_ids": physics["physics_token_ids"][selected].astype(np.int32),
        "episode_ids": physics["episode_ids"][selected].astype(np.int64),
        "base_indices": physics["base_indices"][selected].astype(np.int64),
    }
    if len(arrays["action_token_ids"]) == 0:
        raise ValueError(f"{name} has no complete clips")
    report = {"task": name, "samples": len(selected), "full_future_only": True,
              "action": str(args.action), "physics": str(args.physics), "feature": str(args.feature)}
    shard_index = getattr(args, "shard_index", None)
    if shard_index is not None:
        shard_count = getattr(args, "shard_count", None)
        if shard_count is None or not 0 <= shard_index < shard_count:
            raise ValueError("join-task needs a valid shard index and shard count")
        report.update(shard_index=shard_index, shard_count=shard_count)
    write(args.output, arrays, report)
    print(f"{name}: {len(selected)} complete clips", flush=True)


def combine_shards(args: argparse.Namespace) -> None:
    """Reassemble complete, disjoint episode shards into the original task cache."""
    parts = []
    for shard_index in range(args.shard_count):
        path = args.shard_dir / f"{args.task}.shard{shard_index}.npz"
        part = read(path, ("action_token_ids", "physics_token_ids", "episode_ids", "base_indices"))
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("task") != f"Randomized/{args.task}"
                or manifest.get("shard_index") != shard_index
                or manifest.get("shard_count") != args.shard_count
                or manifest.get("samples") != len(part["episode_ids"])):
            raise ValueError(f"shard identity or sample count mismatch in {manifest_path}")
        if (part["action_token_ids"].shape[1:] != (32,)
                or part["physics_token_ids"].shape[1:] != (64,)):
            raise ValueError(f"unexpected token widths in {path}")
        parts.append(part)
    keys = ("action_token_ids", "physics_token_ids", "episode_ids", "base_indices")
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    sample_keys = list(zip(arrays["episode_ids"].tolist(), arrays["base_indices"].tolist()))
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError(f"Randomized/{args.task} shards contain duplicate sample keys")
    order = np.lexsort((arrays["base_indices"], arrays["episode_ids"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    write(args.output, arrays, {"task": f"Randomized/{args.task}",
                               "samples": len(sample_keys), "full_future_only": True,
                               "shard_count": args.shard_count})
    print(f"Randomized/{args.task}: merged {args.shard_count} shards, {len(sample_keys)} clips", flush=True)


def combine(args: argparse.Namespace) -> None:
    clean = read(args.clean, ("action_token_ids", "physics_token_ids", "task_ids", "task_names",
                              "episode_ids", "base_indices"))
    tasks = [line.strip() for line in args.task_list.read_text().splitlines() if line.strip()]
    if len(tasks) != 50 or clean["task_names"].tolist() != tasks:
        raise ValueError("Clean cache/task list must contain the same 50 tasks in order")
    keys = ("action_token_ids", "physics_token_ids", "task_ids", "episode_ids", "base_indices")
    parts = [{key: clean[key] for key in keys}]
    for task_id, task in enumerate(tasks, start=50):
        path = args.random_dir / f"{task}.npz"
        part = read(path, ("action_token_ids", "physics_token_ids", "episode_ids", "base_indices"))
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("task") != f"Randomized/{task}" or manifest.get("samples") != len(part["episode_ids"]):
            raise ValueError(f"task identity or sample count mismatch in {manifest_path}")
        if part["action_token_ids"].shape[1:] != (32,) or part["physics_token_ids"].shape[1:] != (64,):
            raise ValueError(f"unexpected token widths in {path}")
        part["task_ids"] = np.full(len(part["episode_ids"]), task_id, dtype=np.int64)
        parts.append(part)
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    arrays["task_names"] = np.asarray(tasks + [f"Randomized/{task}" for task in tasks])
    if len(set(zip(arrays["task_ids"].tolist(), arrays["episode_ids"].tolist(),
                   arrays["base_indices"].tolist()))) != len(arrays["task_ids"]):
        raise ValueError("combined cache has duplicate qualified keys")
    report = {"kind": "clean_randomized_dual_targets", "clean_samples": len(clean["episode_ids"]),
              "randomized_samples": len(arrays["episode_ids"]) - len(clean["episode_ids"]),
              "tasks_per_domain": 50, "randomized_frame_stride": 4,
              "full_future_only": True, "clean_cache": str(args.clean)}
    write(args.output, arrays, report)
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    task = subparsers.add_parser("join-task")
    task.add_argument("--task", required=True)
    task.add_argument("--action", type=Path, required=True)
    task.add_argument("--physics", type=Path, required=True)
    task.add_argument("--feature", type=Path, required=True)
    task.add_argument("--output", type=Path, required=True)
    task.add_argument("--shard-index", type=int)
    task.add_argument("--shard-count", type=int)
    shards = subparsers.add_parser("combine-shards")
    shards.add_argument("--task", required=True)
    shards.add_argument("--shard-dir", type=Path, required=True)
    shards.add_argument("--shard-count", type=int, required=True)
    shards.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--clean", type=Path, required=True)
    combined.add_argument("--random-dir", type=Path, required=True)
    combined.add_argument("--task-list", type=Path, required=True)
    combined.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"join-task": join_task, "combine-shards": combine_shards,
     "combine": combine}[args.mode](args)


if __name__ == "__main__":
    main()
