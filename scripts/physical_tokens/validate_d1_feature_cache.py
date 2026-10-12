#!/usr/bin/env python3
"""Validate the exact D1 multi-task feature-cache contract without loading features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def validate_cache(
    cache_path: Path,
    task_names: tuple[str, ...],
    excluded_episode_ids: tuple[int, ...],
    *,
    episodes_per_task: int,
    validation_episodes_per_task: int,
    max_train_clips_per_task: int,
    expected_episode_ids: tuple[int, ...] | None = None,
) -> dict:
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"D1 cache or manifest is missing: {cache_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("D1 feature manifest must use schema version 2")
    if tuple(manifest.get("data_names", ())) != task_names:
        raise ValueError("D1 manifest task order does not match the requested tasks")
    if tuple(manifest.get("excluded_episode_ids", ())) != tuple(
        sorted(excluded_episode_ids)
    ):
        raise ValueError("D1 manifest episode exclusions do not match the policy split")
    entries = manifest.get("datasets", ())
    if len(entries) != len(task_names):
        raise ValueError("D1 manifest dataset count does not match task count")

    expected_episodes = {}
    for task_id, (task_name, entry) in enumerate(zip(task_names, entries)):
        if entry.get("task_id") != task_id or entry.get("data_name") != task_name:
            raise ValueError(f"D1 manifest dataset entry mismatch at task {task_id}")
        episode_ids = tuple(int(value) for value in entry.get("episode_ids", ()))
        if len(episode_ids) != episodes_per_task or len(set(episode_ids)) != len(episode_ids):
            raise ValueError(f"task {task_name} does not contain {episodes_per_task} unique episodes")
        if set(episode_ids).intersection(excluded_episode_ids):
            raise ValueError(f"task {task_name} contains a policy-held-out episode")
        if expected_episode_ids is not None and episode_ids != expected_episode_ids:
            raise ValueError(f"task {task_name} does not contain the expected episode IDs")
        expected_episodes[task_id] = episode_ids

    with np.load(cache_path, allow_pickle=False) as data:
        for key in ("task_ids", "task_names", "episode_ids", "future_valid_mask"):
            if key not in data.files:
                raise ValueError(f"D1 cache is missing {key}")
        cached_names = tuple(str(value) for value in data["task_names"].tolist())
        if cached_names != task_names:
            raise ValueError("D1 cache task_names differ from its contract")
        task_ids = np.asarray(data["task_ids"], dtype=np.int64)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
        complete = np.asarray(data["future_valid_mask"], dtype=np.bool_).all(axis=1)

    if not (len(task_ids) == len(episode_ids) == len(complete)):
        raise ValueError("D1 cache sample-key arrays have different lengths")
    if int(manifest.get("num_samples", -1)) != len(task_ids):
        raise ValueError("D1 manifest num_samples differs from the cache")
    expected_episode_count = len(task_names) * episodes_per_task
    if int(manifest.get("num_episodes", -1)) != expected_episode_count:
        raise ValueError("D1 manifest num_episodes differs from the contract")
    train_clips = validation_clips = capped_train_clips = 0
    validation_clips_per_task = {}
    for task_id, task_name in enumerate(task_names):
        task_mask = task_ids == task_id
        present = tuple(sorted(set(int(value) for value in episode_ids[task_mask])))
        expected = tuple(sorted(expected_episodes[task_id]))
        if present != expected:
            raise ValueError(f"task {task_name} cache episodes differ from the manifest")
        validation_ids = set(expected[-validation_episodes_per_task:])
        validation_mask = task_mask & np.isin(episode_ids, list(validation_ids)) & complete
        training_mask = task_mask & ~np.isin(episode_ids, list(validation_ids)) & complete
        task_train = int(training_mask.sum())
        task_validation = int(validation_mask.sum())
        train_clips += task_train
        validation_clips += task_validation
        capped_train_clips += min(task_train, max_train_clips_per_task)
        validation_clips_per_task[task_name] = task_validation

    report = {
        "cache": str(cache_path.resolve()),
        "num_tasks": len(task_names),
        "num_episodes": sum(len(values) for values in expected_episodes.values()),
        "num_unique_frames": int(manifest["num_unique_frames"]),
        "num_samples": len(task_ids),
        "train_complete_clips": train_clips,
        "validation_complete_clips": validation_clips,
        "capped_train_clips": capped_train_clips,
        "validation_clips_per_task": validation_clips_per_task,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-names", required=True)
    parser.add_argument("--excluded-episode-ids", default="")
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--expected-episode-ids", default=None)
    parser.add_argument("--validation-episodes-per-task", type=int, default=2)
    parser.add_argument("--max-train-clips-per-task", type=int, default=300)
    parser.add_argument("--expected-unique-frames", type=int, default=None)
    parser.add_argument("--expected-train-clips", type=int, default=None)
    parser.add_argument("--expected-validation-clips", type=int, default=None)
    parser.add_argument("--expected-capped-train-clips", type=int, default=None)
    args = parser.parse_args()
    report = validate_cache(
        args.input,
        parse_csv(args.data_names),
        tuple(int(value) for value in parse_csv(args.excluded_episode_ids)),
        episodes_per_task=args.episodes_per_task,
        validation_episodes_per_task=args.validation_episodes_per_task,
        max_train_clips_per_task=args.max_train_clips_per_task,
        expected_episode_ids=(
            tuple(int(value) for value in parse_csv(args.expected_episode_ids))
            if args.expected_episode_ids is not None
            else None
        ),
    )
    expected = {
        "num_unique_frames": args.expected_unique_frames,
        "train_complete_clips": args.expected_train_clips,
        "validation_complete_clips": args.expected_validation_clips,
        "capped_train_clips": args.expected_capped_train_clips,
    }
    for key, value in expected.items():
        if value is not None and report[key] != value:
            raise ValueError(f"D1 cache {key}={report[key]}, expected {value}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
