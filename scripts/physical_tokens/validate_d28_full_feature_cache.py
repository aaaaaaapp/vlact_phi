#!/usr/bin/env python3
"""Validate the merged D28 full-episode spatiotemporal feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_OFFSETS = tuple(range(0, 33, 2))
VALIDATION_EPISODES = (5, 20, 25, 39, 49)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualified_keys(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.stack(
        [data["task_ids"], data["episode_ids"], data["base_indices"]], axis=1
    ).astype(np.int64, copy=False)


def validate_cache(cache_path: Path, action_cache_path: Path) -> dict:
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"D28 cache or manifest is missing: {cache_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "robotwin_merged_multitask_visual_features":
        raise ValueError("unexpected D28 manifest kind")
    if manifest.get("exact_action_key_alignment") is not True:
        raise ValueError("manifest does not assert exact action-key alignment")

    with np.load(cache_path, allow_pickle=False) as data:
        required = {
            "frame_features",
            "source_frame_indices",
            "future_valid_mask",
            "task_ids",
            "task_names",
            "episode_ids",
            "base_indices",
            "offsets",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"D28 cache is missing arrays: {sorted(missing)}")
        features = np.asarray(data["frame_features"])
        indices = np.asarray(data["source_frame_indices"], dtype=np.int64)
        valid = np.asarray(data["future_valid_mask"], dtype=np.bool_)
        keys = qualified_keys(data)
        names = tuple(str(value) for value in data["task_names"].tolist())
        offsets = tuple(int(value) for value in data["offsets"].tolist())

    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError(f"expected 8 unique tasks, got {names}")
    if offsets != EXPECTED_OFFSETS:
        raise ValueError(f"unexpected offsets: {offsets}")
    if features.ndim != 3 or features.shape[1] != 16:
        raise ValueError(f"expected frame features [F,16,D], got {features.shape}")
    if features.dtype != np.float16 or not np.isfinite(features).all():
        raise ValueError("frame features must be finite float16")
    if indices.shape != valid.shape or indices.shape != (len(keys), len(offsets)):
        raise ValueError("temporal arrays do not match sample keys and offsets")
    if indices.min() < 0 or indices.max() >= len(features):
        raise ValueError("source frame index outside frame_features")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("D28 cache contains duplicate qualified keys")

    episode_counts = {}
    for task_id, name in enumerate(names):
        episodes = tuple(sorted(set(keys[keys[:, 0] == task_id, 1].tolist())))
        if episodes != tuple(range(50)):
            raise ValueError(f"task {name} does not contain exact episodes 0..49")
        episode_counts[name] = len(episodes)

    with np.load(action_cache_path, allow_pickle=False) as action:
        action_names = tuple(str(value) for value in action["task_names"].tolist())
        action_keys = qualified_keys(action)
    if names != action_names or not np.array_equal(keys, action_keys):
        raise ValueError("D28 qualified keys are not exactly aligned to the action cache")

    complete = valid.all(axis=1)
    validation = np.isin(keys[:, 1], VALIDATION_EPISODES) & complete
    training = ~np.isin(keys[:, 1], VALIDATION_EPISODES) & complete
    if not training.any() or not validation.any():
        raise ValueError("fixed episode-level training/validation split is empty")
    if int(manifest.get("num_samples", -1)) != len(keys):
        raise ValueError("manifest num_samples does not match cache")
    if int(manifest.get("num_unique_frames", -1)) != len(features):
        raise ValueError("manifest num_unique_frames does not match cache")

    return {
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "manifest_sha256": sha256_file(manifest_path),
        "action_cache": str(action_cache_path.resolve()),
        "task_names": list(names),
        "episodes_per_task": episode_counts,
        "num_samples": len(keys),
        "num_unique_frames": len(features),
        "frame_feature_shape": list(features.shape),
        "complete_clips": int(complete.sum()),
        "train_complete_clips": int(training.sum()),
        "validation_complete_clips": int(validation.sum()),
        "validation_episode_ids": list(VALIDATION_EPISODES),
        "exact_action_key_alignment": True,
        "all_features_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate_cache(args.input, args.action_cache)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
