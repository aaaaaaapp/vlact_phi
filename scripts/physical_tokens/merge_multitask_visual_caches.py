#!/usr/bin/env python3
"""Merge single-task visual caches and verify exact action-cache key alignment."""

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-names", nargs="+", required=True)
    parser.add_argument("--action-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.task_names):
        raise ValueError("--inputs and --task-names must have equal lengths")
    if len(set(args.task_names)) != len(args.task_names):
        raise ValueError("task names must be unique")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    frame_features = []
    source_frame_indices = []
    future_valid_masks = []
    task_ids = []
    episode_ids = []
    base_indices = []
    input_records = []
    frame_offset = 0
    reference = None

    for expected_task_id, (path, expected_name) in enumerate(zip(args.inputs, args.task_names)):
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        with np.load(path, allow_pickle=False) as cache:
            names = [str(value) for value in cache["task_names"].tolist()]
            if names != [expected_name] or manifest.get("data_names") != [expected_name]:
                raise ValueError(f"task mismatch in {path}: {names} / {manifest.get('data_names')}")
            current = {
                "offsets": cache["offsets"].copy(),
                "feature_tail": tuple(cache["frame_features"].shape[1:]),
                "feature_dtype": str(cache["frame_features"].dtype),
                "camera": manifest["camera"],
                "base_vlm_config_sha256": manifest["base_vlm_config_sha256"],
                "checkpoint_sha256": manifest["pretrained_checkpoint"]["sha256"],
                "stride": int(manifest["stride"]),
            }
            if reference is None:
                reference = current
            else:
                for key in current:
                    if key == "offsets":
                        if not np.array_equal(current[key], reference[key]):
                            raise ValueError(f"offset mismatch in {path}")
                    elif current[key] != reference[key]:
                        raise ValueError(f"{key} mismatch in {path}")

            local_task_ids = cache["task_ids"]
            if not np.all(local_task_ids == 0):
                raise ValueError(f"single-task cache has nonzero task IDs: {path}")
            features = cache["frame_features"].copy()
            indices = cache["source_frame_indices"].copy()
            valid = cache["future_valid_mask"].copy()
            episodes = cache["episode_ids"].copy()
            bases = cache["base_indices"].copy()
            if indices.shape != valid.shape or indices.shape[0] != len(episodes):
                raise ValueError(f"row/temporal shape mismatch in {path}")
            if indices.min() < 0 or indices.max() >= len(features):
                raise ValueError(f"source frame index out of range in {path}")
            keys = np.stack(
                [np.full(len(episodes), expected_task_id), episodes, bases], axis=1
            )
            if len(np.unique(keys, axis=0)) != len(keys):
                raise ValueError(f"duplicate qualified keys in {path}")

            frame_features.append(features)
            source_frame_indices.append(indices + frame_offset)
            future_valid_masks.append(valid)
            task_ids.append(keys[:, 0])
            episode_ids.append(episodes)
            base_indices.append(bases)
            frame_offset += len(features)
            input_records.append(
                {
                    "task_id": expected_task_id,
                    "task_name": expected_name,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "num_samples": len(episodes),
                    "num_unique_frames": len(features),
                    "episode_ids": manifest["datasets"][0]["episode_ids"],
                }
            )

    arrays = {
        "frame_features": np.concatenate(frame_features, axis=0),
        "source_frame_indices": np.concatenate(source_frame_indices, axis=0),
        "future_valid_mask": np.concatenate(future_valid_masks, axis=0),
        "task_ids": np.concatenate(task_ids, axis=0),
        "task_names": np.asarray(args.task_names),
        "episode_ids": np.concatenate(episode_ids, axis=0),
        "base_indices": np.concatenate(base_indices, axis=0),
        "offsets": reference["offsets"],
    }
    visual_keys = np.stack(
        [arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"]], axis=1
    )
    with np.load(args.action_cache, allow_pickle=False) as action:
        action_names = [str(value) for value in action["task_names"].tolist()]
        action_keys = np.stack(
            [action["task_ids"], action["episode_ids"], action["base_indices"]], axis=1
        )
    if action_names != args.task_names:
        raise ValueError(f"action task order mismatch: {action_names}")
    if not np.array_equal(visual_keys, action_keys):
        mismatch = np.flatnonzero(np.any(visual_keys != action_keys, axis=1))
        first = int(mismatch[0]) if len(mismatch) else min(len(visual_keys), len(action_keys))
        raise ValueError(
            f"visual/action key mismatch at row {first}; shapes={visual_keys.shape}/{action_keys.shape}"
        )
    if arrays["source_frame_indices"].max() >= len(arrays["frame_features"]):
        raise RuntimeError("merged source frame indices are out of range")

    manifest = {
        "schema_version": 1,
        "kind": "robotwin_merged_multitask_visual_features",
        "task_names": args.task_names,
        "num_tasks": len(args.task_names),
        "num_samples": len(visual_keys),
        "num_unique_frames": len(arrays["frame_features"]),
        "frame_feature_shape": list(arrays["frame_features"].shape),
        "frame_feature_dtype": str(arrays["frame_features"].dtype),
        "offsets": arrays["offsets"].tolist(),
        "camera": reference["camera"],
        "stride": reference["stride"],
        "base_vlm_config_sha256": reference["base_vlm_config_sha256"],
        "pretrained_checkpoint_sha256": reference["checkpoint_sha256"],
        "action_cache": str(args.action_cache.resolve()),
        "action_cache_sha256": sha256_file(args.action_cache),
        "exact_action_key_alignment": True,
        "inputs": input_records,
    }
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
