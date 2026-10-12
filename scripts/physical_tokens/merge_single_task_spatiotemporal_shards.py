#!/usr/bin/env python3
"""Merge disjoint episode shards into one single-task feature cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from merge_multitask_visual_caches import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    features_list = []
    indices_list = []
    valid_list = []
    episodes_list = []
    bases_list = []
    records = []
    reference = None
    frame_offset = 0
    for path in args.inputs:
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(value) for value in data["task_names"].tolist())
            current = (
                names,
                tuple(data["offsets"].tolist()),
                tuple(data["frame_features"].shape[1:]),
                str(data["frame_features"].dtype),
                manifest["camera"],
                manifest["base_vlm_config_sha256"],
                manifest["pretrained_checkpoint"]["sha256"],
            )
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError(f"shard contract mismatch: {path}")
            if names != tuple(manifest["data_names"]) or len(names) != 1:
                raise ValueError(f"expected one matching task in {path}")
            if not np.all(data["task_ids"] == 0):
                raise ValueError(f"single-task shard has nonzero task IDs: {path}")
            features = np.asarray(data["frame_features"])
            indices = np.asarray(data["source_frame_indices"], dtype=np.int64)
            valid = np.asarray(data["future_valid_mask"], dtype=np.bool_)
            episodes = np.asarray(data["episode_ids"], dtype=np.int64)
            bases = np.asarray(data["base_indices"], dtype=np.int64)
            features_list.append(features)
            indices_list.append(indices + frame_offset)
            valid_list.append(valid)
            episodes_list.append(episodes)
            bases_list.append(bases)
            frame_offset += len(features)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "manifest_sha256": sha256_file(manifest_path),
                "episode_ids": manifest["datasets"][0]["episode_ids"],
            }
        )

    episodes = np.concatenate(episodes_list)
    bases = np.concatenate(bases_list)
    order = np.lexsort((bases, episodes))
    keys = np.stack((episodes[order], bases[order]), axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("shards contain duplicate episode/base keys")
    all_episodes = sorted(set(int(value) for value in episodes.tolist()))
    arrays = {
        "frame_features": np.concatenate(features_list),
        "source_frame_indices": np.concatenate(indices_list)[order],
        "future_valid_mask": np.concatenate(valid_list)[order],
        "task_ids": np.zeros(len(order), dtype=np.int64),
        "task_names": np.asarray(reference[0]),
        "episode_ids": episodes[order],
        "base_indices": bases[order],
        "offsets": np.asarray(reference[1], dtype=np.int64),
    }
    manifest = {
        "schema_version": 2,
        "kind": "robotwin_deduplicated_spatiotemporal_visual_features",
        "data_names": list(reference[0]),
        "datasets": [
            {
                "task_id": 0,
                "data_name": reference[0][0],
                "episode_ids": all_episodes,
                "num_episodes": len(all_episodes),
                "num_samples": len(order),
            }
        ],
        "camera": reference[4],
        "offsets": list(reference[1]),
        "num_episodes": len(all_episodes),
        "num_samples": len(order),
        "num_unique_frames": len(arrays["frame_features"]),
        "frame_feature_shape": list(arrays["frame_features"].shape),
        "frame_feature_dtype": str(arrays["frame_features"].dtype),
        "base_vlm_config_sha256": reference[5],
        "pretrained_checkpoint": {"sha256": reference[6]},
        "stride": 1,
        "shards": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
