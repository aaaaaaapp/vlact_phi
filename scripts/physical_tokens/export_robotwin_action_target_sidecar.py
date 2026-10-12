#!/usr/bin/env python3
"""Export key-aligned sampled action targets for a spatiotemporal feature cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_actions(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"expected 14-D actions in {path}, got {actions.shape}")
    actions[:, :12] = np.remainder(actions[:, :12] + np.pi, 2.0 * np.pi) - np.pi
    actions[:, 12:14] = (actions[:, 12:14] > 0.49).astype(np.float32)
    return actions


def main() -> None:
    args = parse_args()
    with np.load(args.feature_cache.resolve(), allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in ("episode_ids", "base_indices", "offsets")
        }
        if "task_ids" in data.files or "task_names" in data.files:
            if "task_ids" not in data.files or "task_names" not in data.files:
                raise ValueError("feature cache must provide both task_ids and task_names")
            arrays["task_ids"] = np.asarray(data["task_ids"], dtype=np.int64)
            arrays["task_names"] = np.asarray(data["task_names"])
        else:
            arrays["task_ids"] = np.zeros_like(arrays["episode_ids"], dtype=np.int64)
            arrays["task_names"] = np.asarray([args.data_root.resolve().name])
    targets = np.empty(
        (len(arrays["episode_ids"]), len(arrays["offsets"]), 14), dtype=np.float32
    )
    cache: dict[tuple[int, int], np.ndarray] = {}
    for row, (task_id, episode_id, base_index) in enumerate(
        zip(arrays["task_ids"], arrays["episode_ids"], arrays["base_indices"])
    ):
        key = (int(task_id), int(episode_id))
        if key not in cache:
            task_name = str(arrays["task_names"][key[0]])
            task_root = args.data_root / task_name
            if not task_root.is_dir():
                task_root = args.data_root
            matches = sorted(
                task_root.glob(f"data/**/episode_{key[1]:06d}.parquet")
            )
            if len(matches) != 1:
                raise ValueError(f"expected one parquet for {key}, got {matches}")
            cache[key] = load_actions(matches[0])
        actions = cache[key]
        indices = np.clip(int(base_index) + arrays["offsets"], 0, len(actions) - 1)
        targets[row] = actions[indices]
    output_arrays = {**arrays, "action_targets": targets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "spatiotemporal_action_target_sidecar",
        "feature_cache": str(args.feature_cache.resolve()),
        "data_root": str(args.data_root.resolve()),
        "num_samples": len(targets),
        "shape": list(targets.shape),
        "definition": "wrapped absolute joint targets and binary grippers sampled at visual offsets",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
