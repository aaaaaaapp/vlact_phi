#!/usr/bin/env python3
"""Align the D19 predictable-residual teacher to a 17-frame feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


KEY_ARRAYS = ("task_ids", "task_names", "episode_ids", "base_indices", "offsets")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_key_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(KEY_ARRAYS).difference(data.files)
        if missing:
            raise ValueError(f"key template is missing arrays: {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in KEY_ARRAYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-template", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    arrays = load_key_arrays(args.key_template)
    with np.load(args.teacher_cache, allow_pickle=False) as data:
        teacher = np.asarray(data["actions"], dtype=np.float32)
        teacher_keys = list(
            zip(data["task_ids"].tolist(), data["episode_ids"].tolist(), data["base_indices"].tolist())
        )
    rows = {tuple(map(int, key)): index for index, key in enumerate(teacher_keys)}
    if len(rows) != len(teacher_keys):
        raise ValueError("teacher cache has duplicate qualified keys")
    template_keys = list(
        zip(arrays["task_ids"].tolist(), arrays["episode_ids"].tolist(), arrays["base_indices"].tolist())
    )
    indices = np.asarray([rows[tuple(map(int, key))] for key in template_keys], dtype=np.int64)
    offsets = arrays["offsets"].astype(np.int64)
    if offsets.tolist() != list(range(0, 33, 2)):
        raise ValueError(f"unexpected spatiotemporal offsets: {offsets.tolist()}")
    decoder_indices = np.minimum(offsets[1:], teacher.shape[1] - 1)
    action_targets = np.zeros((len(indices), len(offsets), teacher.shape[2]), dtype=np.float32)
    action_targets[:, 1:] = teacher[indices[:, None], decoder_indices[None, :]]
    output_arrays = {**arrays, "action_targets": action_targets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "spatiotemporal_predictable_residual_teacher_sidecar",
        "key_template": str(args.key_template.resolve()),
        "key_template_sha256": sha256_file(args.key_template),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": sha256_file(args.teacher_cache),
        "num_samples": len(indices),
        "shape": list(action_targets.shape),
        "decoder_indices": decoder_indices.tolist(),
        "exact_qualified_key_join": True,
    }
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
