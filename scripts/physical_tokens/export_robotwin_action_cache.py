#!/usr/bin/env python3
"""Export normalized RoboTwin action chunks without decoding video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-name", default="click_alarmclock")
    parser.add_argument("--robot-type", default="robotwin_wrap32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_action_chunk(dataset, trajectory_id: int, base_index: int) -> np.ndarray:
    raw = {
        key: dataset.get_data_by_modality(trajectory_id, "action", key, base_index)
        for key in dataset.modality_keys["action"]
    }
    raw = dataset._apply_action_mode(raw)
    transformed = dataset.transforms(raw)
    parts = []
    for key in dataset.modality_keys["action"]:
        value = transformed[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        parts.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(parts, axis=1)


def export_cache(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    dataset_path = (args.data_root / args.data_name).resolve()
    dataset = make_LeRobotSingleDataset(
        args.data_root,
        args.data_name,
        args.robot_type,
        data_cfg={"video_backend": "pyav", "lerobot_version": "v2.0"},
    )
    dataset.transforms.eval()

    episode_ids = [int(value) for value in dataset.trajectory_ids]
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    allowed = set(episode_ids)
    selected_steps = [
        (int(episode_id), int(base_index))
        for episode_id, base_index in dataset.all_steps
        if int(episode_id) in allowed and int(base_index) % args.stride == 0
    ]
    if not selected_steps:
        raise ValueError("no action chunks matched the requested episode/stride selection")

    first_action_key = dataset.modality_keys["action"][0]
    action_offsets = np.asarray(dataset.delta_indices[first_action_key], dtype=np.int64)
    trajectory_lengths = {
        int(episode_id): int(length)
        for episode_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths)
    }
    actions = []
    valid_masks = []
    source_indices = []
    current_episode = None
    for sample_index, (episode_id, base_index) in enumerate(selected_steps, start=1):
        if current_episode != episode_id:
            dataset.curr_traj_data = dataset.get_trajectory_data(episode_id)
            dataset.curr_traj_id = episode_id
            current_episode = episode_id
        actions.append(transform_action_chunk(dataset, episode_id, base_index))
        target_indices = action_offsets + base_index
        valid_masks.append(
            np.logical_and(target_indices >= 0, target_indices < trajectory_lengths[episode_id])
        )
        source_indices.append((episode_id, base_index))
        if sample_index == 1 or sample_index % 5000 == 0:
            print(f"exported {sample_index}/{len(selected_steps)} chunks", flush=True)

    action_array = np.stack(actions).astype(np.float32)
    valid_mask = np.stack(valid_masks).astype(np.bool_)
    source_array = np.asarray(source_indices, dtype=np.int64)
    arrays = {
        "actions": action_array,
        "action_valid_mask": valid_mask,
        "action_dim_mask": np.ones(action_array.shape[-1], dtype=np.bool_),
        "episode_ids": source_array[:, 0],
        "base_indices": source_array[:, 1],
    }
    info_path = dataset_path / "meta" / "info.json"
    modality_path = dataset_path / "meta" / "modality.json"
    manifest = {
        "schema_version": 1,
        "data_root": str(args.data_root.resolve()),
        "data_name": args.data_name,
        "dataset_path": str(dataset_path),
        "robot_type": args.robot_type,
        "num_episodes": len(episode_ids),
        "num_chunks": len(action_array),
        "action_shape": list(action_array.shape),
        "stride": args.stride,
        "action_offsets": action_offsets.tolist(),
        "action_keys": list(dataset.modality_keys["action"]),
        "packed_layout": [
            "left_joint_0:6",
            "right_joint_6:12",
            "left_gripper_12",
            "right_gripper_13",
        ],
        "gripper_dims": [12, 13],
        "angular_dims": list(range(12)),
        "normalization": "robotwin_wrap32: wrap joints to [-pi, pi), binary grippers",
        "source_fingerprint": {
            "info.json": sha256_file(info_path),
            "modality.json": sha256_file(modality_path),
        },
    }
    return manifest, arrays


def main() -> None:
    args = parse_args()
    manifest, arrays = export_cache(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(f"wrote {manifest['num_chunks']} chunks to {args.output}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
