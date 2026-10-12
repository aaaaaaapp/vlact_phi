#!/usr/bin/env python3
"""Export one task's free-running action-token predictions with qualified keys."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_action_token_reasoner import load_reasoner
from evaluate_qwenpi_oracle import sha256_file
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-token-cache", type=Path, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verify-first-batch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(os.environ.get("D17_EXPORT_CPU_THREADS", "4")))
    cv2.setNumThreads(1)
    config = OmegaConf.load(args.config)
    config.datasets.vla_data.discrete_token_cache = str(args.source_token_cache.resolve())
    data_cfg = config.datasets.vla_data
    data_cfg.exclude_episode_ids = None
    data_cfg.include_episode_ids = list(range(50))
    dataset = make_LeRobotSingleDataset(
        Path(str(data_cfg.data_root_dir)),
        args.data_name,
        "robotwin_wrap32",
        data_cfg=data_cfg,
    )
    transforms = getattr(dataset, "transforms", None)
    if transforms is not None:
        transforms.eval()
    device = torch.device(args.device)
    model = load_reasoner(config, args.checkpoint.resolve(), device)
    raw_dataset = dataset.dataset
    bases_by_episode: dict[int, list[int]] = {
        int(episode): [] for episode in raw_dataset.trajectory_ids
    }
    for episode, base in raw_dataset.all_steps:
        bases_by_episode[int(episode)].append(int(base))
    lengths = {
        int(episode): int(length)
        for episode, length in zip(raw_dataset.trajectory_ids, raw_dataset.trajectory_lengths)
    }
    action_predictions = []
    physics_predictions = []
    exported_keys = []
    completed = 0
    verified = False
    first_batch_prediction_agreement = None
    row_by_key = {
        (int(episode), int(base)): row
        for row, (episode, base) in enumerate(raw_dataset.all_steps)
    }
    original_get_video = raw_dataset.get_video
    original_delta_indices = {
        camera: raw_dataset.delta_indices[camera]
        for camera in raw_dataset.modality_keys["video"]
    }
    try:
        for episode in raw_dataset.trajectory_ids:
            episode = int(episode)
            length = lengths[episode]
            raw_dataset.curr_traj_data = raw_dataset.get_trajectory_data(episode)
            raw_dataset.curr_traj_id = episode
            frames_by_camera = {}
            for camera in raw_dataset.modality_keys["video"]:
                raw_dataset.delta_indices[camera] = np.arange(length, dtype=np.int64)
                frames_by_camera[camera] = original_get_video(episode, camera, 0)
                raw_dataset.delta_indices[camera] = original_delta_indices[camera]

            def cached_get_video(self, trajectory_id, key, base_index):
                if int(trajectory_id) != episode:
                    raise ValueError("cached video requested for a different episode")
                indices = np.asarray(self.delta_indices[key], dtype=np.int64) + int(base_index)
                indices = np.clip(indices, 0, length - 1)
                return frames_by_camera[key][indices]

            raw_dataset.get_video = types.MethodType(cached_get_video, raw_dataset)
            bases = bases_by_episode[episode]
            for start in range(0, len(bases), args.batch_size):
                batch_bases = bases[start : start + args.batch_size]
                batch_rows = [row_by_key[(episode, base)] for base in batch_bases]
                samples = [dataset[index] for index in batch_rows]
                slow_samples = None
                if args.verify_first_batch and not verified:
                    raw_dataset.get_video = original_get_video
                    slow_samples = [dataset[index] for index in batch_rows]
                    raw_dataset.get_video = types.MethodType(cached_get_video, raw_dataset)
                    for fast, slow in zip(samples, slow_samples):
                        if fast["lang"] != slow["lang"] or any(
                            not np.array_equal(np.asarray(a), np.asarray(b))
                            for a, b in zip(fast["image"], slow["image"])
                        ):
                            raise RuntimeError("fast episode decode changed first-batch inputs")
                batch_output = model.predict_token_ids(samples)
                if isinstance(batch_output, dict):
                    batch_action = np.asarray(batch_output["action_token_ids"], dtype=np.int32)
                    batch_physics = np.asarray(batch_output["physics_token_ids"], dtype=np.int32)
                else:
                    batch_action = np.asarray(batch_output, dtype=np.int32)
                    batch_physics = None
                if slow_samples is not None:
                    slow_output = model.predict_token_ids(slow_samples)
                    slow_action = np.asarray(
                        slow_output["action_token_ids"]
                        if isinstance(slow_output, dict)
                        else slow_output,
                        dtype=np.int32,
                    )
                    first_batch_prediction_agreement = float(
                        np.mean(batch_action == slow_action)
                    )
                    if batch_physics is not None:
                        slow_physics = np.asarray(
                            slow_output["physics_token_ids"], dtype=np.int32
                        )
                        first_batch_prediction_agreement = min(
                            first_batch_prediction_agreement,
                            float(np.mean(batch_physics == slow_physics)),
                        )
                    verified = True
                    print(
                        f"{args.data_name}: fast/slow inputs exact; repeated-forward "
                        f"token agreement={first_batch_prediction_agreement:.6f}",
                        flush=True,
                    )
                action_predictions.append(batch_action)
                if batch_physics is not None:
                    physics_predictions.append(batch_physics)
                exported_keys.extend((episode, base) for base in batch_bases)
                completed += len(batch_bases)
                if completed == len(batch_bases) or completed % 400 < len(batch_bases):
                    print(f"{args.data_name}: {completed}/{len(dataset)}", flush=True)
    finally:
        raw_dataset.get_video = original_get_video
        for camera, indices in original_delta_indices.items():
            raw_dataset.delta_indices[camera] = indices
    action_token_ids = np.concatenate(action_predictions)
    keys = np.asarray(exported_keys, dtype=np.int64)
    if not np.array_equal(keys, np.asarray(raw_dataset.all_steps, dtype=np.int64)):
        raise RuntimeError("fast export order differs from dataset.all_steps")
    arrays = {
        "action_token_ids": action_token_ids,
        "episode_ids": keys[:, 0],
        "base_indices": keys[:, 1],
        "task_names": np.asarray([args.data_name]),
    }
    if physics_predictions:
        arrays["physics_token_ids"] = np.concatenate(physics_predictions)
    manifest = {
        "schema_version": 1,
        "kind": (
            "single_task_observation_predicted_dual_tokens"
            if physics_predictions
            else "single_task_observation_predicted_action_tokens"
        ),
        "data_name": args.data_name,
        "num_samples": len(dataset),
        "action_token_shape": list(action_token_ids.shape),
        "physics_token_shape": (
            list(arrays["physics_token_ids"].shape) if physics_predictions else None
        ),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "source_token_cache": str(args.source_token_cache.resolve()),
        "source_token_cache_sha256": sha256_file(args.source_token_cache.resolve()),
        "augmentation_mode": "eval",
        "video_decode_mode": "one full decode per episode and camera",
        "first_batch_equivalence_verified": bool(args.verify_first_batch),
        "first_batch_repeated_forward_token_agreement": first_batch_prediction_agreement,
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
