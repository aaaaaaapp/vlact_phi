#!/usr/bin/env python3
"""Export deduplicated spatial Qwen3-VL features for PhiZero-style motion tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_robotwin_future_features import load_visual_encoder, parse_ints, sha256_file
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_episode_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episode IDs must be integers") from exc
    if not result or any(episode_id < 0 for episode_id in result):
        raise argparse.ArgumentTypeError("episode IDs must be non-negative")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("episode IDs must be unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-name", default=None)
    parser.add_argument(
        "--data-names",
        default=None,
        help="Comma-separated task names. Mutually exclusive with --data-name.",
    )
    parser.add_argument("--robot-type", default="robotwin_wrap32")
    parser.add_argument("--camera", default="video.cam_high")
    parser.add_argument("--offsets", type=parse_ints, default=tuple(range(0, 33, 2)))
    parser.add_argument("--spatial-grid-size", type=int, default=4)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--include-episode-ids",
        type=parse_episode_ids,
        default=(),
        help="Exact episode IDs to include (mutually exclusive with --exclude-episode-ids).",
    )
    parser.add_argument(
        "--exclude-episode-ids",
        type=parse_episode_ids,
        default=(),
        help="Episode IDs to exclude before applying --max-episodes (applied to every task).",
    )
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def resolve_data_names(data_name: str | None, data_names: str | None) -> tuple[str, ...]:
    if data_name and data_names:
        raise ValueError("--data-name and --data-names are mutually exclusive")
    values = data_names.split(",") if data_names else [data_name or "click_alarmclock"]
    result = tuple(value.strip() for value in values if value.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("data names must be non-empty and unique")
    return result


def select_episode_ids(
    trajectory_ids,
    excluded_episode_ids: set[int],
    max_episodes: int,
    included_episode_ids: set[int] | None = None,
) -> list[int]:
    included_episode_ids = included_episode_ids or set()
    if included_episode_ids and excluded_episode_ids:
        raise ValueError("included and excluded episode IDs are mutually exclusive")
    available = {int(value) for value in trajectory_ids}
    unknown = included_episode_ids - available
    if unknown:
        raise ValueError(f"included episode IDs are unavailable: {sorted(unknown)}")
    selected = [
        int(value)
        for value in trajectory_ids
        if (
            int(value) in included_episode_ids
            if included_episode_ids
            else int(value) not in excluded_episode_ids
        )
    ]
    return selected[:max_episodes] if max_episodes > 0 else selected


def pool_image_embedding(
    embedding: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    output_grid_size: int,
) -> torch.Tensor:
    """Restore the merged patch grid and pool it to a fixed spatial grid."""
    temporal, height, width = (int(value) for value in grid_thw.tolist())
    if temporal != 1:
        raise ValueError(f"expected image temporal grid 1, got {temporal}")
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("vision grid is not divisible by spatial_merge_size")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    if embedding.shape[0] != merged_height * merged_width:
        raise ValueError(
            f"embedding has {embedding.shape[0]} tokens, expected "
            f"{merged_height * merged_width} from grid {tuple(grid_thw.tolist())}"
        )
    grid = embedding.reshape(merged_height, merged_width, embedding.shape[-1])
    grid = grid.permute(2, 0, 1).float().unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(grid, (output_grid_size, output_grid_size))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(output_grid_size**2, -1)


def build_source_frame_indices(
    base_indices: np.ndarray,
    offsets: tuple[int, ...],
    *,
    episode_length: int,
    global_frame_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(base_indices, dtype=np.int64)[:, None] + np.asarray(
        offsets, dtype=np.int64
    )[None, :]
    valid = np.logical_and(raw >= 0, raw < episode_length)
    local = np.clip(raw, 0, episode_length - 1)
    return local + int(global_frame_start), valid


@torch.inference_mode()
def encode_spatial_frames(
    model,
    processor,
    frames: np.ndarray,
    *,
    batch_size: int,
    device: str,
    output_grid_size: int,
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}")
    merge_size = int(model.config.vision_config.spatial_merge_size)
    pooled = []
    for start in range(0, len(frames), batch_size):
        images = [np.asarray(frame, dtype=np.uint8) for frame in frames[start : start + batch_size]]
        inputs = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
        grid_thw = inputs["image_grid_thw"].to(device, non_blocking=True)
        image_embeddings, _ = model.get_image_features(pixel_values, grid_thw)
        pooled.extend(
            pool_image_embedding(
                embedding,
                grid,
                spatial_merge_size=merge_size,
                output_grid_size=output_grid_size,
            ).cpu()
            for embedding, grid in zip(image_embeddings, grid_thw)
        )
    return torch.stack(pooled).numpy().astype(np.float16)


def export_cache(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    if args.batch_size < 1 or args.stride < 1 or args.spatial_grid_size < 1:
        raise ValueError("batch size, stride, and spatial grid size must be positive")
    data_names = resolve_data_names(args.data_name, args.data_names)
    model, processor, checkpoint_info = load_visual_encoder(args)
    frame_features = []
    source_indices = []
    valid_masks = []
    keys = []
    frame_start = 0
    dataset_entries = []
    fps_values = set()
    total_episodes = 0
    excluded_episode_ids = {int(value) for value in args.exclude_episode_ids}
    included_episode_ids = {
        int(value) for value in getattr(args, "include_episode_ids", ())
    }
    for task_id, data_name in enumerate(data_names):
        dataset_path = (args.data_root / data_name).resolve()
        info_path = dataset_path / "meta" / "info.json"
        modality_path = dataset_path / "meta" / "modality.json"
        info = json.loads(info_path.read_text())
        fps = int(info["fps"])
        fps_values.add(fps)
        dataset = make_LeRobotSingleDataset(
            args.data_root.resolve(),
            data_name,
            args.robot_type,
            data_cfg={"video_backend": "pyav", "lerobot_version": "v2.0"},
        )
        dataset.transforms.eval()
        if args.camera not in dataset.modality_keys["video"]:
            raise ValueError(
                f"unknown camera {args.camera} for {data_name}; "
                f"available={dataset.modality_keys['video']}"
            )
        episode_ids = select_episode_ids(
            dataset.trajectory_ids,
            excluded_episode_ids,
            args.max_episodes,
            included_episode_ids,
        )
        allowed = set(episode_ids)
        selected_steps = [
            (int(episode_id), int(base_index))
            for episode_id, base_index in dataset.all_steps
            if int(episode_id) in allowed and int(base_index) % args.stride == 0
        ]
        if not selected_steps:
            raise ValueError(f"no samples matched task {data_name}")
        lengths = {
            int(episode_id): int(length)
            for episode_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths)
        }
        per_episode_bases: dict[int, list[int]] = {
            episode_id: [] for episode_id in episode_ids
        }
        for episode_id, base_index in selected_steps:
            per_episode_bases[episode_id].append(base_index)

        original_delta_indices = dataset.delta_indices[args.camera]
        try:
            for position, episode_id in enumerate(episode_ids, start=1):
                length = lengths[episode_id]
                dataset.curr_traj_data = dataset.get_trajectory_data(episode_id)
                dataset.curr_traj_id = episode_id
                dataset.delta_indices[args.camera] = np.arange(length, dtype=np.int64)
                frames = dataset.get_video(episode_id, args.camera, 0)
                features = encode_spatial_frames(
                    model,
                    processor,
                    frames,
                    batch_size=args.batch_size,
                    device=args.device,
                    output_grid_size=args.spatial_grid_size,
                )
                bases = np.asarray(per_episode_bases[episode_id], dtype=np.int64)
                indices, valid = build_source_frame_indices(
                    bases,
                    args.offsets,
                    episode_length=length,
                    global_frame_start=frame_start,
                )
                frame_features.append(features)
                source_indices.append(indices)
                valid_masks.append(valid)
                keys.extend(
                    (task_id, episode_id, int(base_index)) for base_index in bases
                )
                frame_start += length
                print(
                    f"encoded task {task_id + 1}/{len(data_names)} {data_name}, "
                    f"episode {position}/{len(episode_ids)}: episode={episode_id}, "
                    f"frames={length}, spatial_tokens={features.shape[1]}",
                    flush=True,
                )
        finally:
            dataset.delta_indices[args.camera] = original_delta_indices
        total_episodes += len(episode_ids)
        dataset_entries.append(
            {
                "task_id": task_id,
                "data_name": data_name,
                "dataset_path": str(dataset_path),
                "fps": fps,
                "num_episodes": len(episode_ids),
                "episode_ids": episode_ids,
                "num_samples": len(selected_steps),
                "source_fingerprint": {
                    "info.json": sha256_file(info_path),
                    "modality.json": sha256_file(modality_path),
                },
            }
        )
    if len(fps_values) != 1:
        raise ValueError(f"all tasks must have the same FPS, got {sorted(fps_values)}")
    fps = next(iter(fps_values))

    keys_array = np.asarray(keys, dtype=np.int64)
    features_array = np.concatenate(frame_features, axis=0)
    indices_array = np.concatenate(source_indices, axis=0).astype(np.int64)
    valid_array = np.concatenate(valid_masks, axis=0).astype(np.bool_)
    if indices_array.min() < 0 or indices_array.max() >= len(features_array):
        raise RuntimeError("source frame indices are outside the deduplicated feature array")
    arrays = {
        "frame_features": features_array,
        "source_frame_indices": indices_array,
        "future_valid_mask": valid_array,
        "task_ids": keys_array[:, 0],
        "task_names": np.asarray(data_names),
        "episode_ids": keys_array[:, 1],
        "base_indices": keys_array[:, 2],
        "offsets": np.asarray(args.offsets, dtype=np.int64),
    }
    manifest = {
        "schema_version": 2,
        "kind": "robotwin_deduplicated_spatiotemporal_visual_features",
        "data_root": str(args.data_root.resolve()),
        "data_names": list(data_names),
        "datasets": dataset_entries,
        "robot_type": args.robot_type,
        "camera": args.camera,
        "offsets": list(args.offsets),
        "fps": fps,
        "horizon_seconds": args.offsets[-1] / fps,
        "num_episodes": total_episodes,
        "excluded_episode_ids": sorted(excluded_episode_ids),
        "included_episode_ids": sorted(included_episode_ids),
        "num_samples": len(keys_array),
        "num_unique_frames": len(features_array),
        "frame_feature_shape": list(features_array.shape),
        "frame_feature_dtype": str(features_array.dtype),
        "spatial_grid_size": args.spatial_grid_size,
        "feature_definition": "fixed-grid Qwen3-VL visual tokens before global pooling",
        "boundary_policy": "clamp indices to episode end and preserve future_valid_mask",
        "base_vlm": str(args.base_vlm.resolve()),
        "base_vlm_config_sha256": sha256_file(args.base_vlm.resolve() / "config.json"),
        "pretrained_checkpoint": checkpoint_info,
        "augmentation_mode": "eval/no augmentation",
        "stride": args.stride,
    }
    if len(data_names) == 1:
        manifest["data_name"] = data_names[0]
        manifest["dataset_path"] = dataset_entries[0]["dataset_path"]
        manifest["source_fingerprint"] = dataset_entries[0]["source_fingerprint"]
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
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
