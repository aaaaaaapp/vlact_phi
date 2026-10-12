#!/usr/bin/env python3
"""Export frozen Qwen3-VL future visual-feature deltas for RoboTwin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or result[0] != 0 or any(value < 0 for value in result):
        raise argparse.ArgumentTypeError("offsets must be non-negative and begin with 0")
    if tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("offsets must be unique and increasing")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-name", default="click_alarmclock")
    parser.add_argument("--robot-type", default="robotwin_wrap32")
    parser.add_argument("--camera", default="video.cam_high")
    parser.add_argument("--offsets", type=parse_ints, default=(0, 8, 16, 31))
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_future_feature_targets(
    frame_features: np.ndarray,
    base_indices: np.ndarray,
    offsets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather future frame features and return deltas, validity, and clamped indices."""
    frame_features = np.asarray(frame_features)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    if frame_features.ndim != 2 or len(frame_features) == 0:
        raise ValueError("frame_features must be a non-empty [frames, feature_dim] array")
    if base_indices.ndim != 1:
        raise ValueError("base_indices must be one-dimensional")
    raw_indices = base_indices[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
    valid = np.logical_and(raw_indices >= 0, raw_indices < len(frame_features))
    clamped = np.clip(raw_indices, 0, len(frame_features) - 1)
    gathered = frame_features[clamped].astype(np.float32, copy=False)
    deltas = gathered - gathered[:, :1]
    return deltas, valid, clamped


def load_visual_encoder(args: argparse.Namespace):
    if not args.base_vlm.is_dir():
        raise FileNotFoundError(f"base VLM directory does not exist: {args.base_vlm}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.base_vlm.resolve()),
        dtype=dtype,
        attn_implementation="sdpa",
    )
    checkpoint_info = None
    if args.pretrained_checkpoint is not None:
        checkpoint_path = args.pretrained_checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"pretrained checkpoint does not exist: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        prefix = "qwen_vl_interface.model.model.visual."
        visual_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if not visual_state:
            raise ValueError(f"checkpoint contains no parameters under {prefix}")
        model.model.visual.load_state_dict(visual_state, strict=True)
        checkpoint_sha256 = args.pretrained_checkpoint_sha256
        if checkpoint_sha256 is not None:
            checkpoint_sha256 = checkpoint_sha256.lower()
            if len(checkpoint_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_sha256):
                raise ValueError("--pretrained-checkpoint-sha256 must be 64 hexadecimal characters")
        else:
            checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_info = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "sha256_source": "argument" if args.pretrained_checkpoint_sha256 else "computed",
            "visual_parameter_tensors": len(visual_state),
        }
        del state, visual_state
    model.requires_grad_(False).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(str(args.base_vlm.resolve()))
    return model, processor, checkpoint_info


@torch.inference_mode()
def encode_frames(model, processor, frames: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}")
    pooled = []
    for start in range(0, len(frames), batch_size):
        images = [np.asarray(frame, dtype=np.uint8) for frame in frames[start : start + batch_size]]
        inputs = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
        grid_thw = inputs["image_grid_thw"].to(device, non_blocking=True)
        image_embeddings, _ = model.get_image_features(pixel_values, grid_thw)
        pooled.extend(embedding.float().mean(dim=0).cpu() for embedding in image_embeddings)
    return torch.stack(pooled).numpy().astype(np.float32)


def export_cache(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    if args.batch_size < 1 or args.stride < 1:
        raise ValueError("batch size and stride must be positive")
    dataset = make_LeRobotSingleDataset(
        args.data_root.resolve(),
        args.data_name,
        args.robot_type,
        data_cfg={"video_backend": "pyav", "lerobot_version": "v2.0"},
    )
    dataset.transforms.eval()
    if args.camera not in dataset.modality_keys["video"]:
        raise ValueError(f"unknown camera {args.camera}; available={dataset.modality_keys['video']}")

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
        raise ValueError("no samples matched the requested episode/stride selection")

    model, processor, checkpoint_info = load_visual_encoder(args)
    lengths = {
        int(episode_id): int(length)
        for episode_id, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths)
    }
    per_episode_bases: dict[int, list[int]] = {episode_id: [] for episode_id in episode_ids}
    for episode_id, base_index in selected_steps:
        per_episode_bases[episode_id].append(base_index)

    all_deltas = []
    all_valid = []
    all_indices = []
    all_keys = []
    original_delta_indices = dataset.delta_indices[args.camera]
    try:
        for position, episode_id in enumerate(episode_ids, start=1):
            length = lengths[episode_id]
            dataset.curr_traj_data = dataset.get_trajectory_data(episode_id)
            dataset.curr_traj_id = episode_id
            dataset.delta_indices[args.camera] = np.arange(length, dtype=np.int64)
            frames = dataset.get_video(episode_id, args.camera, 0)
            frame_features = encode_frames(model, processor, frames, args.batch_size, args.device)
            bases = np.asarray(per_episode_bases[episode_id], dtype=np.int64)
            deltas, valid, indices = build_future_feature_targets(
                frame_features, bases, args.offsets
            )
            all_deltas.append(deltas.astype(np.float16))
            all_valid.append(valid)
            all_indices.append(indices)
            all_keys.extend((episode_id, int(base_index)) for base_index in bases)
            print(
                f"encoded episode {position}/{len(episode_ids)}: "
                f"episode={episode_id}, frames={length}, samples={len(bases)}",
                flush=True,
            )
    finally:
        dataset.delta_indices[args.camera] = original_delta_indices

    keys = np.asarray(all_keys, dtype=np.int64)
    feature_deltas = np.concatenate(all_deltas, axis=0)
    valid_mask = np.concatenate(all_valid, axis=0).astype(np.bool_)
    source_indices = np.concatenate(all_indices, axis=0).astype(np.int64)
    if not np.all(feature_deltas[:, 0] == 0):
        raise RuntimeError("the current-frame feature delta must be exactly zero")
    arrays = {
        "feature_deltas": feature_deltas,
        "future_valid_mask": valid_mask,
        "source_frame_indices": source_indices,
        "episode_ids": keys[:, 0],
        "base_indices": keys[:, 1],
        "offsets": np.asarray(args.offsets, dtype=np.int64),
    }

    dataset_path = (args.data_root / args.data_name).resolve()
    manifest = {
        "schema_version": 1,
        "kind": "robotwin_future_visual_feature_deltas",
        "dataset_path": str(dataset_path),
        "data_name": args.data_name,
        "robot_type": args.robot_type,
        "camera": args.camera,
        "offsets": list(args.offsets),
        "fps": 50,
        "horizon_seconds": args.offsets[-1] / 50.0,
        "num_episodes": len(episode_ids),
        "num_samples": len(keys),
        "feature_shape": list(feature_deltas.shape),
        "feature_dtype": str(feature_deltas.dtype),
        "feature_definition": "mean-pooled Qwen3-VL vision embeddings; future minus current",
        "boundary_policy": "clamp indices to episode end and preserve future_valid_mask",
        "base_vlm": str(args.base_vlm.resolve()),
        "base_vlm_config_sha256": sha256_file(args.base_vlm.resolve() / "config.json"),
        "pretrained_checkpoint": checkpoint_info,
        "augmentation_mode": "eval/no augmentation",
        "stride": args.stride,
        "source_fingerprint": {
            "info.json": sha256_file(dataset_path / "meta" / "info.json"),
            "modality.json": sha256_file(dataset_path / "meta" / "modality.json"),
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
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(f"wrote {manifest['num_samples']} samples to {args.output}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
