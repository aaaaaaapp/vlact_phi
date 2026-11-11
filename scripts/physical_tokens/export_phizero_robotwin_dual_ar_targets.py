#!/usr/bin/env python3
"""Build exact-key Clean+Randomized RoboTwin AR targets for PhiZero FSQ64.

Run with torchrun on one node. Ranks split the 100 (domain, task) datasets,
encode PhiZero clips, and join action IDs by (domain, task, episode, base).
Only samples with a complete 32-step action target are emitted.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHYSICAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(PHYSICAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PHYSICAL_SCRIPT_DIR))

from diffsynth.models.phizero import (  # noqa: E402
    PhiZeroModelConfig,
    PhysicalLanguageTokenizer,
    load_physical_language_tokenizer,
)
from export_action_token_ids import build_model  # noqa: E402
from export_robotwin_action_cache import transform_action_chunk  # noqa: E402
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_episode_id(record: dict) -> int:
    """Read the episode key from current and legacy clip-manifest schemas."""
    episode_id = record.get("episode_id")
    episode_index = record.get("episode_index")
    if episode_id is None and episode_index is None:
        raise ValueError(f"Clip manifest row lacks episode_id/episode_index: {record}")
    if episode_id is not None and episode_index is not None and int(episode_id) != int(episode_index):
        raise ValueError(f"Conflicting episode_id and episode_index in clip manifest row: {record}")
    return int(episode_id if episode_id is not None else episode_index)


def read_video(path: Path, source_height: int, source_width: int) -> torch.Tensor:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    )
    expected = 33 * source_height * source_width * 3
    if len(result.stdout) != expected:
        raise ValueError(
            f"Expected 33 source frames of {source_width}x{source_height}, "
            f"got {len(result.stdout)} bytes: {path}"
        )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        33, source_height, source_width, 3
    ).copy()
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Optional local root for manifest-relative video paths; defaults to the manifest directory.",
    )
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-action-cache", type=Path, required=True)
    parser.add_argument("--clean-action-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--motion-height", type=int, default=256)
    parser.add_argument("--motion-width", type=int, default=448)
    parser.add_argument("--physics-batch-size", type=int, default=2,
                        help="Number of video clips per PhiZero encode forward pass.")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Smoke-test only; selects the first N manifest rows and writes a separate cache.")
    parser.add_argument("--smoke-task", type=str, default=None,
                        help="Smoke-test one canonical task in both domains instead of scanning all tasks.")
    return parser.parse_args()


def load_clean_action_lookup(ids_path: Path, source_path: Path, expected_tasks: list[str]):
    with np.load(ids_path, allow_pickle=False) as ids, np.load(source_path, allow_pickle=False) as source:
        required = {"action_token_ids", "task_ids", "task_names", "episode_ids", "base_indices"}
        if not required.issubset(ids.files):
            raise ValueError(f"Clean action IDs lack {sorted(required - set(ids.files))}")
        if not {"action_valid_mask", "task_ids", "task_names", "episode_ids", "base_indices"}.issubset(source.files):
            raise ValueError("Clean raw action cache lacks validity masks or exact sample keys")
        task_ids = np.asarray(ids["task_ids"])
        task_names = np.asarray(ids["task_names"])
        episode_ids = np.asarray(ids["episode_ids"])
        base_indices = np.asarray(ids["base_indices"])
        for key, values in (("task_ids", task_ids), ("task_names", task_names),
                            ("episode_ids", episode_ids), ("base_indices", base_indices)):
            if not np.array_equal(values, source[key]):
                raise ValueError(f"Clean action IDs and raw actions are misaligned on {key}")
        names = [str(x) for x in task_names.tolist()]
        if names != expected_tasks:
            raise ValueError("Clean action task order does not match docs/robotwin_50task_list.txt")
        token_ids = np.asarray(ids["action_token_ids"], dtype=np.int32)
        valid = np.asarray(source["action_valid_mask"], dtype=bool).all(axis=1)
        if token_ids.shape != (len(valid), 32):
            raise ValueError(f"Unexpected Clean action target shape {token_ids.shape}")
        lookup = {}
        for row in np.flatnonzero(valid):
            key = (int(task_ids[row]), int(episode_ids[row]), int(base_indices[row]))
            if key in lookup:
                raise ValueError(f"Duplicate Clean action key {key}")
            lookup[key] = token_ids[row]
        return lookup


def validate_records(records: list[dict], tasks: list[str], *, require_full: bool) -> dict[str, int]:
    expected_names = set(tasks)
    domain_task_names = {"clean": expected_names, "randomized": expected_names}
    counts = Counter()
    keys = set()
    for row, record in enumerate(records):
        domain = record.get("source_domain")
        task = record.get("task_name")
        if domain not in domain_task_names or task not in domain_task_names[domain]:
            raise ValueError(f"Unexpected domain/task at manifest row {row}: {domain}/{task}")
        expected_fps = 50 if domain == "clean" else 15
        if (record.get("source_fps") != expected_fps or record.get("target_fps") != 16
                or record.get("num_frames") != 33 or record.get("max_padded_frames") != 8):
            raise ValueError(f"FPS/frame protocol mismatch at manifest row {row}: {record}")
        real, padded = int(record["real_frame_count"]), int(record["padded_frame_count"])
        if real + padded != 33:
            raise ValueError(f"Invalid real/padded frame counts at manifest row {row}")
        if record.get("forced_short_episode"):
            allowed_starts = {0, 20} if domain == "clean" else {0, 6}
            if not record.get("episode_is_short") or int(record["base_index"]) not in allowed_starts:
                raise ValueError(f"Invalid forced-short window at manifest row {row}")
        elif padded > 8:
            raise ValueError(f"Normal window exceeds the 8-frame pad cap at manifest row {row}")
        task_id = tasks.index(task) + (50 if domain == "randomized" else 0)
        key = (task_id, manifest_episode_id(record), int(record["base_index"]))
        if key in keys:
            raise ValueError(f"Duplicate domain-qualified physical key {key}")
        keys.add(key)
        counts[f"{domain}/{task}"] += 1
    if {record["source_domain"] for record in records} != {"clean", "randomized"}:
        raise ValueError("Selected manifest rows must contain both Clean and Randomized domains")
    if require_full and {record["task_name"] for record in records} != expected_names:
        raise ValueError("Full manifest does not cover exactly the canonical 50 tasks")
    return dict(counts)


def encode_actions(model, action_chunks: list[np.ndarray], device: torch.device,
                   batch_size: int) -> list[np.ndarray]:
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(action_chunks), batch_size):
            batch = torch.as_tensor(np.stack(action_chunks[start:start + batch_size]),
                                    dtype=torch.float32, device=device)
            token_ids, _, _ = model.encode(batch)
            encoded.extend(token_ids.detach().cpu().numpy().astype(np.int32))
    return encoded


def encode_physics(model, motion_videos: list[torch.Tensor]) -> np.ndarray:
    if not motion_videos:
        return np.empty((0, 64), dtype=np.int32)
    batch = torch.cat(motion_videos, dim=0)
    with torch.inference_mode():
        token_ids = model(batch).reshape(len(motion_videos), -1)
    if token_ids.shape[1] != 64:
        raise ValueError(f"Expected 64 physical IDs per clip, got {tuple(token_ids.shape)}")
    return token_ids.to(torch.int32).cpu().numpy()


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.tokenizer_checkpoint, args.action_checkpoint,
                 args.clean_action_cache, args.clean_action_source, args.task_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing cache or manifest: {args.output}")
    if args.encode_batch_size < 1 or args.physics_batch_size < 1:
        raise ValueError("Action and physics encode batch sizes must be positive")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # Rank-local task assignments do not synchronize dataset construction;
        # only the final shard barrier is collective and can wait for a long tail.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    if not torch.cuda.is_available():
        raise RuntimeError("PhiZero FSQ64 target export requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks = [line.strip() for line in args.task_list.read_text().splitlines() if line.strip()]
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Canonical RoboTwin task list must contain 50 unique names")
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    video_root = args.video_root or args.manifest.parent
    if not video_root.is_dir():
        raise NotADirectoryError(video_root)
    if args.smoke_task is not None:
        if args.smoke_task not in tasks:
            raise ValueError(f"--smoke-task must be one of the canonical 50 tasks: {args.smoke_task}")
        records = [record for record in records if record["task_name"] == args.smoke_task]
    if args.max_clips is not None:
        if args.max_clips < 1:
            raise ValueError("--max-clips must be positive")
        per_group = Counter()
        selected = []
        for record in records:
            group = (record["source_domain"], record["task_name"])
            if per_group[group] < args.max_clips:
                selected.append(record)
                per_group[group] += 1
        records = selected
    domain_counts = validate_records(records, tasks, require_full=args.max_clips is None)

    model_config = PhiZeroModelConfig.from_checkpoint(args.tokenizer_checkpoint)
    config = model_config.motion_tokenizer
    if config.quantizer != "fsq" or config.codebook_size != 25000 or config.physical_language_length(33) != 64:
        raise ValueError("Expected 33-frame PhiZero FSQ with 64 tokens and K=25000")
    with torch.device("meta"):
        physical_tokenizer = PhysicalLanguageTokenizer(config)
    load_physical_language_tokenizer(physical_tokenizer, args.tokenizer_checkpoint)
    physical_tokenizer = physical_tokenizer.to(device=device, dtype=torch.bfloat16).eval()

    action_checkpoint = torch.load(args.action_checkpoint, map_location="cpu", weights_only=False)
    if int(action_checkpoint["model_config"]["num_tokens"]) != 32 or int(action_checkpoint["model_config"]["codebook_size"]) != 525:
        raise ValueError("Expected action tokenizer with 32 positions and K=525")
    action_tokenizer = build_model(action_checkpoint["model_config"])
    action_tokenizer.load_state_dict(action_checkpoint["model"])
    action_tokenizer = action_tokenizer.to(device).eval()

    clean_lookup = load_clean_action_lookup(args.clean_action_cache, args.clean_action_source, tasks)
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for record in records:
        task_id = tasks.index(record["task_name"]) + (50 if record["source_domain"] == "randomized" else 0)
        grouped[task_id].append((task_id, record))

    local_rows = []
    local_counts = Counter()
    for task_id in sorted(grouped):
        if task_id % world_size != rank:
            continue
        first = grouped[task_id][0][1]
        domain = first["source_domain"]
        task_name = first["task_name"]
        qualified_name = task_name if domain == "clean" else f"Randomized/{task_name}"
        dataset = None
        valid_action_keys = set()
        trajectory_lengths = {}
        if domain == "randomized":
            dataset = make_LeRobotSingleDataset(
                args.data_root,
                qualified_name,
                "robotwin_wrap32",
                data_cfg={
                    "video_backend": "pyav",
                    "lerobot_version": "v2.0",
                    # Each task is owned by one rank; dataset construction is
                    # rank-local and must not enter collectives in task-dependent order.
                    "disable_distributed_metadata_sync": True,
                },
            )
            valid_steps = set((int(ep), int(base)) for ep, base in dataset.all_steps)
            offsets = np.asarray(dataset.delta_indices[dataset.modality_keys["action"][0]], dtype=np.int64)
            trajectory_lengths = {
                int(ep): int(length)
                for ep, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths)
            }
            for _, record in grouped[task_id]:
                key = (manifest_episode_id(record), int(record["base_index"]))
                if key not in valid_steps:
                    raise ValueError(f"Physical clip key {qualified_name}/{key} is absent from LeRobot actions")
                if key[1] + offsets.min() >= 0 and key[1] + offsets.max() < trajectory_lengths[key[0]]:
                    valid_action_keys.add(key)

        action_chunks: list[np.ndarray] = []
        action_chunk_records: list[dict] = []
        random_physics_by_key: dict[tuple[int, int], np.ndarray] = {}
        physics_video_batch: list[torch.Tensor] = []
        physics_record_batch: list[dict] = []
        physics_action_batch: list[np.ndarray | None] = []

        def flush_physics_batch() -> None:
            physics_ids_batch = encode_physics(physical_tokenizer, physics_video_batch)
            for record_batch, action_ids_batch, physics_ids in zip(
                    physics_record_batch, physics_action_batch, physics_ids_batch):
                episode_id = manifest_episode_id(record_batch)
                key = (episode_id, int(record_batch["base_index"]))
                if physics_ids.min() < 0 or physics_ids.max() >= 25000:
                    raise ValueError(f"Invalid physical IDs for {qualified_name}/{key}")
                if domain == "clean":
                    if action_ids_batch is None:
                        raise RuntimeError("Clean physics row is missing its cached action IDs")
                    local_rows.append((record_batch, action_ids_batch.copy(), physics_ids))
                else:
                    random_physics_by_key[key] = physics_ids
                local_counts[f"{domain}_encoded"] += 1
                if local_counts[f"{domain}_encoded"] % 100 == 0:
                    print(
                        f"rank={rank} {qualified_name}: "
                        f"{local_counts[f'{domain}_encoded']} targets", flush=True
                    )
            physics_video_batch.clear()
            physics_record_batch.clear()
            physics_action_batch.clear()

        current_episode = None
        for _, record in grouped[task_id]:
            episode_id, base_index = manifest_episode_id(record), int(record["base_index"])
            key = (episode_id, base_index)
            if domain == "clean":
                action_ids = clean_lookup.get((task_id, episode_id, base_index))
                if action_ids is None:
                    local_counts[f"{domain}_incomplete_action"] += 1
                    continue
            else:
                if key not in valid_action_keys:
                    local_counts[f"{domain}_incomplete_action"] += 1
                    continue
                if current_episode != episode_id:
                    dataset.curr_traj_data = dataset.get_trajectory_data(episode_id)
                    dataset.curr_traj_id = episode_id
                    current_episode = episode_id
                action_chunks.append(transform_action_chunk(dataset, episode_id, base_index))
                action_chunk_records.append(record)
                action_ids = None

            video_path = video_root / record["video"]
            video = read_video(
                video_path, int(record["source_height"]), int(record["source_width"])
            ).to(device=device, dtype=torch.float32)
            video = video.mul_(2.0 / 255.0).sub_(1.0)
            frames = video.permute(0, 2, 1, 3, 4).reshape(
                33, 3, int(record["source_height"]), int(record["source_width"])
            )
            if frames.shape[-2:] != (args.height, args.width):
                frames = F.interpolate(
                    frames, size=(args.height, args.width), mode="bilinear", align_corners=False
                )
            frames = F.interpolate(frames, size=(args.motion_height, args.motion_width),
                                    mode="bilinear", align_corners=False)
            motion_video = frames.reshape(1, 33, 3, args.motion_height, args.motion_width).permute(0, 2, 1, 3, 4)
            physics_video_batch.append(motion_video.to(dtype=torch.bfloat16))
            physics_record_batch.append(record)
            physics_action_batch.append(action_ids)
            if len(physics_video_batch) >= args.physics_batch_size:
                flush_physics_batch()

        if physics_video_batch:
            flush_physics_batch()

        if domain == "randomized" and action_chunk_records:
            random_action_ids = encode_actions(action_tokenizer, action_chunks, device, args.encode_batch_size)
            if len(random_action_ids) != len(action_chunk_records):
                raise RuntimeError("Randomized action encoder returned a mismatched number of rows")
            # Physical IDs were appended in the same record order; attach tokens by exact key.
            for record, action_ids in zip(action_chunk_records, random_action_ids):
                key = (manifest_episode_id(record), int(record["base_index"]))
                if action_ids.shape != (32,) or action_ids.min() < 0 or action_ids.max() >= 525:
                    raise ValueError(f"Invalid action IDs for {qualified_name}/{key}")
                physics_ids = random_physics_by_key.pop(key, None)
                if physics_ids is None:
                    raise RuntimeError(f"Missing physical IDs for {qualified_name}/{key}")
                local_rows.append((record, action_ids, physics_ids))
            if random_physics_by_key:
                raise RuntimeError(f"Unjoined physical rows for {qualified_name}: {len(random_physics_by_key)}")

    work_dir = args.output.parent / (args.output.stem + ".rank_shards")
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_path = work_dir / f"rank-{rank:02d}.npz"
    shard_arrays = rows_to_arrays(local_rows, tasks)
    np.savez_compressed(shard_path, **shard_arrays)
    (work_dir / f"rank-{rank:02d}.json").write_text(json.dumps(dict(local_counts), sort_keys=True) + "\n")
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        all_rows = []
        all_counts = Counter()
        for shard_rank in range(world_size):
            with np.load(work_dir / f"rank-{shard_rank:02d}.npz", allow_pickle=False) as shard:
                n = len(shard["episode_ids"])
                for index in range(n):
                    task_id = int(shard["task_ids"][index])
                    domain = "clean" if task_id < 50 else "randomized"
                    task = tasks[task_id % 50]
                    record = {
                        "source_domain": domain,
                        "task_name": task,
                        "episode_id": int(shard["episode_ids"][index]),
                        "base_index": int(shard["base_indices"][index]),
                        "source_fps": int(shard["source_fps"][index]),
                        "target_fps": 16,
                        "num_frames": 33,
                        "real_frame_count": int(shard["real_frame_counts"][index]),
                        "padded_frame_count": int(shard["padded_frame_counts"][index]),
                        "forced_short_episode": bool(shard["forced_short_episode"][index]),
                        "episode_is_short": bool(shard["episode_is_short"][index]),
                    }
                    all_rows.append((record, shard["action_token_ids"][index].copy(),
                                     shard["physics_token_ids"][index].copy()))
            count_path = work_dir / f"rank-{shard_rank:02d}.json"
            all_counts.update(json.loads(count_path.read_text()))
        arrays = rows_to_arrays(all_rows, tasks)
        order = np.lexsort((arrays["base_indices"], arrays["episode_ids"], arrays["task_ids"]))
        arrays = {key: value if key == "task_names" else value[order] for key, value in arrays.items()}
        keys = list(zip(arrays["task_ids"].tolist(), arrays["episode_ids"].tolist(), arrays["base_indices"].tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("Final dual-AR target cache contains duplicate qualified keys")
        validate_token_arrays(arrays, require_full=args.max_clips is None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, args.output)
        report = {
            "schema_version": 1,
            "kind": "robotwin_clean_randomized_phizero_fsq64_dual_ar_targets",
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_sha256": sha256_file(args.manifest),
            "phi_zero_checkpoint": str(args.tokenizer_checkpoint.resolve()),
            "phi_zero_checkpoint_sha256": sha256_file(args.tokenizer_checkpoint),
            "action_tokenizer_checkpoint": str(args.action_checkpoint.resolve()),
            "action_tokenizer_checkpoint_sha256": sha256_file(args.action_checkpoint),
            "clean_action_cache_sha256": sha256_file(args.clean_action_cache),
            "clean_action_source_sha256": sha256_file(args.clean_action_source),
            "count": len(arrays["episode_ids"]),
            "task_names": [str(x) for x in arrays["task_names"].tolist()],
            "action_shape": list(arrays["action_token_ids"].shape),
            "physics_shape": list(arrays["physics_token_ids"].shape),
            "codebook_size": 25000,
            "frame_protocol": {"clean_source_fps": 50, "randomized_source_fps": 15,
                               "target_fps": 16, "num_frames": 33, "max_padded_frames": 8,
                               "short_starts": {"clean": [0, 20], "randomized": [0, 6]}},
            "domain_counts_in_source_manifest": domain_counts,
            "rank_counts": dict(all_counts),
            "complete_32_action_horizons_only": True,
            "exact_key_join": "(domain-qualified task_id, episode_id, base_index)",
        }
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp, manifest_path)
        print(json.dumps({key: report[key] for key in ("count", "action_shape", "physics_shape",
                                                        "domain_counts_in_source_manifest", "rank_counts")},
                         indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def rows_to_arrays(rows: list[tuple[dict, np.ndarray, np.ndarray]], tasks: list[str]) -> dict[str, np.ndarray]:
    task_names = np.asarray(tasks + [f"Randomized/{task}" for task in tasks])
    if not rows:
        return {
            "action_token_ids": np.empty((0, 32), dtype=np.int32),
            "physics_token_ids": np.empty((0, 64), dtype=np.int32),
            "task_ids": np.empty((0,), dtype=np.int32),
            "task_names": task_names,
            "episode_ids": np.empty((0,), dtype=np.int64),
            "base_indices": np.empty((0,), dtype=np.int64),
            "source_fps": np.empty((0,), dtype=np.int16),
            "real_frame_counts": np.empty((0,), dtype=np.int16),
            "padded_frame_counts": np.empty((0,), dtype=np.int16),
            "forced_short_episode": np.empty((0,), dtype=bool),
            "episode_is_short": np.empty((0,), dtype=bool),
        }
    task_ids, action_ids, physics_ids = [], [], []
    episode_ids, base_indices, source_fps = [], [], []
    real_counts, padded_counts, forced, short = [], [], [], []
    for record, action, physics in rows:
        task_id = tasks.index(record["task_name"]) + (50 if record["source_domain"] == "randomized" else 0)
        task_ids.append(task_id)
        action_ids.append(action)
        physics_ids.append(physics)
        episode_ids.append(manifest_episode_id(record))
        base_indices.append(int(record["base_index"]))
        source_fps.append(int(record["source_fps"]))
        real_counts.append(int(record["real_frame_count"]))
        padded_counts.append(int(record["padded_frame_count"]))
        forced.append(bool(record["forced_short_episode"]))
        short.append(bool(record["episode_is_short"]))
    return {
        "action_token_ids": np.asarray(action_ids, dtype=np.int32),
        "physics_token_ids": np.asarray(physics_ids, dtype=np.int32),
        "task_ids": np.asarray(task_ids, dtype=np.int32),
        "task_names": task_names,
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "base_indices": np.asarray(base_indices, dtype=np.int64),
        "source_fps": np.asarray(source_fps, dtype=np.int16),
        "real_frame_counts": np.asarray(real_counts, dtype=np.int16),
        "padded_frame_counts": np.asarray(padded_counts, dtype=np.int16),
        "forced_short_episode": np.asarray(forced, dtype=bool),
        "episode_is_short": np.asarray(short, dtype=bool),
    }


def validate_token_arrays(arrays: dict[str, np.ndarray], *, require_full: bool) -> None:
    if arrays["action_token_ids"].shape != (len(arrays["episode_ids"]), 32):
        raise ValueError("Final action token shape must be [N,32]")
    if arrays["physics_token_ids"].shape != (len(arrays["episode_ids"]), 64):
        raise ValueError("Final physics token shape must be [N,64]")
    if not len(arrays["episode_ids"]):
        raise ValueError("Final target cache is empty")
    if arrays["action_token_ids"].min() < 0 or arrays["action_token_ids"].max() >= 525:
        raise ValueError("Action IDs are outside [0,524]")
    if arrays["physics_token_ids"].min() < 0 or arrays["physics_token_ids"].max() >= 25000:
        raise ValueError("Physics IDs are outside [0,24999]")
    if len(arrays["task_names"]) != 100:
        raise ValueError("Cache task-name table must contain 100 domain-qualified tasks")
    if require_full and set(arrays["task_ids"].tolist()) != set(range(100)):
        raise ValueError("Full cache must contain all 100 domain-qualified RoboTwin tasks")
    if np.any(arrays["padded_frame_counts"][~arrays["forced_short_episode"]] > 8):
        raise ValueError("A regular clip exceeds the terminal-frame pad cap")


if __name__ == "__main__":
    main()
