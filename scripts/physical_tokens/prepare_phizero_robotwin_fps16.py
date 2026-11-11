#!/usr/bin/env python3
"""Export Clean + Randomized RoboTwin windows for 16 FPS PhiZero FSQ64 training.

Both domains are sampled on a common 16 FPS, 33-frame, 2-second target grid.
Clean sources are 50 FPS; Randomized sources are 15 FPS. Normal windows may
repeat at most 8 terminal frames. Episodes shorter than 2 seconds are retained
with two starts separated by approximately 0.4 seconds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


SOURCE_FPS_BY_DOMAIN = {"clean": 50, "randomized": 15}
TARGET_FPS = 16
NUM_FRAMES = 33
MAX_PADDED_FRAMES = 8
WINDOW_STRIDE_SECONDS = 0.4


def temporal_grid(source_fps: int) -> tuple[np.ndarray, int, tuple[int, int]]:
    offsets = np.floor(
        source_fps * np.arange(NUM_FRAMES) / TARGET_FPS + 0.5
    ).astype(np.int64)
    stride = max(1, int(np.floor(source_fps * WINDOW_STRIDE_SECONDS + 0.5)))
    return offsets, stride, (0, stride)


def run(command: list[str], *, capture: bool = True) -> bytes:
    try:
        result = subprocess.run(command, check=True, capture_output=capture)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace") if error.stderr else str(error)
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}") from error
    return result.stdout if capture else b""


def probe_video(path: Path) -> tuple[int, int, int, float]:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ])
    stream = json.loads(result.decode())["streams"][0]
    rate_num, rate_den = (int(value) for value in stream["avg_frame_rate"].split("/"))
    fps = rate_num / rate_den
    frame_count = int(stream.get("nb_frames") or 0)
    return int(stream["width"]), int(stream["height"]), frame_count, fps


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def choose_starts(length: int, source_fps: int) -> list[tuple[int, bool]]:
    offsets, stride, short_starts = temporal_grid(source_fps)
    short_episode_frames = int(offsets[-1]) + 1
    # The last eight target positions may repeat the terminal source frame.
    # Use the ninth-from-last target index as the last guaranteed-real one.
    max_start_with_padding = (
        length - 1 - int(offsets[NUM_FRAMES - MAX_PADDED_FRAMES - 1])
    )
    starts = (
        list(range(0, max_start_with_padding + 1, stride))
        if max_start_with_padding >= 0 else []
    )
    if max_start_with_padding >= 0 and starts[-1] != max_start_with_padding:
        starts.append(max_start_with_padding)

    forced_starts: set[int] = set()
    if length < short_episode_frames:
        if length <= short_starts[-1]:
            raise ValueError(
                f"Episode of {length} frames cannot provide the required starts "
                f"{short_starts} at {source_fps} FPS"
            )
        forced_starts.update(short_starts)
        starts = sorted(set(starts) | forced_starts)
    return [(start, start in forced_starts) for start in starts]


def read_episode(
    path: Path, expected_frames: int, expected_fps: int
) -> tuple[np.ndarray, int, int]:
    width, height, probed_frames, fps = probe_video(path)
    if abs(fps - expected_fps) > 0.01:
        raise ValueError(f"Expected {expected_fps} FPS, got {fps}: {path}")
    if probed_frames and probed_frames != expected_frames:
        raise ValueError(
            f"Metadata says {expected_frames} frames, ffprobe says {probed_frames}: {path}"
        )
    decoded = run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    frame_bytes = width * height * 3
    if len(decoded) % frame_bytes:
        raise ValueError(f"Decoded byte count is not frame aligned: {path}")
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) != expected_frames:
        raise ValueError(
            f"Metadata says {expected_frames} frames, decoded {len(frames)}: {path}"
        )
    return frames, width, height


def encode_clip(frames: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite clip: {destination}")
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    height, width = frames.shape[1:3]
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(TARGET_FPS), "-i", "pipe:0",
        "-frames:v", str(NUM_FRAMES), "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "18", str(temporary),
    ], input=frames.tobytes(), check=True)
    _, _, encoded_frames, encoded_fps = probe_video(temporary)
    if encoded_frames != NUM_FRAMES or abs(encoded_fps - TARGET_FPS) > 0.01:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected {NUM_FRAMES} frames at {TARGET_FPS} FPS, got "
            f"{encoded_frames} at {encoded_fps}: {temporary}"
        )
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", "--data-root", dest="clean_root", type=Path, required=True)
    parser.add_argument("--randomized-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-domain", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.max_episodes_per_domain is not None and args.max_episodes_per_domain < 1:
        parser.error("--max-episodes-per-domain must be positive")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split.read_text())
    training_ids = set(map(int, split["training_episode_ids"]))
    validation_ids = set(map(int, split["validation_episode_ids"]))
    if training_ids & validation_ids:
        raise ValueError("Training and validation episode IDs overlap")
    tasks = [line.strip() for line in args.tasks.read_text().splitlines() if line.strip()]
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError(f"Expected 50 unique RoboTwin tasks, got {len(tasks)}")

    jobs = []
    source_roots = (
        ("clean", args.clean_root),
        ("randomized", args.randomized_root),
    )
    for source_domain, source_root in source_roots:
        expected_fps = SOURCE_FPS_BY_DOMAIN[source_domain]
        domain_job_count = 0
        for task_id, task in enumerate(tasks):
            task_root = source_root / task
            info = json.loads((task_root / "meta/info.json").read_text())
            if int(info["fps"]) != expected_fps:
                raise ValueError(
                    f"Expected {expected_fps} FPS metadata for {source_domain}: {task_root}"
                )
            episode_infos = [
                json.loads(line)
                for line in (task_root / "meta/episodes.jsonl").read_text().splitlines()
                if line.strip()
            ]
            video_dir = task_root / "videos/chunk-000/observation.images.cam_high"
            for episode_info in episode_infos:
                episode = int(episode_info["episode_index"])
                length = int(episode_info["length"])
                if episode not in training_ids and episode not in validation_ids:
                    continue
                source = video_dir / f"episode_{episode:06d}.mp4"
                if not source.is_file():
                    raise FileNotFoundError(source)
                partition = "train" if episode in training_ids else "val"
                starts = choose_starts(length, expected_fps)
                jobs.append((
                    partition, source_domain, task_id, task, episode, length,
                    expected_fps, starts, source,
                ))
                domain_job_count += 1
                if args.max_episodes is not None and len(jobs) >= args.max_episodes:
                    break
                if (args.max_episodes_per_domain is not None
                        and domain_job_count >= args.max_episodes_per_domain):
                    break
            if args.max_episodes is not None and len(jobs) >= args.max_episodes:
                break
            if (args.max_episodes_per_domain is not None
                    and domain_job_count >= args.max_episodes_per_domain):
                break
        if args.max_episodes is not None and len(jobs) >= args.max_episodes:
            break

    def prepare(job):
        (partition, source_domain, task_id, task, episode, length,
         source_fps, starts, source) = job
        offsets, _, short_starts = temporal_grid(source_fps)
        frames, width, height = read_episode(source, length, source_fps)
        src_hash = source_sha256(source)
        records = []
        for start, forced_short_window in starts:
            raw_indices = start + offsets
            padded_count = int(np.count_nonzero(raw_indices >= length))
            if not forced_short_window and padded_count > MAX_PADDED_FRAMES:
                raise AssertionError(f"Regular window exceeds pad cap: {task}/{episode}/{start}")
            indices = np.minimum(raw_indices, length - 1).astype(np.int64)
            relative = (
                Path("clips") / source_domain / task
                / f"episode_{episode:06d}_start_{start:06d}.mp4"
            )
            destination = output / relative
            encode_clip(frames[indices], destination)
            records.append({
                "split": partition,
                "source_domain": source_domain,
                "task_id": task_id,
                "task_name": task,
                "episode_id": episode,
                "base_index": start,
                "video": relative.as_posix(),
                "source_video": str(source),
                "source_video_sha256": src_hash,
                "source_fps": source_fps,
                "target_fps": TARGET_FPS,
                "num_frames": NUM_FRAMES,
                "source_frame_indices": indices.tolist(),
                "real_frame_count": NUM_FRAMES - padded_count,
                "padded_frame_count": padded_count,
                "max_padded_frames": MAX_PADDED_FRAMES,
                "forced_short_episode": forced_short_window,
                "episode_is_short": length < int(offsets[-1]) + 1,
                "short_episode_starts": list(short_starts) if length < int(offsets[-1]) + 1 else None,
                "first_frame_robot_arm_label": None,
                "episode_source_frames": length,
                "camera": "observation.images.cam_high",
                "source_width": width,
                "source_height": height,
                "source_fps_metadata": source_fps,
            })
        return records

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare, job): job for job in jobs}
        for finished, future in enumerate(as_completed(futures), start=1):
            records.extend(future.result())
            if finished % 50 == 0 or finished == len(futures):
                print(f"prepared episodes {finished}/{len(futures)}; windows={len(records)}", flush=True)

    records.sort(key=lambda row: (
        row["split"], row["source_domain"], row["task_id"],
        row["episode_id"], row["base_index"],
    ))
    for partition in ("train", "val"):
        with (output / f"{partition}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["video"])
            writer.writeheader()
            writer.writerows({"video": row["video"]} for row in records if row["split"] == partition)
    with (output / "manifest.jsonl").open("w") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    train = [row for row in records if row["split"] == "train"]
    val = [row for row in records if row["split"] == "val"]
    by_domain = {}
    for domain in SOURCE_FPS_BY_DOMAIN:
        domain_train = [row for row in train if row["source_domain"] == domain]
        domain_val = [row for row in val if row["source_domain"] == domain]
        by_domain[domain] = {
            "source_fps": SOURCE_FPS_BY_DOMAIN[domain],
            "train_episodes": len({(r["task_id"], r["episode_id"]) for r in domain_train}),
            "val_episodes": len({(r["task_id"], r["episode_id"]) for r in domain_val}),
            "train_windows": len(domain_train),
            "val_windows": len(domain_val),
            "train_forced_short_windows": sum(r["forced_short_episode"] for r in domain_train),
            "val_forced_short_windows": sum(r["forced_short_episode"] for r in domain_val),
            "train_padding_max": max((r["padded_frame_count"] for r in domain_train), default=0),
            "val_padding_max": max((r["padded_frame_count"] for r in domain_val), default=0),
        }
    short_rules = {}
    for domain, source_fps in SOURCE_FPS_BY_DOMAIN.items():
        offsets, _, starts = temporal_grid(source_fps)
        short_rules[domain] = {
            "source_fps": source_fps,
            "if_source_frames_lt": int(offsets[-1]) + 1,
            "starts": list(starts),
            "max_observed_padding_exception": max(
                (r["padded_frame_count"] for r in records
                 if r["source_domain"] == domain and r["forced_short_episode"]),
                default=0,
            ),
        }
    report = {
        "source_roots": {
            "clean": str(args.clean_root.resolve()),
            "randomized": str(args.randomized_root.resolve()),
        },
        "tasks": len(tasks),
        "source_fps_by_domain": SOURCE_FPS_BY_DOMAIN,
        "target_fps": TARGET_FPS,
        "num_frames": NUM_FRAMES,
        "max_padded_frames": MAX_PADDED_FRAMES,
        "window_stride_seconds": WINDOW_STRIDE_SECONDS,
        "short_episode_rule_by_domain": short_rules,
        "domains": by_domain,
        "train_episodes": len({(r["source_domain"], r["task_id"], r["episode_id"]) for r in train}),
        "val_episodes": len({(r["source_domain"], r["task_id"], r["episode_id"]) for r in val}),
        "train_windows": len(train),
        "val_windows": len(val),
        "train_forced_short_windows": sum(r["forced_short_episode"] for r in train),
        "val_forced_short_windows": sum(r["forced_short_episode"] for r in val),
        "train_padding_max": max((r["padded_frame_count"] for r in train), default=0),
        "val_padding_max": max((r["padded_frame_count"] for r in val), default=0),
        "max_episodes_smoke_limit": args.max_episodes,
        "max_episodes_per_domain_smoke_limit": args.max_episodes_per_domain,
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
