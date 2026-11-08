#!/usr/bin/env python3
"""Export LIBERO LeRobot episodes as PhiZero 33-frame, 8-FPS RGB clips.

The source is assumed to be 20 FPS. Splits are deterministic at episode level;
the output manifest preserves original frame indices for cache alignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


SUITES = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)
FRAME_OFFSETS = tuple((5 * j + 1) // 2 for j in range(33))  # floor(20*j/8 + 0.5)


def window_starts(length: int, stride: int) -> list[int]:
    if length < 1 or stride < 1:
        raise ValueError("episode length and stride must be positive")
    if stride == 1:
        # Dense AR supervision needs every decision frame, including all tails.
        return list(range(length))
    last_complete = length - 81
    if last_complete < 0:
        return [0]
    starts = list(range(0, last_complete + 1, stride))
    if starts[-1] != last_complete:
        starts.append(last_complete)
    first_partial = (last_complete // stride + 1) * stride
    if first_partial < length:
        starts.append(first_partial)
    return sorted(set(starts))


def frame_indices(length: int, start: int) -> list[int]:
    if not 0 <= start < length:
        raise ValueError("window start must lie within its episode")
    return [min(length - 1, start + offset) for offset in FRAME_OFFSETS]


def split_name(suite: str, episode: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{suite}:{episode}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if fraction < validation_fraction else "train"


def letterbox(frame, height: int, width: int):
    import numpy as np
    from PIL import Image

    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = Image.fromarray(frame).resize(
        (round(source_width * scale), round(source_height * scale)),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return np.asarray(canvas)


def read_episode(source: Path, length: int) -> list:
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    frame_bytes = 256 * 256 * 3
    if len(result.stdout) % frame_bytes:
        raise ValueError(f"Unexpected source frame dimensions: {source}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 256, 256, 3)
    if len(frames) != length:
        raise ValueError(f"Video frame count differs from metadata: {source}")
    return frames


def export_clip(frames: list, destination: Path, indices: list[int],
                height: int, width: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "8", "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(temporary)],
        stdin=subprocess.PIPE,
    )
    try:
        for index in indices:
            writer.stdin.write(letterbox(frames[index], height, width).tobytes())
    finally:
        writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError(f"Failed to encode clip: {destination}")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8,
                        help="Source-frame start stride: 8 for tokenizer training, 1 for dense cache")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stride < 1 or not 0 <= args.validation_fraction < 1:
        parser.error("stride must be positive and validation fraction must be in [0,1)")
    if args.height < 1 or args.width < 1:
        parser.error("image dimensions must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("max episodes must be positive")
    if not args.dry_run and (args.output_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"Refusing to replace an existing export: {args.output_dir}")

    rows = {"train": [], "val": []}
    manifest = []
    episode_count = 0
    for suite in SUITES:
        root = args.data_root / suite
        info = json.loads((root / "meta/info.json").read_text())
        if int(info["fps"]) != 20:
            raise ValueError(f"Expected 20 FPS LIBERO source: {root}")
        with (root / "meta/episodes.jsonl").open() as file:
            episodes = [json.loads(line) for line in file]
        for episode_info in episodes:
            episode = int(episode_info["episode_index"])
            length = int(episode_info["length"])
            source = (root / "videos/chunk-000/observation.images.image"
                      / f"episode_{episode:06d}.mp4")
            if not source.is_file():
                raise FileNotFoundError(source)
            partition = split_name(suite, episode, args.validation_fraction)
            tasks = episode_info.get("tasks", [])
            if len(tasks) != 1:
                raise ValueError(f"Expected one LIBERO task per episode: {suite}/{episode}")
            task_name = f"{suite}/{tasks[0]}"
            starts = window_starts(length, args.stride)
            frames = None
            for start in starts:
                indices = frame_indices(length, start)
                relative = Path("clips") / suite / f"episode_{episode:06d}_start_{start:06d}.mp4"
                destination = args.output_dir / relative
                if not args.dry_run and (not destination.exists() or destination.stat().st_size == 0):
                    if frames is None:
                        frames = read_episode(source, length)
                    export_clip(frames, destination, indices, args.height, args.width)
                rows[partition].append({"video": str(relative)})
                manifest.append({
                    "suite": suite, "episode_index": episode, "base_index": start,
                    "task_name": task_name,
                    "split": partition, "video": str(relative), "source_video": str(source),
                    "source_frame_indices": indices,
                    "real_frame_count": sum(start + offset < length for offset in FRAME_OFFSETS),
                    "padded_frame_count": sum(start + offset >= length for offset in FRAME_OFFSETS),
                    "camera": "observation.images.image", "source_fps": 20, "output_fps": 8,
                    "preprocess": f"RGB letterbox {args.width}x{args.height}",
                })
            episode_count += 1
            if args.max_episodes is not None and episode_count >= args.max_episodes:
                break
        if args.max_episodes is not None and episode_count >= args.max_episodes:
            break

    if args.dry_run:
        print(json.dumps({"episodes": episode_count, "train": len(rows["train"]),
                          "val": len(rows["val"])}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for partition in ("train", "val"):
        with (args.output_dir / f"{partition}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["video"])
            writer.writeheader()
            writer.writerows(rows[partition])
    with (args.output_dir / "manifest.jsonl").open("w") as file:
        for record in manifest:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": episode_count, "train": len(rows["train"]),
                      "val": len(rows["val"])}))


if __name__ == "__main__":
    main()
