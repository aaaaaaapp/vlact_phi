#!/usr/bin/env python3
"""Create PhiZero-compatible 33-frame/8-FPS RoboTwin clips and split CSVs.

Randomized episodes produce 4.125-second windows at the requested stride.
Clean RoboTwin episodes are shorter, so each complete episode is temporally
resampled to 33 frames; the final frame is repeated only if needed to satisfy
PhiZero's fixed temporal shape. No frames are crossed between episodes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


def probe(path: Path) -> tuple[float, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration,nb_frames:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    duration = float(stream.get("duration") or metadata["format"]["duration"])
    frames = int(stream.get("nb_frames") or 0)
    return duration, frames


def window_starts(frame_count: int, window_frames: int, stride_frames: int) -> list[int]:
    if frame_count < window_frames:
        return []
    last_start = frame_count - window_frames
    starts = list(range(0, last_start + 1, stride_frames))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def decoded_hash(path: Path) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--clean-root",
        type=Path,
        help="Optional Clean RoboTwin root; task folders live directly under this path.",
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="For a small smoke test; counts source episodes across all tasks.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent ffmpeg workers for video preparation.")
    parser.add_argument("--stride-seconds", type=float, default=2.0,
                        help="Sliding-window stride; the final window is aligned to the episode end.")
    args = parser.parse_args()
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stride_seconds <= 0:
        parser.error("--stride-seconds must be positive")

    split = json.loads(args.split.read_text())
    train_ids = set(split["training_episode_ids"])
    val_ids = set(split["validation_episode_ids"])
    if train_ids & val_ids:
        raise ValueError("Training and validation episode IDs overlap")
    tasks = [line.strip() for line in args.tasks.read_text().splitlines() if line.strip()]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    skipped_short = []
    jobs = []
    seen = 0
    sources = []
    if args.clean_root is not None:
        sources.append(("clean", args.clean_root))
    sources.append(("randomized", args.data_root))
    for source_name, source_root in sources:
        for task in tasks:
            source_dir = source_root / task / "videos" / "chunk-000" / "observation.images.cam_high"
            if not source_dir.is_dir():
                raise FileNotFoundError(source_dir)
            for source in sorted(source_dir.glob("episode_*.mp4")):
                episode = int(source.stem.removeprefix("episode_"))
                partition = "train" if episode in train_ids else "val" if episode in val_ids else None
                if partition is None:
                    continue
                if args.max_videos is not None and seen >= args.max_videos:
                    break
                seen += 1
                jobs.append((partition, source_name, task, episode, source))
            if args.max_videos is not None and seen >= args.max_videos:
                break
        if args.max_videos is not None and seen >= args.max_videos:
            break

    def prepare_episode(job):
        partition, source_name, task, episode, source = job
        duration, source_frames = probe(source)
        if source_name == "clean":
            # 33 frames span 32 intervals. Stretch timestamps to exactly four
            # seconds before encoding at 8 FPS, retaining real source frames.
            source_fps = source_frames / duration if duration > 0 and source_frames > 0 else 0
            source_span = duration - (1 / source_fps) if source_fps > 0 else duration
            if source_span <= 0:
                raise ValueError(f"Invalid Clean episode duration/frame count: {source}")
            starts = [0]
        else:
            sampled_frames = math.floor(duration * 8 + 1e-6)
            starts = window_starts(sampled_frames, 33, round(args.stride_seconds * 8))
        episode_dir = output / "clips" / source_name / task
        episode_dir.mkdir(parents=True, exist_ok=True)
        rows_for_episode = []
        seen_hashes = set()
        for start in starts:
            destination = episode_dir / f"episode_{episode:06d}_start_{start:05d}.mp4"
            if not destination.exists():
                if source_name == "clean":
                    video_filter = (
                        f"setpts={4.0 / source_span:.12f}*PTS,"
                        "tpad=stop_mode=clone:stop_duration=0.125,fps=8,"
                        "trim=end_frame=33,setpts=N/(8*TB)"
                    )
                else:
                    video_filter = (
                        f"fps=8,trim=start_frame={start}:end_frame={start + 33},"
                        "setpts=PTS-STARTPTS"
                    )
                subprocess.run(
                    ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                     "-vf", video_filter,
                     "-frames:v", "33", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                     str(destination)],
                    check=True,
                )
            _, frames = probe(destination)
            if frames != 33:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"Expected 33 frames, got {frames}: {destination}")
            content_hash = decoded_hash(destination)
            if content_hash in seen_hashes:
                destination.unlink(missing_ok=True)
                continue
            seen_hashes.add(content_hash)
            rows_for_episode.append((partition, source_name, task, episode, start, start + 32,
                                     str(source), str(destination)))
        return rows_for_episode

    manifest_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_jobs = {executor.submit(prepare_episode, job): job for job in jobs}
        for index, future in enumerate(as_completed(future_jobs), start=1):
            episode_rows = future.result()
            manifest_rows.extend(episode_rows)
            if not episode_rows:
                skipped_short.append(str(future_jobs[future][4]))
            for row in episode_rows:
                rows[row[0]].append([row[7]])
            if index % 100 == 0 or index == len(future_jobs):
                print(f"prepared {index}/{len(future_jobs)} episodes", flush=True)

    for partition, items in rows.items():
        items.sort()
        with (output / f"{partition}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video"])
            writer.writerows(items)
    manifest_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    with (output / "clip_manifest.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["partition", "source_domain", "task", "episode", "start_frame_8fps",
                         "end_frame_8fps", "source", "video"])
        writer.writerows(manifest_rows)
    train_episodes = {(row[1], row[2], row[3]) for row in manifest_rows if row[0] == "train"}
    val_episodes = {(row[1], row[2], row[3]) for row in manifest_rows if row[0] == "val"}
    clean_clips = sum(row[1] == "clean" for row in manifest_rows)
    randomized_clips = sum(row[1] == "randomized" for row in manifest_rows)
    report = {"source": str(args.data_root.resolve()), "train_clips": len(rows["train"]),
              "val_clips": len(rows["val"]), "skipped_short_count": len(skipped_short),
              "train_episodes": len(train_episodes), "val_episodes": len(val_episodes),
              "frames_per_clip": 33, "fps": 8, "stride_seconds": args.stride_seconds,
              "clean_root": str(args.clean_root.resolve()) if args.clean_root else None,
              "clean_clips": clean_clips, "randomized_clips": randomized_clips,
              "clean_sampling": "one full-episode temporal resample to 33 frames; edge-repeat only if needed",
              "skipped_short_examples": skipped_short[:20], "max_videos": args.max_videos}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
