#!/usr/bin/env python3
"""Create PhiZero-compatible 33-frame/8-FPS RoboTwin clips and split CSVs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="For a small smoke test; counts source episodes across all tasks.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent ffmpeg workers for video preparation.")
    args = parser.parse_args()
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

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
    for task in tasks:
        source_dir = args.data_root / task / "videos" / "chunk-000" / "observation.images.cam_high"
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
            destination = output / "clips" / task / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((partition, source, destination))
        if args.max_videos is not None and seen >= args.max_videos:
            break

    def prepare_clip(job):
        partition, source, destination = job
        duration, _ = probe(source)
        if duration < 4.125:
            return partition, source, destination, duration, 0
        valid_existing = False
        if destination.exists():
            try:
                _, existing_frames = probe(destination)
                valid_existing = existing_frames == 33
            except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError):
                pass
        if not valid_existing:
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-vf", "fps=8", "-frames:v", "33", "-an", "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "18", str(destination)],
                check=True,
            )
        _, frames = probe(destination)
        if frames != 33:
            raise RuntimeError(f"Expected 33 frames, got {frames}: {destination}")
        return partition, source, destination, duration, frames

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_clip, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            partition, source, destination, duration, frames = future.result()
            if frames == 0:
                skipped_short.append(str(source))
            else:
                rows[partition].append([str(destination)])
            if index % 100 == 0 or index == len(futures):
                print(f"prepared {index}/{len(futures)} clips", flush=True)

    for partition, items in rows.items():
        items.sort()
        with (output / f"{partition}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video"])
            writer.writerows(items)
    report = {"source": str(args.data_root.resolve()), "train_clips": len(rows["train"]),
              "val_clips": len(rows["val"]), "skipped_short_count": len(skipped_short),
              "skipped_short_examples": skipped_short[:20], "max_videos": args.max_videos}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
