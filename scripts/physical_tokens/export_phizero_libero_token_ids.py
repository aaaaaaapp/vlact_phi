#!/usr/bin/env python3
"""Encode prepared LIBERO PhiZero clips into a keyed 64-token physics NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diffsynth.models.phizero import (
    PhiZeroModelConfig, PhysicalLanguageTokenizer, load_physical_language_tokenizer,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clip(path: Path, height: int, width: int) -> torch.Tensor:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    expected = 33 * height * width * 3
    if len(result.stdout) != expected:
        raise ValueError(f"Expected 33 frames of {width}x{height}: {path}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(33, height, width, 3).copy()
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().mul_(2 / 255).sub_(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--motion-height", type=int, default=256)
    parser.add_argument("--motion-width", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    all_records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    records = all_records
    if args.split != "all":
        records = [record for record in records if record["split"] == args.split]
    if args.max_clips is not None:
        records = records[:args.max_clips]
    if not records:
        raise ValueError("No clips selected")

    config = PhiZeroModelConfig.from_checkpoint(args.checkpoint)
    token_count = config.motion_tokenizer.physical_language_length(33)
    if token_count != 64 or config.motion_tokenizer.quantizer != "fsq":
        raise ValueError(f"Expected 64-token PhiZero FSQ checkpoint, got {token_count} tokens")
    with torch.device("meta"):
        tokenizer = PhysicalLanguageTokenizer(config.motion_tokenizer)
    load_physical_language_tokenizer(tokenizer, args.checkpoint)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = tokenizer.to(device=device, dtype=dtype).eval()

    task_names = sorted({record["task_name"] for record in all_records})
    task_ids_by_name = {name: index for index, name in enumerate(task_names)}
    ids_rows = []
    with torch.inference_mode():
        for row, record in enumerate(records):
            clip_path = args.manifest.parent / record["video"]
            video = load_clip(clip_path, args.height, args.width).to(device=device, dtype=dtype)
            motion_video = video.permute(0, 2, 1, 3, 4).reshape(33, 3, args.height, args.width)
            motion_video = F.interpolate(motion_video, size=(args.motion_height, args.motion_width),
                                         mode="bilinear", align_corners=False)
            motion_video = motion_video.reshape(1, 33, 3, args.motion_height,
                                                args.motion_width).permute(0, 2, 1, 3, 4)
            ids = tokenizer(motion_video).reshape(-1).cpu().numpy()
            if ids.shape != (64,) or ids.min() < 0 or ids.max() >= config.motion_tokenizer.codebook_size:
                raise ValueError(f"Invalid physics IDs at manifest row {row}")
            ids_rows.append(ids.astype(np.int32))
            if (row + 1) % 100 == 0:
                print(f"encoded {row + 1}/{len(records)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        physics_token_ids=np.stack(ids_rows),
        task_ids=np.asarray([task_ids_by_name[record["task_name"]] for record in records], dtype=np.int32),
        task_names=np.asarray(task_names),
        episode_ids=np.asarray([record["episode_index"] for record in records], dtype=np.int32),
        base_indices=np.asarray([record["base_index"] for record in records], dtype=np.int32),
    )
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.manifest),
        "count": len(records), "token_count": 64,
        "codebook_size": config.motion_tokenizer.codebook_size,
        "split": args.split,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
