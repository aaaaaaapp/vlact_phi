"""Render held-out clips through a PhiZero checkpoint and make a frame sheet."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm
from torchvision.io import write_video

from diffsynth.pipelines.phizero import (
    PHIZERO_FPS,
    PHIZERO_NUM_FRAMES,
    PhiZeroReconstructionPipeline,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPS = (
    "artifacts/robotwin_full50_study/phizero_robotwin_videos/clips/place_shoe/episode_000039.mp4",
    "artifacts/robotwin_full50_study/phizero_robotwin_videos/clips/adjust_bottle/episode_000005.mp4",
    "artifacts/robotwin_full50_study/phizero_robotwin_videos/clips/place_mouse_pad/episode_000025.mp4",
)
FRAME_INDICES = (0, 8, 16, 24, 32)


def load_frames(path: Path) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    try:
        frames = [Image.fromarray(reader.get_data(i)).convert("RGB") for i in range(PHIZERO_NUM_FRAMES)]
    finally:
        reader.close()
    return frames


def make_contact_sheet(examples: list[tuple[str, list[Image.Image], list[Image.Image]]], output: Path) -> None:
    tile_w, tile_h, label_h = 320, 184, 28
    left_w, top_h = 170, 42
    sheet = Image.new(
        "RGB",
        (left_w + tile_w * len(FRAME_INDICES), top_h + len(examples) * 2 * (tile_h + label_h)),
        "#20252b",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, frame_index in enumerate(FRAME_INDICES):
        draw.text((left_w + column * tile_w + 8, 14), f"frame {frame_index}", fill="white", font=font)
    for case_index, (name, source, generated) in enumerate(examples):
        for row_offset, (row_name, frames) in enumerate((("source", source), ("reconstruction", generated))):
            y = top_h + (case_index * 2 + row_offset) * (tile_h + label_h)
            draw.text((10, y + 8), f"{name}\n{row_name}", fill="white", font=font, spacing=5)
            for column, frame_index in enumerate(FRAME_INDICES):
                tile = ImageOps.fit(frames[frame_index].convert("RGB"), (tile_w, tile_h), method=Image.Resampling.LANCZOS)
                x = left_w + column * tile_w
                sheet.paste(tile, (x, y))
                draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#707780")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dit-path", nargs="+", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipeline = PhiZeroReconstructionPipeline.from_pretrained(
        args.checkpoint,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
    )
    examples = []
    for clip_value in tqdm(args.clips, desc="Reconstruct clips"):
        clip_path = Path(clip_value)
        if not clip_path.is_absolute():
            clip_path = ROOT / clip_path
        source_frames = load_frames(clip_path)
        tokens, reconstructed = pipeline.reconstruct(
            source_frames,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            progress=True,
        )
        case_name = f"{clip_path.parent.name}_{clip_path.stem}"
        case_dir = args.output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "physical_tokens.txt").write_text(" ".join(map(str, tokens.tolist())) + "\n")
        video = torch.from_numpy(
            np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in reconstructed])
        )
        write_video(
            str(case_dir / "reconstruction.mp4"),
            video,
            fps=PHIZERO_FPS,
            video_codec="libx264",
            options={"pix_fmt": "yuv420p", "crf": "23"},
        )
        examples.append((case_name, source_frames, reconstructed))
        print(f"Saved {case_dir / 'reconstruction.mp4'}", flush=True)

    make_contact_sheet(examples, args.output_dir / "reconstruction_contact_sheet.png")
    print(f"Saved {args.output_dir / 'reconstruction_contact_sheet.png'}", flush=True)


if __name__ == "__main__":
    main()
