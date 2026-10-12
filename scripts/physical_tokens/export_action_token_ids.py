#!/usr/bin/env python3
"""Encode an action-chunk cache into versioned discrete token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.tokenizers import ActionChunkTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ActionDataset(Dataset):
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.actions = np.asarray(data["actions"])
            self.episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
            self.base_indices = np.asarray(data["base_indices"], dtype=np.int64)
            self.task_ids = (
                np.asarray(data["task_ids"], dtype=np.int64)
                if "task_ids" in data.files
                else None
            )
            self.task_names = (
                np.asarray(data["task_names"])
                if "task_names" in data.files
                else None
            )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.actions[index], dtype=np.float32))


def build_model(config: dict) -> ActionChunkTokenizer:
    return ActionChunkTokenizer(
        action_dim=config["action_dim"],
        horizon=config["horizon"],
        num_tokens=config["num_tokens"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        quantizer=config["quantizer"],
        codebook_size=config["codebook_size"],
        fsq_levels=tuple(config["fsq_levels"]),
        ema_decay=config["ema_decay"],
        commitment_weight=config["commitment_weight"],
        dead_code_steps=config["dead_code_steps"],
    )


@torch.inference_mode()
def encode(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    dataset = ActionDataset(args.input)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    batches = []
    for actions in loader:
        token_ids, _, _ = model.encode(actions.to(device))
        batches.append(token_ids.cpu().numpy().astype(np.int32))
    token_ids = np.concatenate(batches, axis=0)
    arrays = {
        "action_token_ids": token_ids,
        "episode_ids": dataset.episode_ids,
        "base_indices": dataset.base_indices,
    }
    if dataset.task_ids is not None:
        arrays["task_ids"] = dataset.task_ids
    if dataset.task_names is not None:
        arrays["task_names"] = dataset.task_names
    manifest = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "quantizer": checkpoint["model_config"]["quantizer"],
        "codebook_size": model.codebook_size,
        "num_samples": len(token_ids),
        "num_tokens": token_ids.shape[1],
    }
    return manifest, arrays


def main() -> None:
    args = parse_args()
    manifest, arrays = encode(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
