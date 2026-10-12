#!/usr/bin/env python3
"""Encode a future-feature cache into physical-language token IDs."""

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

from starVLA.model.modules.tokenizers import PhysicalLanguageTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.features = np.asarray(data["feature_deltas"])
            self.valid_mask = np.asarray(data["future_valid_mask"])
            self.episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
            self.base_indices = np.asarray(data["base_indices"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(np.array(self.features[index], dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.array(self.valid_mask[index], dtype=np.bool_)),
        }


def build_model(config: dict) -> PhysicalLanguageTokenizer:
    return PhysicalLanguageTokenizer(
        feature_dim=config["feature_dim"],
        horizon=config["horizon"],
        num_tokens=config["num_tokens"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        quantizer=config["quantizer"],
        codebook_size=config["codebook_size"],
        fsq_levels=tuple(config["fsq_levels"]),
        ema_decay=config["ema_decay"],
        commitment_weight=config["commitment_weight"],
        dead_code_steps=config["dead_code_steps"],
    )


@torch.inference_mode()
def encode(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    checkpoint_path = args.checkpoint.resolve()
    input_path = args.input.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    dataset = FeatureDataset(input_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    batches = []
    for batch in loader:
        token_ids, _, _ = model.encode(
            batch["features"].to(device), batch["valid_mask"].to(device)
        )
        batches.append(token_ids.cpu().numpy().astype(np.int32))
    token_ids = np.concatenate(batches, axis=0)
    arrays = {
        "physics_token_ids": token_ids,
        "episode_ids": dataset.episode_ids,
        "base_indices": dataset.base_indices,
    }
    manifest = {
        "schema_version": 1,
        "kind": "oracle_future_physics_tokens",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "quantizer": checkpoint["model_config"]["quantizer"],
        "codebook_size": model.codebook_size,
        "num_samples": len(token_ids),
        "num_tokens": token_ids.shape[1],
        "privileged_future_information": True,
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
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
