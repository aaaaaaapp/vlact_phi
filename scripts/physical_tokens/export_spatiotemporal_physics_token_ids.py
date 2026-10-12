#!/usr/bin/env python3
"""Encode the spatial-feature cache into 256 PhiZero-style token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_spatiotemporal_physical_tokenizer import SpatiotemporalFeatureDataset
from starVLA.model.modules.tokenizers import SpatiotemporalPhysicalLanguageTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict) -> SpatiotemporalPhysicalLanguageTokenizer:
    return SpatiotemporalPhysicalLanguageTokenizer(
        feature_dim=config["feature_dim"],
        num_frames=config["num_frames"],
        spatial_tokens=config["spatial_tokens"],
        queries_per_transition=config["queries_per_transition"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_heads=config["num_heads"],
        quantizer=config["quantizer"],
        codebook_size=config["codebook_size"],
        fsq_levels=tuple(config["fsq_levels"]),
        ema_decay=config["ema_decay"],
        commitment_weight=config["commitment_weight"],
        dead_code_steps=config["dead_code_steps"],
        token_routing=config.get("token_routing", "global_qformer"),
        action_aux_dim=config.get("action_aux_dim", 0),
        action_aux_weight=config.get("action_aux_weight", 0.0),
        action_velocity_weight=config.get("action_velocity_weight", 0.25),
        action_gripper_weight=config.get("action_gripper_weight", 0.1),
        action_aux_target_mode=config.get("action_aux_target_mode", "absolute"),
        action_aux_pooling=config.get("action_aux_pooling", "mean"),
    )


@torch.inference_mode()
def encode(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    dataset = SpatiotemporalFeatureDataset(input_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    token_batches = []
    for batch in loader:
        token_ids, _, _ = model.encode(
            batch["features"].to(device), batch["valid_mask"].to(device)
        )
        token_batches.append(token_ids.cpu().numpy().astype(np.int32))
    token_ids = np.concatenate(token_batches, axis=0)
    arrays = {
        "physics_token_ids": token_ids,
        "episode_ids": dataset.episode_ids,
        "base_indices": dataset.base_indices,
    }
    if dataset.task_ids is not None:
        arrays["task_ids"] = dataset.task_ids
        arrays["task_names"] = np.asarray(dataset.task_names)
    manifest = {
        "schema_version": 1,
        "kind": "oracle_future_spatiotemporal_physics_tokens",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "quantizer": checkpoint["model_config"]["quantizer"],
        "codebook_size": model.codebook_size,
        "num_samples": len(token_ids),
        "num_tokens": token_ids.shape[1],
        "num_transitions": model.num_transitions,
        "queries_per_transition": model.queries_per_transition,
        "token_routing": model.token_routing,
        "privileged_future_information": True,
        "task_qualified_keys": dataset.task_ids is not None,
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
