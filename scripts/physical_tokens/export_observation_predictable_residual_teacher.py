#!/usr/bin/env python3
"""Export episode-cross-fitted observation-predictable residual targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from probe_observation_predictable_residual import (
    action_one_hot,
    keyed_alignment,
    load,
    project_visual,
    task_one_hot,
)
from probe_physics_action_relevance import fit_ridge, predict, standardize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-feature-cache", type=Path, required=True)
    parser.add_argument("--residual-action-cache", type=Path, required=True)
    parser.add_argument("--action-token-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    visual_path = args.visual_feature_cache.resolve()
    residual_path = args.residual_action_cache.resolve()
    token_path = args.action_token_cache.resolve()
    visual = load(visual_path)
    residual = load(residual_path)
    tokens = load(token_path)
    keyed_alignment(residual, visual, "visual feature cache")
    keyed_alignment(residual, tokens, "action token cache")

    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    if not gate.get("passed", False):
        raise ValueError("D18 gate did not pass")
    source_report = json.loads(Path(gate["source_report"]).read_text(encoding="utf-8"))
    alpha = float(
        source_report["results"]["aligned_current_visual"]["selected_alpha"]
    )
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    training_episodes = np.asarray(split["training_episode_ids"], dtype=np.int64)
    validation_episodes = np.asarray(split["validation_episode_ids"], dtype=np.int64)
    if args.folds < 2 or args.folds > len(training_episodes):
        raise ValueError("fold count must be in [2, number of training episodes]")

    task_ids = residual.get(
        "task_ids", np.zeros(len(residual["episode_ids"]), dtype=np.int64)
    ).astype(np.int64)
    episode_ids = residual["episode_ids"].astype(np.int64)
    visual_rows = visual["frame_features"]
    if "source_frame_indices" in visual:
        visual_rows = visual_rows[visual["source_frame_indices"][:, 0]]
    projected = project_visual(
        visual_rows,
        task_ids,
        episode_ids,
        residual["base_indices"],
        [0],
        args.projection_dim,
        args.seed,
    )
    features = np.concatenate(
        (task_one_hot(task_ids), action_one_hot(tokens["action_token_ids"]), projected),
        axis=1,
    )
    targets = residual["actions"].astype(np.float32)
    flat_targets = targets.reshape(len(targets), -1)
    predictions = np.full_like(flat_targets, np.nan)

    generator = np.random.default_rng(args.seed)
    shuffled_training = training_episodes.copy()
    generator.shuffle(shuffled_training)
    fold_records = []
    for fold_index, held_out_episodes in enumerate(np.array_split(shuffled_training, args.folds)):
        fit_mask = np.isin(episode_ids, training_episodes) & ~np.isin(
            episode_ids, held_out_episodes
        )
        predict_mask = np.isin(episode_ids, held_out_episodes)
        fit_x, predict_x = standardize(features[fit_mask], features[predict_mask])
        model = fit_ridge(fit_x, flat_targets[fit_mask], alpha, args.device)
        predictions[predict_mask] = predict(predict_x, model)
        fold_records.append(
            {
                "fold": fold_index,
                "held_out_episode_ids": sorted(map(int, held_out_episodes)),
                "fit_samples": int(fit_mask.sum()),
                "predicted_samples": int(predict_mask.sum()),
            }
        )

    train_mask = np.isin(episode_ids, training_episodes)
    validation_mask = np.isin(episode_ids, validation_episodes)
    train_x, validation_x = standardize(features[train_mask], features[validation_mask])
    model = fit_ridge(train_x, flat_targets[train_mask], alpha, args.device)
    predictions[validation_mask] = predict(validation_x, model)
    covered = train_mask | validation_mask
    if not covered.all() or not np.isfinite(predictions).all():
        raise ValueError("cross-fitted teacher did not produce one finite prediction per row")
    teacher = predictions.reshape(targets.shape).astype(np.float32)

    output_arrays = {key: value for key, value in residual.items() if key != "actions"}
    output_arrays["actions"] = teacher
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "kind": "episode_cross_fitted_observation_predictable_residual_teacher",
        "visual_feature_cache": str(visual_path),
        "visual_feature_cache_sha256": sha256_file(visual_path),
        "residual_action_cache": str(residual_path),
        "residual_action_cache_sha256": sha256_file(residual_path),
        "action_token_cache": str(token_path),
        "action_token_cache_sha256": sha256_file(token_path),
        "gate_report": str(args.gate_report.resolve()),
        "gate_report_sha256": sha256_file(args.gate_report),
        "ridge_alpha": alpha,
        "projection_dim": args.projection_dim,
        "projection_seed": args.seed,
        "folds": fold_records,
        "validation_episode_ids": sorted(map(int, validation_episodes)),
        "validation_fit_samples": int(train_mask.sum()),
        "validation_predicted_samples": int(validation_mask.sum()),
        "shape": list(teacher.shape),
        "teacher_target_mae": float(np.abs(teacher - targets).mean()),
        "no_episode_self_fit": True,
    }
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
