#!/usr/bin/env python3
"""Compare aligned and episode-shuffled physics using the same DiT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_qwenpi_paired_evaluations import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    paired_episode_bootstrap,
)


def compare(shuffled: dict, aligned: dict) -> dict:
    exact = (
        "task_name",
        "checkpoint_sha256",
        "base_checkpoint_sha256",
        "action_init_checkpoint_sha256",
        "validation_episode_ids",
        "num_samples",
        "panel_seed",
        "sample_ids",
        "num_inference_timesteps",
        "metrics_space",
        "discrete_conditioning",
    )
    for key in exact:
        if shuffled.get(key) != aligned.get(key):
            raise ValueError(f"aligned/shuffled evaluations differ in {key}")
    if shuffled.get("token_cache_sha256") == aligned.get("token_cache_sha256"):
        raise ValueError("aligned and shuffled evaluations unexpectedly use the same cache")
    metrics = {}
    for key in LOWER_IS_BETTER:
        baseline = float(shuffled["metrics"][key])
        value = float(aligned["metrics"][key])
        metrics[key] = {
            "shuffled": baseline,
            "aligned": value,
            "relative_improvement": (baseline - value) / max(abs(baseline), 1e-12),
        }
    for key in HIGHER_IS_BETTER:
        baseline = float(shuffled["metrics"][key])
        value = float(aligned["metrics"][key])
        metrics[key] = {
            "shuffled": baseline,
            "aligned": value,
            "aligned_minus_shuffled": value - baseline,
        }
    result = {
        "schema_version": 1,
        "comparison": "same_checkpoint_aligned_vs_episode_local_shuffled_physics",
        "checkpoint_sha256": aligned["checkpoint_sha256"],
        "aligned_token_cache_sha256": aligned["token_cache_sha256"],
        "shuffled_token_cache_sha256": shuffled["token_cache_sha256"],
        "metrics": metrics,
    }
    result["uncertainty"] = paired_episode_bootstrap(
        shuffled["episode_metrics"], aligned["episode_metrics"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.shuffled.read_text()), json.loads(args.aligned.read_text()))
    report["shuffled_report"] = str(args.shuffled.resolve())
    report["aligned_report"] = str(args.aligned.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
