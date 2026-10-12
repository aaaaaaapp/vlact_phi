#!/usr/bin/env python3
"""Compare two evaluations from a matched training experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compare_qwenpi_paired_evaluations import (
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER,
        paired_episode_bootstrap,
    )
except ImportError:
    from compare_qwenpi_paired_evaluations import (
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER,
        paired_episode_bootstrap,
    )


def compare(baseline: dict, candidate: dict, label: str) -> dict:
    for key in (
        "task_name",
        "base_checkpoint_sha256",
        "validation_episode_ids",
        "num_samples",
        "panel_seed",
        "sample_ids",
        "num_inference_timesteps",
        "metrics_space",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"matched evaluations differ in {key}")
    metrics = {}
    for key in LOWER_IS_BETTER:
        base = float(baseline["metrics"][key])
        value = float(candidate["metrics"][key])
        metrics[key] = {
            "baseline": base,
            "candidate": value,
            "relative_improvement": (base - value) / max(abs(base), 1e-12),
        }
    for key in HIGHER_IS_BETTER:
        base = float(baseline["metrics"][key])
        value = float(candidate["metrics"][key])
        metrics[key] = {
            "baseline": base,
            "candidate": value,
            "candidate_minus_baseline": value - base,
        }
    return {
        "schema_version": 1,
        "comparison": label,
        "task_name": baseline.get("task_name"),
        "base_checkpoint_sha256": baseline["base_checkpoint_sha256"],
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "metrics": metrics,
        "uncertainty": paired_episode_bootstrap(
            baseline["episode_metrics"], candidate["episode_metrics"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        args.label,
    )
    report["baseline_report"] = str(args.baseline.resolve())
    report["candidate_report"] = str(args.candidate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
