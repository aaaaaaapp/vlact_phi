#!/usr/bin/env python3
"""Require a useful oracle upper bound before the full-data VLM probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_gate(report: dict) -> dict:
    if report.get("kind") != "action_conditioned_oracle_physics_residual_probe":
        raise ValueError("unexpected conditioned-probe report kind")
    improvements = report["final"]["aligned_relative_improvement"]
    versus_action = float(improvements["action_only"]["joint_mae"])
    versus_shuffle = float(improvements["episode_shuffled"]["joint_mae"])
    checks = {
        "oracle_improves_action_only_by_1pct": versus_action >= 0.01,
        "oracle_improves_episode_local_shuffle_by_1pct": versus_shuffle >= 0.01,
    }
    return {
        "schema_version": 1,
        "gate": "d30_action_conditioned_oracle_upper_bound",
        "metric": "macro_joint_mae",
        "threshold": 0.01,
        "oracle_relative_improvement_vs_action_only": versus_action,
        "oracle_relative_improvement_vs_episode_local_shuffle": versus_shuffle,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_gate(json.loads(args.input.read_text()))
    report["input"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
