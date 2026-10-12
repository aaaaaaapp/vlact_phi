#!/usr/bin/env python3
"""Require both utility and correspondence gains from a residual probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("gate_metric") != "macro_task_action_mae":
        raise ValueError("residual probe must use macro_task_action_mae for gating")

    baseline = float(report["zero_residual_validation"]["macro_task_action_mae"])
    aligned = float(
        report["results"]["aligned_current_visual"]["validation"][
            "macro_task_action_mae"
        ]
    )
    shuffled = float(
        report["results"]["episode_shuffled_current_visual"]["validation"][
            "macro_task_action_mae"
        ]
    )
    utility_gain = (baseline - aligned) / baseline
    correspondence_gain = (shuffled - aligned) / shuffled
    threshold = args.minimum_relative_improvement
    passed = utility_gain >= threshold and correspondence_gain >= threshold
    gate = {
        "schema_version": 1,
        "kind": "continuous_residual_upper_bound_gate",
        "source_report": str(args.report.resolve()),
        "gate_metric": "macro_task_action_mae",
        "minimum_relative_improvement": threshold,
        "zero_residual_macro_mae": baseline,
        "aligned_macro_mae": aligned,
        "task_local_shuffled_macro_mae": shuffled,
        "aligned_relative_improvement_vs_zero_residual": utility_gain,
        "aligned_relative_improvement_vs_task_local_shuffled": correspondence_gain,
        "passed": passed,
    }
    rendered = json.dumps(gate, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
