#!/usr/bin/env python3
"""Require a 256-token physics sequence to contain non-trivial query diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_gate(
    report: dict[str, Any],
    *,
    min_unique_ratio: float = 0.25,
    min_pairwise_change: float = 0.20,
) -> dict[str, Any]:
    if report.get("schema_version") != 1 or report.get("kind") != (
        "spatiotemporal_physics_token_structure"
    ):
        raise ValueError("token-structure report has an unsupported schema or kind")
    metrics = report.get("metrics", {})
    unique_ratio = float(metrics["within_transition_unique_ratio"])
    pairwise_change = float(metrics["within_transition_pairwise_token_change"])
    thresholds = {
        "within_transition_unique_ratio": float(min_unique_ratio),
        "within_transition_pairwise_token_change": float(min_pairwise_change),
    }
    checks = {
        "within_transition_unique_ratio": unique_ratio >= min_unique_ratio,
        "within_transition_pairwise_token_change": pairwise_change >= min_pairwise_change,
    }
    return {
        "schema_version": 1,
        "kind": "physics_token_structure_gate",
        "source_report": report.get("checkpoint"),
        "observed": {
            "within_transition_unique_ratio": unique_ratio,
            "within_transition_pairwise_token_change": pairwise_change,
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-unique-ratio", type=float, default=0.25)
    parser.add_argument("--min-pairwise-change", type=float, default=0.20)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = evaluate_gate(
        report,
        min_unique_ratio=args.min_unique_ratio,
        min_pairwise_change=args.min_pairwise_change,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        failed = [name for name, passed in result["checks"].items() if not passed]
        raise RuntimeError(f"physics token structure gate failed: {failed}")


if __name__ == "__main__":
    main()
