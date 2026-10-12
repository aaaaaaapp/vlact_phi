#!/usr/bin/env python3
"""Exit successfully only when the preregistered D1 representation gate passed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(f"D1 comparison report does not exist: {args.report}")
    report = json.loads(args.report.read_text())
    gate = report.get("gate", {})
    if not gate.get("passed", False):
        failed = [name for name, passed in gate.get("checks", {}).items() if not passed]
        raise RuntimeError(f"D1 representation gate did not pass; failed checks: {failed}")
    print(f"D1 representation gate passed: {args.report}")


if __name__ == "__main__":
    main()
