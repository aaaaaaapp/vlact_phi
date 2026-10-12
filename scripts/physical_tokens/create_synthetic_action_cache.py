#!/usr/bin/env python3
"""Create a deterministic structured cache for tokenizer smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_dim < 2 or args.horizon < 2 or args.samples < args.episodes:
        raise ValueError("invalid synthetic cache dimensions")
    rng = np.random.default_rng(args.seed)
    time = np.linspace(0, 1, args.horizon, dtype=np.float32)[None, :, None]
    frequency = rng.integers(1, 4, size=(args.samples, 1, 1))
    phase = rng.uniform(-np.pi, np.pi, size=(args.samples, 1, args.action_dim))
    amplitude = rng.uniform(0.1, 1.0, size=(args.samples, 1, args.action_dim))
    actions = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    actions += rng.normal(0, 0.01, size=actions.shape)
    actions[..., -2:] = (actions[..., -2:] > 0).astype(np.float32)
    episode_ids = np.arange(args.samples, dtype=np.int64) % args.episodes
    valid_mask = np.ones((args.samples, args.horizon), dtype=np.bool_)
    dim_mask = np.ones((args.samples, args.action_dim), dtype=np.bool_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        actions=actions.astype(np.float32),
        action_valid_mask=valid_mask,
        action_dim_mask=dim_mask,
        episode_ids=episode_ids,
    )
    print(f"wrote {args.samples} synthetic chunks to {args.output}")


if __name__ == "__main__":
    main()
