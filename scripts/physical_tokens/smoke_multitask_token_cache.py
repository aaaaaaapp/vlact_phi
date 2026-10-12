#!/usr/bin/env python3
"""Verify task-qualified token joins against real RoboTwin samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.token_cache, allow_pickle=False) as cache:
        names = [str(value) for value in cache["task_names"].tolist()]
        expected = {
            (int(task), int(episode), int(base)): np.asarray(ids)
            for task, episode, base, ids in zip(
                cache["task_ids"],
                cache["episode_ids"],
                cache["base_indices"],
                cache["action_token_ids"],
            )
        }
    for task_id, name in enumerate(names):
        dataset = make_LeRobotSingleDataset(
            args.data_root,
            name,
            "robotwin_wrap32",
            data_cfg={
                "video_backend": "pyav",
                "lerobot_version": "v2.0",
                "discrete_token_cache": str(args.token_cache),
            },
        )
        episode, base = map(int, dataset.all_steps[0])
        actual = np.asarray(dataset[0]["action_token_ids"])
        target = expected[(task_id, episode, base)]
        if not np.array_equal(actual, target):
            raise RuntimeError(f"token mismatch for task {name}")
        print(f"{task_id} {name}: key=({episode},{base}), ids={actual.tolist()}")


if __name__ == "__main__":
    main()
