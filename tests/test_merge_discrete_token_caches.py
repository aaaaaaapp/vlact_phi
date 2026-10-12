from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/physical_tokens/merge_discrete_token_caches.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_merge_aligns_by_key_and_shuffle_stays_within_episode(tmp_path: Path) -> None:
    action_path = tmp_path / "action.npz"
    physics_path = tmp_path / "physics.npz"
    keys = np.asarray([[1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.int64)
    np.savez(
        action_path,
        action_token_ids=np.asarray([[10], [11], [20], [21]]),
        episode_ids=keys[:, 0],
        base_indices=keys[:, 1],
    )
    reverse = keys[::-1]
    np.savez(
        physics_path,
        physics_token_ids=np.asarray([[201], [200], [101], [100]]),
        episode_ids=reverse[:, 0],
        base_indices=reverse[:, 1],
    )
    _, arrays = MODULE["merge"](action_path, physics_path)
    assert arrays["physics_token_ids"].flatten().tolist() == [100, 101, 200, 201]

    _, shuffled = MODULE["merge"](action_path, physics_path, 42)
    assert shuffled["physics_token_ids"].flatten().tolist() == [101, 100, 201, 200]
