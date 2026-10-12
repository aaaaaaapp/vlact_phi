import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_multitask_dual_token_probe import episode_local_shuffled_rows


def test_episode_local_shuffle_never_crosses_episode_or_keeps_multirow_group() -> None:
    panel = [
        {"episode_id": 5},
        {"episode_id": 5},
        {"episode_id": 5},
        {"episode_id": 20},
        {"episode_id": 20},
        {"episode_id": 49},
    ]

    shuffled = episode_local_shuffled_rows(panel, np.random.default_rng(42))
    episodes = np.asarray([item["episode_id"] for item in panel])

    np.testing.assert_array_equal(episodes[shuffled], episodes)
    assert np.all(shuffled[:3] != np.arange(3))
    assert np.all(shuffled[3:5] != np.arange(3, 5))
    assert shuffled[5] == 5


def test_episode_local_shuffle_is_seed_reproducible() -> None:
    panel = [{"episode_id": episode} for episode in (5, 5, 5, 20, 20, 20)]
    first = episode_local_shuffled_rows(panel, np.random.default_rng(7))
    second = episode_local_shuffled_rows(panel, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
