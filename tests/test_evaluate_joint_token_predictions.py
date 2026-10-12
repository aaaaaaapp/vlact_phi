from pathlib import Path

import numpy as np

from scripts.physical_tokens.evaluate_joint_token_predictions import (
    train_position_majority_baseline,
)


def test_train_position_majority_baseline_excludes_validation_episodes(tmp_path: Path):
    cache_path = tmp_path / "tokens.npz"
    np.savez_compressed(
        cache_path,
        task_names=np.asarray(["task_a", "task_b"]),
        task_ids=np.asarray([0, 0, 0, 0, 1]),
        episode_ids=np.asarray([1, 2, 3, 9, 1]),
        action_token_ids=np.asarray(
            [[1, 2], [1, 3], [4, 3], [7, 7], [6, 6]], dtype=np.int32
        ),
        physics_token_ids=np.asarray(
            [[5, 8], [5, 9], [6, 9], [7, 7], [4, 4]], dtype=np.int32
        ),
    )
    action_targets = np.asarray([[1, 3], [4, 2]], dtype=np.int64)
    physics_targets = np.asarray([[5, 9], [6, 8]], dtype=np.int64)
    result = train_position_majority_baseline(
        cache_path,
        "task_a",
        validation_episodes=[9],
        action_targets=action_targets,
        physics_targets=physics_targets,
    )

    assert result["action"]["majority_token_ids"] == [1, 3]
    assert result["physics"]["majority_token_ids"] == [5, 9]
    assert result["action"]["token_accuracy"] == 0.5
    assert result["physics"]["token_accuracy"] == 0.5
