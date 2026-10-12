from pathlib import Path

import numpy as np

from scripts.physical_tokens.merge_multitask_dual_token_targets import (
    load,
    qualified_keys,
)


def test_qualified_keys_preserve_task_identity(tmp_path: Path):
    path = tmp_path / "tokens.npz"
    np.savez(
        path,
        physics_token_ids=np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        task_ids=np.asarray([0, 1], dtype=np.int64),
        task_names=np.asarray(["first", "second"]),
        episode_ids=np.asarray([0, 0], dtype=np.int64),
        base_indices=np.asarray([7, 7], dtype=np.int64),
    )
    arrays = load(path, "physics_token_ids")
    assert qualified_keys(arrays) == [(0, 0, 7), (1, 0, 7)]
