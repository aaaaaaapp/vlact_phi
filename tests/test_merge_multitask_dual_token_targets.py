from pathlib import Path

import numpy as np

from scripts.physical_tokens.merge_multitask_dual_token_targets import (
    load,
    main,
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


def test_merge_keeps_only_complete_future_clips(tmp_path: Path, monkeypatch):
    keys = dict(task_ids=np.asarray([0, 0]), task_names=np.asarray(["first"]),
                episode_ids=np.asarray([1, 1]), base_indices=np.asarray([4, 5]))
    action = tmp_path / "action.npz"
    physics = tmp_path / "physics.npz"
    features = tmp_path / "features.npz"
    output = tmp_path / "joined.npz"
    np.savez(action, **keys, action_token_ids=np.asarray([[1] * 8, [2] * 8]))
    np.savez(physics, **keys, physics_token_ids=np.asarray([[3] * 64, [4] * 64]))
    np.savez(features, **keys,
             future_valid_mask=np.asarray([[True, True], [True, False]]))
    monkeypatch.setattr("sys.argv", ["merge", "--action-cache", str(action),
                                    "--physics-cache", str(physics),
                                    "--physics-feature-cache", str(features),
                                    "--output", str(output)])
    main()
    with np.load(output) as joined:
        assert joined["action_token_ids"].shape == (1, 8)
        assert joined["physics_token_ids"].shape == (1, 64)
        assert joined["base_indices"].tolist() == [4]
