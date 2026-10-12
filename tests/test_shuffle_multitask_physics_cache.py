import numpy as np

from scripts.physical_tokens.shuffle_multitask_physics_cache import shuffle_physics


def test_shuffle_stays_within_task_and_episode_and_preserves_other_fields():
    arrays = {
        "action_token_ids": np.arange(24).reshape(6, 4),
        "physics_token_ids": np.arange(12).reshape(6, 2),
        "task_ids": np.asarray([0, 0, 0, 1, 1, 1]),
        "task_names": np.asarray(["a", "b"]),
        "episode_ids": np.asarray([5, 5, 5, 5, 5, 5]),
        "base_indices": np.asarray([0, 1, 2, 0, 1, 2]),
    }
    shuffled = shuffle_physics(arrays, seed=42)
    for name in (
        "action_token_ids",
        "task_ids",
        "task_names",
        "episode_ids",
        "base_indices",
    ):
        np.testing.assert_array_equal(shuffled[name], arrays[name])
    assert not np.array_equal(shuffled["physics_token_ids"], arrays["physics_token_ids"])
    for task_id in (0, 1):
        rows = np.flatnonzero(arrays["task_ids"] == task_id)
        assert {
            tuple(value) for value in shuffled["physics_token_ids"][rows]
        } == {tuple(value) for value in arrays["physics_token_ids"][rows]}
