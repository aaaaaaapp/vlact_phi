import numpy as np

from starVLA.dataloader.lerobot_datasets import _apply_episode_selection


class FakeEpisodeDataset:
    def __init__(self):
        self._trajectory_ids = np.asarray([0, 1, 2], dtype=np.int64)
        self._trajectory_lengths = np.asarray([2, 1, 2], dtype=np.int64)
        self._all_steps = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
        self._filtered_trajectory_count = 3
        self._filtered_step_count = 5

    @property
    def trajectory_ids(self):
        return self._trajectory_ids

    @property
    def trajectory_lengths(self):
        return self._trajectory_lengths

    @property
    def all_steps(self):
        return self._all_steps


def test_episode_exclusion_filters_steps_and_trajectory_metadata():
    dataset = FakeEpisodeDataset()
    _apply_episode_selection(dataset, {"exclude_episode_ids": [1]})
    assert dataset.trajectory_ids.tolist() == [0, 2]
    assert dataset.trajectory_lengths.tolist() == [2, 2]
    assert dataset.all_steps == [(0, 0), (0, 1), (2, 0), (2, 1)]
    assert dataset._filtered_trajectory_count == 2
    assert dataset._filtered_step_count == 4


def test_episode_selection_rejects_conflicting_or_unknown_ids():
    dataset = FakeEpisodeDataset()
    for config, message in (
        ({"include_episode_ids": [0], "exclude_episode_ids": [1]}, "mutually exclusive"),
        ({"include_episode_ids": [7]}, "unknown IDs"),
    ):
        try:
            _apply_episode_selection(dataset, config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"selection must fail: {config}")
