from scripts.physical_tokens.prepare_phizero_libero_videos import (
    frame_indices, split_name, window_starts,
)


def test_training_windows_cover_first_and_last_with_one_partial_tail():
    assert window_starts(40, 8) == [0]
    assert window_starts(81, 8) == [0, 8]
    assert window_starts(110, 8) == [0, 8, 16, 24, 29, 32]


def test_dense_cache_covers_all_decisions():
    assert window_starts(110, 1) == list(range(110))


def test_frame_sampling_and_padding():
    assert frame_indices(110, 0)[:5] == [0, 3, 5, 8, 10]
    assert frame_indices(110, 29)[-1] == 109
    assert frame_indices(40, 0)[-1] == 39
    assert frame_indices(110, 32)[-3:] == [107, 109, 109]


def test_split_is_stable_at_episode_level():
    assert split_name("suite", 7, 0.1) == split_name("suite", 7, 0.1)
    assert split_name("suite", 7, 0) == "train"
