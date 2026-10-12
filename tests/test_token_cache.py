from pathlib import Path

import numpy as np

from starVLA.dataloader.token_cache import DiscreteTokenCache, TokenConditionedDataset


class FakeDataset:
    all_steps = [(4, 0), (4, 1)]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"value": index}


def test_token_cache_join(tmp_path: Path):
    path = tmp_path / "tokens.npz"
    np.savez(
        path,
        action_token_ids=np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        episode_ids=np.asarray([4, 4], dtype=np.int64),
        base_indices=np.asarray([0, 1], dtype=np.int64),
    )
    cache = DiscreteTokenCache(path)
    assert isinstance(cache.data, dict)
    wrapped = TokenConditionedDataset(FakeDataset(), cache)
    assert wrapped[1]["value"] == 1
    assert wrapped[1]["action_token_ids"].tolist() == [3, 4]


def test_token_cache_can_attach_to_a_mixture_sample(tmp_path: Path):
    path = tmp_path / "tokens.npz"
    np.savez(
        path,
        action_token_ids=np.asarray([[5, 6]], dtype=np.int32),
        episode_ids=np.asarray([4], dtype=np.int64),
        base_indices=np.asarray([1], dtype=np.int64),
    )
    wrapped = TokenConditionedDataset(FakeDataset(), DiscreteTokenCache(path))
    sample = wrapped.attach_discrete_tokens({"value": 9}, 4, 1)
    assert sample["value"] == 9
    assert sample["action_token_ids"].tolist() == [5, 6]


def test_token_cache_fails_on_missing_step(tmp_path: Path):
    path = tmp_path / "tokens.npz"
    np.savez(
        path,
        action_token_ids=np.asarray([[1, 2]], dtype=np.int32),
        episode_ids=np.asarray([4], dtype=np.int64),
        base_indices=np.asarray([0], dtype=np.int64),
    )
    wrapped = TokenConditionedDataset(FakeDataset(), DiscreteTokenCache(path))
    try:
        wrapped[1]
    except KeyError as error:
        assert "(4, 1)" in str(error)
    else:
        raise AssertionError("missing cache entries must fail")
