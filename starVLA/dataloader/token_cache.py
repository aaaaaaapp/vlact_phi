"""Attach cached discrete conditions to a LeRobot single-task dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class DiscreteTokenCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        # NPZ members are compressed streams, so mmap_mode does not make their
        # reads independent. Keeping NpzFile open across forked DataLoader
        # workers lets them race on one zip handle and causes sporadic CRC-32
        # failures. Materialize this small token cache before workers fork.
        with np.load(self.path, allow_pickle=False) as archive:
            self.data = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
        required = {"episode_ids", "base_indices"}
        missing = required - set(self.data)
        if missing:
            raise ValueError(f"token cache is missing keys: {sorted(missing)}")
        if not ({"action_token_ids", "physics_token_ids"} & set(self.data)):
            raise ValueError("token cache must contain action_token_ids or physics_token_ids")
        episode_ids = self.data["episode_ids"]
        base_indices = self.data["base_indices"]
        if len(episode_ids) != len(base_indices):
            raise ValueError("episode_ids and base_indices must have equal length")
        self._task_name_to_id = None
        if "task_ids" in self.data:
            if "task_names" not in self.data:
                raise ValueError("task-qualified token cache is missing task_names")
            task_ids = self.data["task_ids"]
            task_names = [str(value) for value in self.data["task_names"].tolist()]
            if len(task_ids) != len(episode_ids):
                raise ValueError("task_ids and episode_ids must have equal length")
            if not np.array_equal(np.unique(task_ids), np.arange(len(task_names))):
                raise ValueError("task_ids must be contiguous and match task_names")
            self._task_name_to_id = {name: index for index, name in enumerate(task_names)}
            keys = [
                (int(task), int(episode), int(base))
                for task, episode, base in zip(task_ids, episode_ids, base_indices)
            ]
            key_description = "(task_id, episode_id, base_index)"
        else:
            keys = [(int(episode), int(base)) for episode, base in zip(episode_ids, base_indices)]
            key_description = "(episode_id, base_index)"
        if len(set(keys)) != len(keys):
            raise ValueError(f"token cache contains duplicate {key_description} keys")
        self._row_by_key = {key: row for row, key in enumerate(keys)}

    def lookup(
        self,
        episode_id: int,
        base_index: int,
        task_name: str | None = None,
    ) -> dict[str, np.ndarray]:
        if self._task_name_to_id is not None:
            if task_name not in self._task_name_to_id:
                raise KeyError(
                    f"token cache {self.path} has no task named {task_name!r}; "
                    f"available={sorted(self._task_name_to_id)}"
                )
            key = (self._task_name_to_id[task_name], int(episode_id), int(base_index))
        else:
            key = (int(episode_id), int(base_index))
        if key not in self._row_by_key:
            raise KeyError(f"token cache {self.path} has no entry for {key}")
        row = self._row_by_key[key]
        result = {}
        for name in ("action_token_ids", "physics_token_ids"):
            if name in self.data:
                result[name] = np.asarray(self.data[name][row], dtype=np.int64)
        return result


class TokenConditionedDataset(Dataset):
    """Dataset wrapper that joins cached token IDs using the source step key."""

    def __init__(
        self, dataset: Dataset, token_cache: DiscreteTokenCache, task_name: str | None = None
    ) -> None:
        if not hasattr(dataset, "all_steps"):
            raise TypeError("wrapped dataset must expose all_steps")
        self.dataset = dataset
        self.token_cache = token_cache
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        episode_id, base_index = self.dataset.all_steps[index]
        sample = dict(self.dataset[index])
        return self.attach_discrete_tokens(sample, episode_id, base_index)

    def attach_discrete_tokens(
        self,
        sample: dict,
        episode_id: int,
        base_index: int,
    ) -> dict:
        sample.update(
            self.token_cache.lookup(
                int(episode_id), int(base_index), task_name=self.task_name
            )
        )
        return sample

    def __getattr__(self, name: str):
        if name in {"dataset", "token_cache", "task_name"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)
