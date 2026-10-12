import numpy as np
import torch

from starVLA.model.framework.QwenPIPhysicalTokens import QwenPIPhysicalTokens


def test_batch_token_ids_stacks_oracle_ids():
    values = QwenPIPhysicalTokens._batch_token_ids(
        [{"action_token_ids": np.array([1, 2])}, {"action_token_ids": [3, 4]}],
        "action_token_ids",
        torch.device("cpu"),
    )
    assert values.tolist() == [[1, 2], [3, 4]]


def test_batch_token_ids_rejects_partially_missing_values():
    try:
        QwenPIPhysicalTokens._batch_token_ids(
            [{"action_token_ids": [1, 2]}, {}],
            "action_token_ids",
            torch.device("cpu"),
        )
    except ValueError as error:
        assert "every sample or none" in str(error)
    else:
        raise AssertionError("partially missing token IDs must fail")
