import pytest
import torch

from starVLA.model.framework.QwenActionTokenReasoner import (
    extract_generated_action_ids,
    format_action_token_solution,
)


def test_format_action_token_solution_uses_existing_special_tokens():
    result = format_action_token_solution([0, 17, 511], vocab_size=512)
    assert result == "<robot_action_0><robot_action_17><robot_action_511>"


def test_format_action_token_solution_rejects_out_of_range_code():
    with pytest.raises(ValueError, match="outside"):
        format_action_token_solution([512], vocab_size=512)


def test_extract_generated_action_ids_uses_only_generation_suffix():
    generated = torch.tensor(
        [
            [10, 11, 151669, 151686, 152180],
            [20, 21, 151670, 151671, 151672],
        ]
    )
    result = extract_generated_action_ids(
        generated,
        prompt_width=2,
        token_offset=151669,
        vocab_size=512,
        token_count=3,
    )
    assert torch.equal(result, torch.tensor([[0, 17, 511], [1, 2, 3]]))


def test_extract_generated_action_ids_rejects_non_action_token():
    with pytest.raises(ValueError, match="outside"):
        extract_generated_action_ids(
            torch.tensor([[10, 11, 151668]]),
            prompt_width=2,
            token_offset=151669,
            vocab_size=512,
            token_count=1,
        )
