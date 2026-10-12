import torch

from starVLA.model.framework.QwenActionTokenProbe import last_valid_hidden


def test_last_valid_hidden_supports_left_and_right_padding():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    selected = last_valid_hidden(hidden, mask)
    assert torch.equal(selected[0], hidden[0, 1])
    assert torch.equal(selected[1], hidden[1, 3])


def test_last_valid_hidden_rejects_empty_rows():
    hidden = torch.zeros(1, 2, 3)
    mask = torch.zeros(1, 2, dtype=torch.long)
    try:
        last_valid_hidden(hidden, mask)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty attention row should fail")


def test_last_valid_hidden_rejects_shape_mismatch():
    try:
        last_valid_hidden(torch.zeros(2, 3, 4), torch.ones(2, 2))
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("mismatched sequence shapes should fail")
