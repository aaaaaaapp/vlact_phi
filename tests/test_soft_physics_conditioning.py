import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "physical_tokens"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_multitask_dual_token_probe import decode_conditioned_soft_residuals


class Quantizer:
    def __init__(self):
        self.codebook = torch.tensor([[0.0], [2.0]])


class Tokenizer:
    num_tokens = 32
    codebook_size = 2
    num_transitions = 16
    queries_per_transition = 2
    quantizer = Quantizer()


def test_soft_codebook_expectation_reaches_conditioned_head():
    head = nn.Linear(15, 14, bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, 0] = 1.0
    logits = np.zeros((1, 32, 2), dtype=np.float32)
    base = np.zeros((1, 32, 14), dtype=np.float32)
    result = decode_conditioned_soft_residuals(
        logits, base, Tokenizer(), head, torch.device("cpu"), temperature=1.0
    )
    np.testing.assert_allclose(result, np.ones((1, 16, 14)), atol=1e-6)


def test_soft_codebook_expectation_rejects_bad_temperature():
    logits = np.zeros((1, 32, 2), dtype=np.float32)
    base = np.zeros((1, 32, 14), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        decode_conditioned_soft_residuals(
            logits, base, Tokenizer(), nn.Linear(15, 14), torch.device("cpu"), 0.0
        )
