import math

import pytest
import torch

from scripts.physical_tokens.analyze_spatiotemporal_token_structure import (
    TokenStructureAccumulator,
)


def test_token_structure_accumulator_measures_redundancy_and_temporal_change() -> None:
    accumulator = TokenStructureAccumulator(
        num_transitions=2, queries_per_transition=2, codebook_size=2
    )
    latents = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]
    )
    accumulator.update(torch.tensor([[0, 0, 0, 1]]), latents)
    metrics = accumulator.compute()

    assert metrics["num_samples"] == 1
    assert metrics["num_tokens_per_sample"] == 4
    assert metrics["global_active_codes"] == 2
    assert metrics["global_active_ratio"] == pytest.approx(1.0)
    assert metrics["global_perplexity"] == pytest.approx(
        math.exp(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    )
    assert metrics["within_transition_unique_ratio"] == pytest.approx(0.75)
    assert metrics["within_transition_pairwise_token_change"] == pytest.approx(0.5)
    assert metrics["adjacent_transition_same_slot_change"] == pytest.approx(0.5)
    assert metrics["adjacent_transition_signature_change"] == pytest.approx(1.0)
    assert metrics["per_sample_unique_code_ratio"] == pytest.approx(1.0)
    assert metrics["mean_position_perplexity"] == pytest.approx(1.0)
    assert metrics["position_code_mi_fraction_of_code_entropy"] == pytest.approx(1.0)
    assert metrics["continuous_within_transition_relative_dispersion"] == pytest.approx(
        0.5
    )
    assert metrics["continuous_within_transition_pairwise_cosine"] == pytest.approx(0.5)
    assert metrics["continuous_within_transition_effective_rank"] == pytest.approx(0.5)


def test_token_structure_accumulator_rejects_bad_shape_and_ids() -> None:
    accumulator = TokenStructureAccumulator(2, 2, 3)
    with pytest.raises(ValueError, match="expected token IDs"):
        accumulator.update(torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="outside the codebook"):
        accumulator.update(torch.tensor([[0, 1, 2, 3]]))
