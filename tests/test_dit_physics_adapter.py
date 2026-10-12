import pytest
import torch

from starVLA.model.modules.conditioning import (
    MultiLayerPhysicsTokenAdapter,
    resolve_injection_layers,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


def make_adapter(selection="all") -> MultiLayerPhysicsTokenAdapter:
    return MultiLayerPhysicsTokenAdapter(
        hidden_dim=16,
        physics_vocab_size=32,
        num_layers=6,
        injection_layers=selection,
        max_physics_tokens=256,
        num_heads=4,
        condition_dropout=0.0,
    ).eval()


def test_named_injection_layer_selections() -> None:
    assert resolve_injection_layers("first", 8) == (0,)
    assert resolve_injection_layers("early-4", 8) == (0, 1, 2, 3)
    assert resolve_injection_layers("all", 3) == (0, 1, 2)
    assert resolve_injection_layers("1,5", 8) == (1, 5)
    with pytest.raises(ValueError, match="outside"):
        resolve_injection_layers("0,8", 8)


def test_grouped_cyclic_shuffle_never_crosses_repeated_groups() -> None:
    ids = torch.tensor([[0], [1], [2], [0], [1], [2]])
    shuffled = LayerwiseFlowmatchingActionHead._cyclically_shuffle_conditions(
        ids, group_size=3
    )
    assert shuffled.tolist() == [[2], [0], [1], [2], [0], [1]]


def test_grouped_cyclic_shuffle_rejects_singleton_groups() -> None:
    with pytest.raises(ValueError, match="group_size >= 2"):
        LayerwiseFlowmatchingActionHead._cyclically_shuffle_conditions(
            torch.tensor([[0], [0]]), group_size=1
        )


def test_zero_initialized_adapters_are_exact_for_256_physics_tokens() -> None:
    adapter = make_adapter("all")
    hidden = torch.randn(2, 12, 16)
    ids = torch.randint(0, 32, (2, 256))
    memory = adapter.prepare_memory(ids)
    assert memory is not None and memory.shape == (2, 256, 16)
    for layer_index in range(6):
        output = adapter.apply_layer(layer_index, hidden, memory)
        assert torch.equal(output, hidden)


def test_transition_mean_pooling_keeps_256_input_tokens_as_16_memory_slots() -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=16,
        physics_vocab_size=32,
        num_layers=1,
        max_physics_tokens=256,
        num_heads=4,
        condition_dropout=0.0,
        memory_pooling="transition_mean",
        tokens_per_transition=16,
    ).eval()
    ids = torch.randint(0, 32, (2, 256))
    memory = adapter.prepare_memory(ids)
    assert memory is not None
    assert memory.shape == (2, 16, 16)


def test_transition_mean_pooling_rejects_partial_transition() -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=8,
        num_layers=1,
        num_heads=2,
        memory_pooling="transition_mean",
        tokens_per_transition=4,
    )
    with pytest.raises(ValueError, match="divisible"):
        adapter.prepare_memory(torch.zeros(1, 7, dtype=torch.long))


def test_temporal_film_aligns_each_transition_to_two_action_queries() -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=16,
        num_layers=1,
        max_physics_tokens=256,
        num_heads=2,
        condition_dropout=0.0,
        memory_pooling="transition_mean",
        tokens_per_transition=16,
        adapter_type="temporal_film",
    ).eval()
    ids = torch.randint(0, 16, (2, 256))
    memory = adapter.prepare_memory(ids)
    hidden = torch.randn(2, 32, 8)
    output = adapter.apply_layer(0, hidden, memory)
    assert memory is not None and memory.shape == (2, 16, 8)
    assert torch.equal(output, hidden)
    torch.nn.init.normal_(adapter.adapters["0"].modulation.weight, std=0.1)
    output = adapter.apply_layer(0, hidden, memory)
    assert output.shape == hidden.shape
    assert not torch.equal(output, hidden)


def test_temporal_film_rejects_unaligned_action_query_count() -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=16,
        num_layers=1,
        num_heads=2,
        condition_dropout=0.0,
        memory_pooling="transition_mean",
        tokens_per_transition=16,
        adapter_type="temporal_film",
    ).eval()
    memory = adapter.prepare_memory(torch.randint(0, 16, (1, 256)))
    with pytest.raises(ValueError, match="divisible"):
        adapter.apply_layer(0, torch.randn(1, 31, 8), memory)


def test_hard_disabled_and_unselected_layers_return_same_tensor() -> None:
    adapter = make_adapter("first")
    torch.nn.init.normal_(adapter.adapters["0"].output_proj.weight)
    hidden = torch.randn(2, 5, 16)
    memory = adapter.prepare_memory(torch.randint(0, 32, (2, 10)))
    assert adapter.apply_layer(0, hidden, memory, enabled=False) is hidden
    assert adapter.apply_layer(1, hidden, memory) is hidden
    assert adapter.prepare_memory(torch.zeros(2, 10, dtype=torch.long), enabled=False) is None


def test_enabled_adapter_changes_hidden_states_after_output_projection_trains() -> None:
    adapter = make_adapter("first")
    torch.nn.init.normal_(adapter.adapters["0"].output_proj.weight, std=0.1)
    hidden = torch.randn(2, 5, 16)
    memory = adapter.prepare_memory(torch.randint(0, 32, (2, 10)))
    output = adapter.apply_layer(0, hidden, memory)
    assert output.shape == hidden.shape
    assert not torch.equal(output, hidden)


def test_codebook_initialization_preserves_geometry_by_repetition() -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=3,
        num_layers=1,
        num_heads=2,
        condition_dropout=0.0,
    )
    codebook = torch.tensor([[1.0, 2.0], [3.0, 5.0], [-2.0, 4.0]])
    adapter.initialize_code_embeddings(codebook)
    expected = codebook.repeat(1, 4) / 2.0
    assert torch.equal(adapter.physics_embedding.weight, expected)
    original_distances = torch.cdist(codebook, codebook)
    initialized_distances = torch.cdist(
        adapter.physics_embedding.weight, adapter.physics_embedding.weight
    )
    assert torch.allclose(initialized_distances, original_distances)


def test_codebook_initialization_rejects_incompatible_shapes() -> None:
    adapter = make_adapter("first")
    with pytest.raises(ValueError, match="shape"):
        adapter.initialize_code_embeddings(torch.zeros(31, 4))
    with pytest.raises(ValueError, match="divisible"):
        adapter.initialize_code_embeddings(torch.zeros(32, 3))


def test_physics_condition_dropout_is_decided_once_when_memory_is_prepared(monkeypatch) -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=16,
        num_layers=2,
        num_heads=2,
        condition_dropout=0.2,
    ).train()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    assert adapter.prepare_memory(torch.zeros(1, 4, dtype=torch.long)) is None


def test_dropped_adapter_only_batch_has_zero_gradients_without_output_change(
    monkeypatch,
) -> None:
    adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=16,
        physics_vocab_size=8,
        num_layers=2,
        injection_layers="first",
        max_physics_tokens=4,
        num_heads=2,
        condition_dropout=0.5,
    ).train()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(()))
    hidden = torch.randn(2, 3, 16)

    memory = adapter.prepare_memory(torch.zeros(2, 4, dtype=torch.long))
    output = adapter.attach_zero_gradient(hidden)
    output.square().mean().backward()

    assert memory is None
    assert torch.equal(output, hidden)
    assert output.requires_grad
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in adapter.parameters())
