import pytest
import torch

from starVLA.model.modules.conditioning import DiscreteTokenConditioner


def test_physics_codebook_initialization_preserves_distances():
    conditioner = DiscreteTokenConditioner(
        16, action_vocab_size=8, physics_vocab_size=4
    )
    codebook = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    conditioner.initialize_physics_code_embeddings(codebook)
    expected = codebook.repeat(1, 2) / (2**0.5)
    torch.testing.assert_close(conditioner.physics_embedding.weight, expected)
    torch.testing.assert_close(
        torch.cdist(
            conditioner.physics_embedding.weight,
            conditioner.physics_embedding.weight,
        ),
        torch.cdist(codebook, codebook),
    )


def test_physics_codebook_initialization_rejects_incompatible_shape():
    conditioner = DiscreteTokenConditioner(
        10, action_vocab_size=8, physics_vocab_size=4
    )
    with pytest.raises(ValueError, match="must be divisible"):
        conditioner.initialize_physics_code_embeddings(torch.zeros(4, 4))


def test_action_codebook_initialization_preserves_distances():
    conditioner = DiscreteTokenConditioner(
        16, action_vocab_size=4, physics_vocab_size=8
    )
    codebook = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    conditioner.initialize_action_code_embeddings(codebook)
    expected = codebook.repeat(1, 2) / (2**0.5)
    torch.testing.assert_close(conditioner.action_embedding.weight, expected)


def test_disabled_conditioner_is_an_exact_bypass():
    conditioner = DiscreteTokenConditioner(16, action_vocab_size=32, physics_vocab_size=64)
    prefixes = [torch.randn(2, 7, 16), torch.randn(2, 7, 16)]
    output = conditioner(
        prefixes,
        action_token_ids=torch.randint(0, 32, (2, 4)),
        physics_token_ids=torch.randint(0, 64, (2, 3)),
        enabled=False,
    )
    assert all(result is source for result, source in zip(output, prefixes))


def test_conditioner_appends_action_and_physics_tokens_to_every_layer():
    torch.manual_seed(3)
    conditioner = DiscreteTokenConditioner(16, action_vocab_size=32, physics_vocab_size=64)
    prefixes = [torch.randn(2, 7, 16), torch.randn(2, 7, 16)]
    output = conditioner(
        prefixes,
        action_token_ids=torch.randint(0, 32, (2, 4)),
        physics_token_ids=torch.randint(0, 64, (2, 3)),
    )
    assert [tuple(value.shape) for value in output] == [(2, 14, 16), (2, 14, 16)]
    assert torch.equal(output[0][:, :7], prefixes[0])
    assert torch.equal(output[1][:, :7], prefixes[1])
    assert torch.allclose(output[0][:, 7:], output[1][:, 7:])


def test_grouped_concat_consumes_all_physics_ids_but_limits_prefix_length():
    conditioner = DiscreteTokenConditioner(
        16,
        action_vocab_size=32,
        physics_vocab_size=64,
        max_physics_tokens=256,
        physics_fusion_mode="grouped_concat",
        physics_group_size=16,
    ).eval()
    prefix = torch.randn(2, 7, 16)
    action_ids = torch.randint(0, 32, (2, 4))
    physics_ids = torch.randint(0, 64, (2, 256))
    output = conditioner(
        [prefix],
        action_token_ids=action_ids,
        physics_token_ids=physics_ids,
    )[0]
    assert output.shape == (2, 7 + 4 + 16, 16)
    torch.testing.assert_close(output[:, :7], prefix)


def test_grouped_concat_rejects_incomplete_physics_group():
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        physics_fusion_mode="grouped_concat",
        physics_group_size=4,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        conditioner.build_tokens(
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 7, dtype=torch.long),
        )


def test_zero_gate_is_not_confused_with_the_explicit_bypass():
    conditioner = DiscreteTokenConditioner(8, action_vocab_size=16, gate_init=0.0)
    prefix = torch.randn(1, 5, 8)
    output = conditioner([prefix], action_token_ids=torch.zeros(1, 2, dtype=torch.long))
    assert output[0].shape == (1, 7, 8)
    assert torch.count_nonzero(output[0][:, 5:]) == 0


def test_physics_condition_dropout_omits_physics_but_keeps_action(monkeypatch):
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        physics_condition_dropout=0.5,
    )
    conditioner.train()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.1))
    tokens = conditioner.build_tokens(
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.long),
    )
    assert tokens.shape == (2, 4, 8)


def test_group_dropout_decision_is_broadcast_in_distributed_training(monkeypatch):
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        physics_condition_dropout=0.5,
    ).train()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.9))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast_as_dropped(draw, src):
        assert src == 0
        draw.fill_(0.1)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_as_dropped)
    assert conditioner._drop_group(0.5) is True


def test_condition_dropout_is_disabled_during_evaluation(monkeypatch):
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        action_condition_dropout=0.9,
        physics_condition_dropout=0.9,
    )
    conditioner.eval()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    tokens = conditioner.build_tokens(
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.long),
    )
    assert tokens.shape == (2, 7, 8)


def test_separate_gate_initialization_and_dropout_validation():
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        action_gate_init=0.2,
        physics_gate_init=0.5,
    )
    assert conditioner.action_gate.item() == pytest.approx(0.2)
    assert conditioner.physics_gate.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="physics_condition_dropout"):
        DiscreteTokenConditioner(
            8,
            action_vocab_size=16,
            physics_vocab_size=16,
            physics_condition_dropout=1.0,
        )


def test_residual_physics_fusion_starts_as_exact_action_only_conditioning():
    torch.manual_seed(9)
    conditioner = DiscreteTokenConditioner(
        16,
        action_vocab_size=32,
        physics_vocab_size=64,
        max_physics_tokens=256,
        physics_fusion_mode="residual_cross_attention",
        physics_fusion_dim=8,
        physics_fusion_heads=2,
        physics_gate_init=0.5,
    ).eval()
    prefixes = [torch.randn(2, 7, 16), torch.randn(2, 7, 16)]
    action_ids = torch.randint(0, 32, (2, 4))
    physics_ids = torch.randint(0, 64, (2, 256))
    expected_action = conditioner._embed_action(action_ids)
    output = conditioner(
        prefixes,
        action_token_ids=action_ids,
        physics_token_ids=physics_ids,
    )
    assert [tuple(value.shape) for value in output] == [(2, 11, 16), (2, 11, 16)]
    for result, prefix in zip(output, prefixes):
        assert torch.equal(result[:, :7], prefix)
        assert torch.equal(result[:, 7:], expected_action)


def test_zero_gate_makes_trained_residual_fusion_an_exact_bypass():
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        physics_fusion_mode="residual_cross_attention",
        physics_fusion_dim=4,
        physics_fusion_heads=1,
        physics_gate_init=0.0,
    ).eval()
    torch.nn.init.normal_(conditioner.physics_output_proj.weight)
    prefix = torch.randn(1, 5, 8)
    action_ids = torch.zeros(1, 2, dtype=torch.long)
    output = conditioner(
        [prefix],
        action_token_ids=action_ids,
        physics_token_ids=torch.zeros(1, 3, dtype=torch.long),
    )[0]
    expected = torch.cat((prefix, conditioner._embed_action(action_ids)), dim=1)
    assert torch.equal(output, expected)


def test_disabled_residual_branch_is_an_exact_action_only_control():
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=16,
        physics_fusion_mode="residual_cross_attention",
        physics_fusion_dim=4,
        physics_fusion_heads=1,
        physics_condition_enabled=False,
    ).eval()
    torch.nn.init.normal_(conditioner.physics_output_proj.weight)
    prefix = torch.randn(1, 5, 8)
    action_ids = torch.zeros(1, 2, dtype=torch.long)
    output = conditioner(
        [prefix],
        action_token_ids=action_ids,
        physics_token_ids=torch.ones(1, 3, dtype=torch.long),
    )[0]
    expected = torch.cat((prefix, conditioner._embed_action(action_ids)), dim=1)
    assert torch.equal(output, expected)


def test_dit_physics_mode_keeps_only_action_tokens_in_vlm_prefix():
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=32,
        max_action_tokens=4,
        max_physics_tokens=256,
        physics_fusion_mode="dit_multilayer_cross_attention",
    ).eval()
    prefix = torch.randn(2, 5, 8)
    action_ids = torch.zeros(2, 4, dtype=torch.long)
    output = conditioner(
        [prefix],
        action_token_ids=action_ids,
        physics_token_ids=None,
    )[0]
    assert output.shape == (2, 9, 8)
    assert torch.equal(output[:, :5], prefix)
    assert torch.equal(output[:, 5:], conditioner._embed_action(action_ids))
    assert conditioner.physics_embedding is None


def test_dit_physics_prefix_does_not_draw_unused_physics_dropout(monkeypatch):
    conditioner = DiscreteTokenConditioner(
        8,
        action_vocab_size=16,
        physics_vocab_size=32,
        physics_fusion_mode="dit_multilayer_cross_attention",
        action_condition_dropout=0.0,
        physics_condition_dropout=0.9,
    ).train()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefix conditioner must not draw unused physics dropout")
        ),
    )
    output = conditioner(
        [torch.randn(1, 3, 8)],
        action_token_ids=torch.zeros(1, 4, dtype=torch.long),
    )
    assert output[0].shape == (1, 7, 8)
