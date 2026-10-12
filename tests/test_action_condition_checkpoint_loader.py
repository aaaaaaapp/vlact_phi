from pathlib import Path
import hashlib

import pytest
import torch
import torch.nn as nn

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.model.modules.conditioning import (
    DiscreteTokenConditioner,
    MultiLayerPhysicsTokenAdapter,
)


class Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_model = nn.Module()
        self.action_model.base = nn.Linear(3, 4)
        self.action_model.physics_adapter = nn.Linear(4, 4)
        self.discrete_conditioner = nn.Module()
        self.discrete_conditioner.action_embedding = nn.Embedding(8, 4)
        self.discrete_conditioner.physics_embedding = nn.Embedding(8, 4)


def save_action_only_checkpoint(path: Path, target: Target) -> None:
    state = {
        key: value.clone()
        for key, value in target.state_dict().items()
        if "physics_" not in key
    }
    torch.save(state, path)


def test_action_init_allows_only_new_physics_parameters(tmp_path: Path) -> None:
    source = Target()
    checkpoint = tmp_path / "action_only.pt"
    save_action_only_checkpoint(checkpoint, source)
    target = Target()
    with torch.no_grad():
        target.action_model.base.weight.zero_()
        target.discrete_conditioner.action_embedding.weight.zero_()

    TrainerUtils.load_action_condition_checkpoint(target, checkpoint)

    assert torch.equal(target.action_model.base.weight, source.action_model.base.weight)
    assert torch.equal(
        target.discrete_conditioner.action_embedding.weight,
        source.discrete_conditioner.action_embedding.weight,
    )


def test_action_init_rejects_missing_original_action_parameter(tmp_path: Path) -> None:
    target = Target()
    checkpoint = tmp_path / "incomplete.pt"
    torch.save(
        {
            "action_model.base.weight": target.action_model.base.weight.clone(),
            "discrete_conditioner.action_embedding.weight": (
                target.discrete_conditioner.action_embedding.weight.clone()
            ),
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="action_model"):
        TrainerUtils.load_action_condition_checkpoint(target, checkpoint)


def test_physics_adapter_initializes_from_sha_verified_ema_codebook(
    tmp_path: Path,
) -> None:
    target = Target()
    target.action_model.physics_adapter = MultiLayerPhysicsTokenAdapter(
        hidden_dim=8,
        physics_vocab_size=4,
        num_layers=1,
        num_heads=2,
        condition_dropout=0.0,
    )
    codebook = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    checkpoint = tmp_path / "tokenizer.pt"
    torch.save({"model": {"quantizer.codebook": codebook}}, checkpoint)
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    metadata = TrainerUtils.initialize_physics_adapter_from_tokenizer(
        target, checkpoint, sha256
    )

    assert metadata["sha256"] == sha256
    assert metadata["codebook_shape"] == [4, 2]
    expected = codebook.repeat(1, 4) / 2.0
    assert torch.equal(
        target.action_model.physics_adapter.physics_embedding.weight, expected
    )
    with pytest.raises(ValueError, match="mismatch"):
        TrainerUtils.initialize_physics_adapter_from_tokenizer(
            target, checkpoint, "0" * 64
        )


def test_action_conditioner_initializes_from_sha_verified_ema_codebook(
    tmp_path: Path,
) -> None:
    target = nn.Module()
    target.pi_discrete_conditioner = DiscreteTokenConditioner(
        hidden_dim=8,
        action_vocab_size=4,
        physics_vocab_size=4,
    )
    codebook = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    checkpoint = tmp_path / "action_tokenizer.pt"
    torch.save({"model": {"quantizer.codebook": codebook}}, checkpoint)
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    metadata = TrainerUtils.initialize_action_conditioner_from_tokenizer(
        target, checkpoint, sha256
    )

    assert metadata["sha256"] == sha256
    expected = codebook.repeat(1, 4) / 2.0
    assert torch.equal(
        target.pi_discrete_conditioner.action_embedding.weight, expected
    )


def test_strict_extension_checkpoint_allows_only_declared_new_module(
    tmp_path: Path,
) -> None:
    source = nn.Module()
    source.base = nn.Linear(3, 4)
    checkpoint = tmp_path / "base.pt"
    torch.save(source.state_dict(), checkpoint)

    target = nn.Module()
    target.base = nn.Linear(3, 4)
    target.extension = nn.Linear(4, 2)
    TrainerUtils.load_extension_checkpoint(target, checkpoint, ["extension"])
    torch.testing.assert_close(target.base.weight, source.base.weight)


def test_strict_extension_checkpoint_rejects_missing_base_parameter(
    tmp_path: Path,
) -> None:
    target = nn.Module()
    target.base = nn.Linear(3, 4)
    target.extension = nn.Linear(4, 2)
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"base.weight": target.base.weight.clone()}, checkpoint)
    with pytest.raises(RuntimeError, match="base.bias"):
        TrainerUtils.load_extension_checkpoint(target, checkpoint, ["extension"])
