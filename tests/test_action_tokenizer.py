from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starVLA.model.modules.tokenizers import (
    ActionChunkTokenizer,
    EMAVectorQuantizer,
    FiniteScalarQuantizer,
)


def _run_distributed_ema_update(rank: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        quantizer = EMAVectorQuantizer(
            2, codebook_size=4, decay=0.5, dead_code_steps=0
        )
        initial = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        quantizer.codebook.copy_(initial)
        quantizer.ema_sum.copy_(initial)
        quantizer.ema_count.fill_(1.0)
        quantizer.initialized.fill_(True)
        if rank == 0:
            local_values = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            local_ids = torch.tensor([0, 1])
        else:
            local_values = torch.tensor([[2.0, 2.0], [0.0, 4.0]])
            local_ids = torch.tensor([1, 2])
        quantizer._ema_update(local_values, local_ids)
        torch.save(
            {
                "codebook": quantizer.codebook,
                "ema_count": quantizer.ema_count,
                "ema_sum": quantizer.ema_sum,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_ema_vq_shapes_updates_and_round_trip():
    torch.manual_seed(7)
    quantizer = EMAVectorQuantizer(16, codebook_size=32, decay=0.9, dead_code_steps=0)
    inputs = torch.randn(4, 5, 16, requires_grad=True)
    before = quantizer.codebook.clone()

    quantized, ids, stats = quantizer(inputs)
    (quantized.square().mean() + stats["commitment_loss"]).backward()

    assert quantized.shape == inputs.shape
    assert ids.shape == (4, 5)
    assert inputs.grad is not None
    assert not torch.equal(before, quantizer.codebook)
    assert torch.allclose(quantizer.decode(ids), quantizer.codebook[ids])
    assert 0 < stats["active_codes"] <= 32


def test_ema_vq_distributed_update_keeps_codebooks_identical(tmp_path: Path):
    mp.spawn(
        _run_distributed_ema_update,
        args=(str(tmp_path / "gloo-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    rank_one = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for key in ("codebook", "ema_count", "ema_sum"):
        assert torch.equal(rank_zero[key], rank_one[key])
    assert torch.equal(rank_zero["ema_count"], torch.tensor([1.0, 1.5, 1.0, 0.5]))


def test_fsq_ids_decode_to_the_same_quantized_values():
    torch.manual_seed(11)
    quantizer = FiniteScalarQuantizer(12, levels=(4, 4, 4))
    inputs = torch.randn(3, 6, 12)

    quantized, ids, _ = quantizer(inputs)

    assert ids.min() >= 0
    assert ids.max() < 64
    assert torch.allclose(quantized, quantizer.decode(ids), atol=1e-6)


def test_fsq_continuous_projection_keeps_learned_bottleneck_without_rounding():
    torch.manual_seed(12)
    quantizer = FiniteScalarQuantizer(12, levels=(4, 4, 8))
    inputs = torch.randn(3, 6, 12, requires_grad=True)

    relaxed = quantizer.project_continuous(inputs)
    quantized, _, _ = quantizer(inputs)

    assert relaxed.shape == inputs.shape
    assert not torch.equal(relaxed, quantized)
    relaxed.square().mean().backward()
    assert inputs.grad is not None
    assert quantizer.project_in.weight.grad is not None
    assert quantizer.project_out.weight.grad is not None


def test_action_tokenizer_masks_invalid_dimensions_and_steps():
    torch.manual_seed(13)
    model = ActionChunkTokenizer(
        action_dim=6,
        horizon=16,
        num_tokens=4,
        hidden_dim=32,
        latent_dim=16,
        codebook_size=32,
        dead_code_steps=0,
    )
    actions = torch.randn(2, 16, 6)
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[0, 12:] = False
    dimensions = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    angular = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)

    output = model.loss(
        actions,
        action_valid_mask=valid,
        action_dim_mask=dimensions,
        angular_dim_mask=angular,
        gripper_dims=(4, 5),
    )
    output["loss"].backward()

    assert output["reconstruction"].shape == actions.shape
    assert output["token_ids"].shape == (2, 4)
    assert torch.isfinite(output["loss"])
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_action_tokenizer_checkpoint_round_trip_and_decode():
    torch.manual_seed(17)
    first = ActionChunkTokenizer(
        action_dim=14,
        horizon=32,
        num_tokens=8,
        hidden_dim=32,
        latent_dim=16,
        quantizer="fsq",
        fsq_levels=(4, 4, 4),
    ).eval()
    actions = torch.randn(2, 32, 14)
    first_output = first(actions)

    second = ActionChunkTokenizer(
        action_dim=14,
        horizon=32,
        num_tokens=8,
        hidden_dim=32,
        latent_dim=16,
        quantizer="fsq",
        fsq_levels=(4, 4, 4),
    ).eval()
    second.load_state_dict(first.state_dict())
    second_output = second(actions)

    assert torch.equal(first_output["token_ids"], second_output["token_ids"])
    assert torch.allclose(first_output["reconstruction"], second_output["reconstruction"])
    assert torch.allclose(
        second.decode_tokens(second_output["token_ids"]),
        second_output["reconstruction"],
        atol=1e-6,
    )


def test_gripper_reconstruction_uses_sigmoid_logits():
    torch.manual_seed(19)
    model = ActionChunkTokenizer(
        action_dim=4,
        horizon=8,
        num_tokens=2,
        hidden_dim=16,
        latent_dim=8,
        codebook_size=16,
        dead_code_steps=0,
    ).eval()
    output = model(torch.randn(2, 8, 4), gripper_dims=(3,))
    assert torch.all(output["reconstruction"][..., 3] >= 0)
    assert torch.all(output["reconstruction"][..., 3] <= 1)
    assert torch.allclose(
        output["reconstruction"][..., 3],
        output["decoder_output"][..., 3].sigmoid(),
    )
