"""Regression test: one scheduler update per optimizer update, not per microbatch."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from starVLA.training import train_starvla


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, _batch):
        return {"action_loss": (self.weight - 2).square()}


class _ToyDeepSpeedModel(_ToyModel):
    def __init__(self, norm=3.5):
        super().__init__()
        self.norm = norm
        self.norm_reads = 0

    def get_global_grad_norm(self):
        self.norm_reads += 1
        return self.norm


class _AccumulatingAccelerator:
    def __init__(self, total_microsteps):
        self.total_microsteps = total_microsteps
        self.microstep = 0
        self.sync_gradients = False
        self.clip_calls = 0

    def accumulate(self, _model):
        self.microstep += 1
        self.sync_gradients = self.microstep % self.total_microsteps == 0
        return nullcontext()

    @staticmethod
    def backward(loss):
        loss.backward()

    def clip_grad_norm_(self, _parameters, _max_norm):
        # Accelerate returns None in DeepSpeed mode; the engine owns clipping.
        self.clip_calls += 1
        return None


class _AccumulatingOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.updates = 0

    def step(self):
        if self.accelerator.sync_gradients:
            self.optimizer.step()
            self.updates += 1

    def zero_grad(self):
        if self.accelerator.sync_gradients:
            self.optimizer.zero_grad()


def test_scheduler_steps_only_on_completed_gradient_accumulation(monkeypatch):
    monkeypatch.setattr(train_starvla.torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    model = _ToyModel()
    accelerator = _AccumulatingAccelerator(total_microsteps=4)
    bare_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = _AccumulatingOptimizer(bare_optimizer, accelerator)
    scheduler = torch.optim.lr_scheduler.LambdaLR(bare_optimizer, lambda step: 1 - step / 10)
    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))

    for _ in range(8):
        trainer._train_step({})

    assert optimizer.updates == 2
    assert scheduler.last_epoch == 2
    assert abs(scheduler.get_last_lr()[0] - 0.08) < 1e-10


def _make_grad_norm_trainer(monkeypatch, norm=3.5):
    monkeypatch.setattr(train_starvla.torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    model = _ToyDeepSpeedModel(norm=norm)
    accelerator = _AccumulatingAccelerator(total_microsteps=4)
    bare_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.model = model
    trainer.optimizer = _AccumulatingOptimizer(bare_optimizer, accelerator)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(bare_optimizer, lambda step: 1 - step / 10)
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=1.0, log_grad_norm=True))
    return trainer


def test_deepspeed_global_grad_norm_only_on_optimizer_boundary(monkeypatch):
    trainer = _make_grad_norm_trainer(monkeypatch, norm=torch.tensor(3.5))
    for _ in range(3):
        assert "grad_norm" not in trainer._train_step({})
    metrics = trainer._train_step({})
    assert metrics["grad_norm"] == 3.5
    assert trainer.model.norm_reads == 1
    assert trainer.accelerator.clip_calls == 1
    assert trainer.optimizer.updates == 1


def test_missing_deepspeed_grad_norm_fails_instead_of_logging_none(monkeypatch):
    trainer = _make_grad_norm_trainer(monkeypatch, norm=None)
    for _ in range(3):
        trainer._train_step({})
    with pytest.raises(RuntimeError, match="global grad norm is unavailable"):
        trainer._train_step({})


def test_grad_norm_is_forwarded_to_wandb(monkeypatch):
    logged = []
    monkeypatch.setattr(train_starvla.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(train_starvla.wandb, "log", lambda values, step: logged.append((values.copy(), step)))
    monkeypatch.setattr(train_starvla.logger, "info", lambda *_args, **_kwargs: None)
    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.completed_steps = 100
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(logging_frequency=100))
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [9e-5])
    trainer.vla_train_dataloader = [None] * 10
    trainer._log_metrics({"grad_norm": 3.5, "action_token_ce_loss": 0.2})
    assert logged == [({"grad_norm": 3.5, "action_token_ce_loss": 0.2,
                        "learning_rate": 9e-5, "epoch": 10.0}, 100)]
