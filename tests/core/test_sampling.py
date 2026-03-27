from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import deep_mkv_gen.core.runtime_utils as runtime_utils

from deep_mkv_gen.core import (
    DeepMKVGen,
    DeepMKVGenConfig,
    SourceSampler,
    sample,
    sample_with_snapshots,
    transport,
    transport_with_snapshots,
)

from tests.helpers import indexed_empirical_sampler, make_source_target_samples


def test_sample_preserves_model_mode_and_seeded_rng() -> None:
    source, _ = make_source_target_samples(seed=8)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=SourceSampler(source),
    )
    model.train(True)

    state_before = torch.get_rng_state().clone()
    out = sample(model, num_samples=8, batch_size=4, seed=123)
    state_after = torch.get_rng_state().clone()

    assert out.shape == (8, 2)
    assert model.training is True
    assert torch.equal(state_before, state_after)


def test_sample_with_snapshots_returns_expected_shapes_and_indices() -> None:
    source, _ = make_source_target_samples(seed=9)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=SourceSampler(source),
    )
    model.train(True)

    snaps, idx = sample_with_snapshots(
        model,
        num_samples=6,
        snapshot_steps=[0, model.N],
        batch_size=4,
        return_source_indices=True,
        seed=77,
    )

    assert set(snaps) == {0, model.N}
    assert snaps[0].shape == (6, 2)
    assert snaps[model.N].shape == (6, 2)
    assert idx.shape == (6,)
    assert torch.all(idx >= 0)
    assert model.training is True


def test_transport_preserves_model_mode_and_supports_single_input() -> None:
    source, _ = make_source_target_samples(seed=10)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=SourceSampler(source),
    )
    model.train(True)

    x0 = torch.tensor([1.0, -1.0])
    out = transport(model, x0, device="cpu", seed=11)

    assert out.shape == (2,)
    assert model.training is True


def test_transport_with_snapshots_returns_requested_steps() -> None:
    source, _ = make_source_target_samples(seed=11)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=SourceSampler(source),
    )
    model.train(True)

    x0 = torch.zeros(3, 2)
    snaps = transport_with_snapshots(model, x0, snapshot_steps=[0, model.N], device="cpu", seed=12)

    assert set(snaps) == {0, model.N}
    assert snaps[0].shape == (3, 2)
    assert snaps[model.N].shape == (3, 2)
    assert model.training is True


def test_sample_accepts_custom_callable_source_sampler() -> None:
    source, _ = make_source_target_samples(seed=12)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=indexed_empirical_sampler(source),
    )

    out = sample(model, num_samples=4, batch_size=2, seed=13)

    assert out.shape == (4, 2)


def test_sample_validates_num_samples_and_batch_size() -> None:
    source, _ = make_source_target_samples(seed=13)
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
        source_sampler=SourceSampler(source),
    )

    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        sample(model, num_samples=0)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        sample(model, num_samples=2, batch_size=0)


def test_inference_helpers_restore_model_device(monkeypatch) -> None:
    class TrackingInferenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.training = True
            self.current_device = torch.device("cuda:0")
            self.to_calls: list[torch.device] = []
            self.N = 4

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def eval(self):
            self.training = False
            return self

        def to(self, device):
            self.current_device = torch.device(device)
            self.to_calls.append(self.current_device)
            return self

        def _sample_source(self, batch_size: int, device=None):
            return torch.zeros(batch_size, 2, device=device)

        def _simulate(self, x0: torch.Tensor, antithetic: bool = False):
            del antithetic
            return x0 + 1.0, torch.zeros_like(x0), {}

        def _rollout(self, x0: torch.Tensor, antithetic: bool = False, snapshot_steps=None, collect_metrics=False):
            del antithetic, collect_metrics
            steps = snapshot_steps or set()
            snaps = {step: x0 + float(step) for step in steps}
            return x0, torch.zeros_like(x0), {}, snaps

    monkeypatch.setattr(runtime_utils, "_get_model_device", lambda model: model.current_device)
    model = TrackingInferenceModel()
    model.train(True)

    generated = sample(model, num_samples=3, batch_size=2, device="cpu", seed=5)
    transported = transport(model, torch.zeros(2), device="cpu", seed=6)

    assert generated.shape == (3, 2)
    assert transported.shape == (2,)
    assert model.current_device == torch.device("cuda:0")
    assert model.training is True
    assert model.to_calls[0] == torch.device("cpu")
    assert model.to_calls[1] == torch.device("cuda:0")
    assert model.to_calls[2] == torch.device("cpu")
    assert model.to_calls[3] == torch.device("cuda:0")
