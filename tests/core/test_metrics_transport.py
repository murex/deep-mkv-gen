from __future__ import annotations

import torch

from deep_mkv_gen.core.metrics import sliced_w2_sq, wasserstein2


def test_transport_metrics_are_zero_for_identical_clouds() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]], dtype=torch.float32)
    y = x[torch.tensor([2, 0, 1])]
    assert wasserstein2(x, y) == pytest.approx(0.0, abs=1e-6)
    assert sliced_w2_sq(x, y, projections=32, seed=7) == pytest.approx(0.0, abs=1e-6)


def test_wasserstein2_is_symmetric_and_nonnegative() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[2.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    xy = wasserstein2(x, y)
    yx = wasserstein2(y, x)
    assert xy >= 0.0
    assert xy == pytest.approx(yx, rel=1e-6, abs=1e-6)


def test_sliced_w2_is_deterministic_when_seeded() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=torch.float32)
    first = sliced_w2_sq(x, y, projections=64, seed=123)
    second = sliced_w2_sq(x, y, projections=64, seed=123)
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_sliced_w2_does_not_perturb_global_torch_rng() -> None:
    x = torch.randn(16, 2)
    y = torch.randn(16, 2)
    state_before = torch.get_rng_state().clone()
    _ = sliced_w2_sq(x, y, projections=16)
    state_after = torch.get_rng_state().clone()
    assert torch.equal(state_before, state_after)


import pytest
