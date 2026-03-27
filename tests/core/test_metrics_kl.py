from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from deep_mkv_gen.core.metrics import (
    estimate_kl_classifier_from_samples,
    estimate_kl_knn_from_samples,
    repeat_kl_classifier_from_samples,
)
from tests.helpers import numpy_state_equal, seed_all


def test_knn_kl_returns_finite_value_on_shifted_gaussians() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    x = torch.randn(128, 2, generator=gen) + 1.0
    y = torch.randn(128, 2, generator=gen)
    result = estimate_kl_knn_from_samples(x, y, k=3)
    assert np.isfinite(result["kl_hat"])
    assert result["k"] == pytest.approx(3.0)


def test_classifier_kl_runs_and_repeated_summary_is_consistent() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(11)
    x = torch.randn(256, 2, generator=gen) + 0.5
    y = torch.randn(256, 2, generator=gen)

    single = estimate_kl_classifier_from_samples(
        x,
        y,
        seed=5,
        hidden_dim=16,
        num_layers=1,
        steps=10,
        batch_size=32,
        eval_fraction=0.2,
    )
    repeated = repeat_kl_classifier_from_samples(
        x,
        y,
        repeats=2,
        seed=5,
        hidden_dim=16,
        num_layers=1,
        steps=10,
        batch_size=32,
        eval_fraction=0.2,
    )

    assert np.isfinite(single.kl_hat)
    assert np.isfinite(repeated.kl_mean)
    assert repeated.kl_std >= 0.0
    assert len(repeated.runs) == 2


def test_classifier_kl_preserves_global_rng_state() -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(19)
    x = torch.randn(256, 2, generator=gen)
    y = torch.randn(256, 2, generator=gen)

    seed_all(123)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    _ = estimate_kl_classifier_from_samples(
        x,
        y,
        seed=9,
        hidden_dim=8,
        num_layers=1,
        steps=5,
        batch_size=32,
        eval_fraction=0.2,
    )

    assert torch.equal(torch_before, torch.get_rng_state())
    assert numpy_state_equal(numpy_before, np.random.get_state())
    assert python_before == random.getstate()
