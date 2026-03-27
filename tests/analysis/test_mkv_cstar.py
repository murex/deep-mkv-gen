from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from deep_mkv_gen.analysis import mkv_cstar


def test_analysis_package_exports_modules() -> None:
    import deep_mkv_gen.analysis as analysis

    assert analysis.__all__ == ["mkv_cstar", "mkv_external_kl"]


def test_log_normal_diag_and_sample_gamma() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    mean = torch.zeros(2)
    var = torch.ones(2)
    out = mkv_cstar.log_normal_diag(x, mean, var)
    expected0 = -0.5 * 2.0 * math.log(2.0 * math.pi)
    assert out.shape == (2,)
    assert out[0].item() == pytest.approx(expected0)

    rng = np.random.default_rng(7)
    gamma = mkv_cstar.sample_gamma(5, np.array([1.0, -1.0]), 0.5, rng)
    assert gamma.shape == (5, 2)


def test_estimate_kl_classifier_and_cstar_classifier_return_finite_outputs() -> None:
    rng = np.random.default_rng(11)
    target = rng.normal(loc=1.0, scale=0.5, size=(256, 2)).astype(np.float32)
    res = mkv_cstar.estimate_kl_classifier(
        target=target,
        gamma_mean=np.zeros(2, dtype=np.float32),
        gamma_std=1.0,
        seed=3,
        device="cpu",
        hidden_dim=16,
        num_layers=1,
        steps=5,
        batch_size=32,
        eval_fraction=0.25,
        logit_clip=10.0,
    )

    assert res.method == "classifier"
    assert np.isfinite(res.kl_hat)
    assert math.isnan(res.c_star_hat)
    assert res.n_target_used == 256
    assert res.n_baseline_used == 256
    assert res.details["n_eval"] >= 128.0
    assert 0.0 <= res.details["heldout_acc"] <= 1.0

    cstar = mkv_cstar.estimate_cstar_classifier(
        target=target,
        gamma_mean=np.zeros(2, dtype=np.float32),
        gamma_std=1.0,
        sigma=0.5,
        seed=3,
        device="cpu",
        hidden_dim=16,
        num_layers=1,
        steps=5,
        batch_size=32,
        eval_fraction=0.25,
        logit_clip=10.0,
    )
    assert cstar.c_star_hat == pytest.approx((0.5**2) * cstar.kl_hat)


def test_kth_neighbor_distance_and_knn_estimators_validate_inputs() -> None:
    query = torch.zeros(2, 2)
    ref = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="same size"):
        mkv_cstar._kth_neighbor_distance(query, ref, k=1, chunk_size=2, exclude_self=True)

    target = np.array([[0.0, 0.0]], dtype=np.float32)
    baseline = np.array([[1.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="n >= 2"):
        mkv_cstar.estimate_kl_knn(target=target, baseline=baseline, k=1)

    target = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    baseline = np.array([[1.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="k must be >= 1"):
        mkv_cstar.estimate_kl_knn(target=target, baseline=baseline, k=0)
    with pytest.raises(ValueError, match="k must be < n_target_used"):
        mkv_cstar.estimate_kl_knn(target=target, baseline=baseline, k=3)
    with pytest.raises(ValueError, match="k must be <= n_baseline_used"):
        mkv_cstar.estimate_kl_knn(target=target, baseline=baseline, k=2, chunk_size=1)

    rng = np.random.default_rng(12)
    target = rng.normal(size=(32, 2)).astype(np.float32)
    baseline = rng.normal(loc=1.0, size=(32, 2)).astype(np.float32)
    res = mkv_cstar.estimate_kl_knn(target=target, baseline=baseline, k=3, chunk_size=8)
    assert res.method == "knn"
    assert np.isfinite(res.kl_hat)
    assert res.details["k"] == pytest.approx(3.0)

    cstar = mkv_cstar.estimate_cstar_knn(target=target, baseline=baseline, sigma=0.4, k=3, chunk_size=8)
    assert cstar.c_star_hat == pytest.approx((0.4**2) * cstar.kl_hat)


def test_estimate_cstar_mc_validates_inputs_and_runs() -> None:
    mean = torch.tensor([1.0, -1.0])
    std = torch.tensor([2.0, 0.5])

    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        mkv_cstar.estimate_cstar_mc(
            mean=mean,
            std=std,
            sigma=0.3,
            T=1.0,
            n_samples=0,
            batch_size=2,
            seed=1,
            device="cpu",
            sample_target_fn=lambda n, device=None: torch.zeros(n, 2, device=device),
            log_target_prob_fn=lambda x: torch.zeros(x.shape[0], device=x.device),
        )

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        mkv_cstar.estimate_cstar_mc(
            mean=mean,
            std=std,
            sigma=0.3,
            T=1.0,
            n_samples=2,
            batch_size=0,
            seed=1,
            device="cpu",
            sample_target_fn=lambda n, device=None: torch.zeros(n, 2, device=device),
            log_target_prob_fn=lambda x: torch.zeros(x.shape[0], device=x.device),
        )

    with pytest.raises(ValueError, match="log_target_prob_fn must return a 1D tensor"):
        mkv_cstar.estimate_cstar_mc(
            mean=mean,
            std=std,
            sigma=0.3,
            T=1.0,
            n_samples=2,
            batch_size=2,
            seed=1,
            device="cpu",
            sample_target_fn=lambda n, device=None: torch.zeros(n, 2, device=device),
            log_target_prob_fn=lambda x: torch.zeros(x.shape[0], 1, device=x.device),
        )

    out = mkv_cstar.estimate_cstar_mc(
        mean=mean,
        std=std,
        sigma=0.3,
        T=1.0,
        n_samples=5,
        batch_size=2,
        seed=1,
        device="cpu",
        sample_target_fn=lambda n, device=None: torch.full((n, 2), 0.5, device=device),
        log_target_prob_fn=lambda x: -0.5 * x.pow(2).sum(dim=-1),
    )
    assert np.isfinite(out["kl_rho_gamma"])
    assert np.isfinite(out["c_star"])
    assert "gamma_mean_x1" in out
    assert "gamma_var_x2" in out
