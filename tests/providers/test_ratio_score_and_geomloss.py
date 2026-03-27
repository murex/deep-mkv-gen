from __future__ import annotations

import importlib.util
import warnings

import pytest
import torch

from deep_mkv_gen.core import TrainConfig
from deep_mkv_gen.providers import GeomLossW2ProviderConfig
from deep_mkv_gen.providers.kl.ratio_score import (
    KLRatioScoreConfig,
    RatioScoreClassifier,
    RatioScoreKLTerminalGradProvider,
)
from deep_mkv_gen.providers.provider_w2_geomloss import GeomLossW2TerminalGradProvider

from tests.helpers import make_source_target_samples, make_target_sampler


def _tiny_train_cfg() -> TrainConfig:
    return TrainConfig(num_steps=3, batch_size=6, k=1.0, lr=1e-3, weight_decay=0.0, grad_clip=1.0, log_every=10)


@pytest.mark.parametrize(
    ("cfg", "pattern"),
    [
        (KLRatioScoreConfig(hidden_dim=0), "hidden_dim"),
        (KLRatioScoreConfig(num_layers=0), "num_layers"),
        (KLRatioScoreConfig(classifier_lr=0.0), "classifier_lr"),
        (KLRatioScoreConfig(classifier_updates_per_step=0), "classifier_updates_per_step"),
        (KLRatioScoreConfig(weight_decay=-1.0), "weight_decay"),
        (KLRatioScoreConfig(grad_clip=-1.0), "grad_clip"),
        (KLRatioScoreConfig(replay_capacity=0), "replay_capacity"),
        (KLRatioScoreConfig(replay_min_size=0), "replay_min_size"),
        (KLRatioScoreConfig(replay_batch_size=0), "replay_batch_size"),
        (KLRatioScoreConfig(classifier_batch_size=0), "classifier_batch_size"),
        (KLRatioScoreConfig(replay_recent_frac=-0.1), "replay_recent_frac"),
        (KLRatioScoreConfig(replay_mix_current=1.1), "replay_mix_current"),
        (KLRatioScoreConfig(uncertainty_is_beta=1.1), "uncertainty_is_beta"),
        (KLRatioScoreConfig(target_grad_norm_clip=0.0), "target_grad_norm_clip"),
    ],
)
def test_ratio_score_config_validation(cfg: KLRatioScoreConfig, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        cfg.validate()


def test_ratio_score_classifier_supports_spectral_norm() -> None:
    clf = RatioScoreClassifier(dim=2, hidden_dim=8, num_layers=1, spectral_norm=True)
    first_linear = clf.net[0]
    last_linear = clf.net[-1]
    assert hasattr(first_linear, "weight_u")
    assert hasattr(last_linear, "weight_u")
    out = clf(torch.randn(4, 2))
    assert out.shape == (4,)


def test_ratio_score_sampling_helpers_cover_recent_and_uncertainty_branches() -> None:
    _, target = make_source_target_samples(seed=121)
    provider = RatioScoreKLTerminalGradProvider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            use_replay=True,
            replay_capacity=4,
            replay_min_size=1,
            replay_batch_size=2,
            replay_recent_frac=0.5,
            uncertainty_is_beta=1.0,
        ),
        dim=2,
    )
    provider.set_target_sampler(make_target_sampler(target))
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())

    with pytest.raises(RuntimeError, match="Replay is empty"):
        provider._sample_replay(1)

    provider._add_to_replay(torch.arange(12.0).reshape(6, 2))
    assert provider._replay_full is True
    assert provider._replay_size() == 4
    assert provider._sample_replay(2).shape == (2, 2)

    xT = torch.randn(3, 2)
    assert provider._sample_current_mu(xT_det=xT[:0], n=2).shape == (0, 2)
    assert torch.allclose(provider._sample_current_mu(xT_det=xT, n=3), xT)

    with torch.no_grad():
        for param in provider.classifier.parameters():
            param.zero_()
    uncertain = provider._sample_current_mu(xT_det=xT, n=2)
    assert uncertain.shape == (2, 2)

    def saturating_logits(x):
        return torch.full((x.shape[0],), 1000.0, device=x.device)

    provider.classifier.forward = saturating_logits  # type: ignore[method-assign]
    confident = provider._sample_current_mu(xT_det=xT, n=2)
    assert confident.shape == (2, 2)


def test_ratio_score_get_grad_target_without_replay_and_without_target_clip() -> None:
    _, target = make_source_target_samples(seed=122)
    provider = RatioScoreKLTerminalGradProvider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=False,
            uncertainty_is_beta=0.0,
            target_grad_norm_clip=None,
        ),
        dim=2,
    )
    provider.set_target_sampler(make_target_sampler(target))
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    out = provider.get_grad_target(xT=torch.randn(4, 2), step=0)

    assert out.grad_target.shape == (4, 2)
    assert out.metrics["classifier_updates_applied"] == 1.0
    assert out.metrics["ratio_score_ready"] == 1.0
    assert out.metrics["target_grad_clip_frac"] == 0.0
    assert out.metrics["target_grad_scale_mean"] == 1.0


def test_ratio_score_replay_mix_current_and_clip_metrics() -> None:
    _, target = make_source_target_samples(seed=123)
    provider = RatioScoreKLTerminalGradProvider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=True,
            replay_capacity=8,
            replay_min_size=2,
            replay_batch_size=4,
            replay_mix_current=0.5,
            replay_recent_frac=0.5,
            target_grad_norm_clip=0.1,
        ),
        dim=2,
    )
    provider.set_target_sampler(make_target_sampler(target))
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    provider.get_grad_target(xT=torch.randn(2, 2), step=0)
    out = provider.get_grad_target(xT=torch.randn(2, 2), step=1)

    assert out.metrics["classifier_updates_applied"] == 1.0
    assert out.metrics["replay_ready"] == 1.0
    assert out.metrics["replay_size"] >= 2.0
    assert out.metrics["replay_size_after"] >= out.metrics["replay_size"]
    assert "classifier_grad_preclip_norm_mean" in out.metrics
    assert "classifier_grad_clip_frac" in out.metrics
    assert out.metrics["provider_step"] == 1.0

@pytest.mark.parametrize(
    ("cfg", "pattern"),
    [
        (GeomLossW2ProviderConfig(blur=0.0), "blur"),
        (GeomLossW2ProviderConfig(scaling=1.0), "scaling"),
        (GeomLossW2ProviderConfig(backend="bad"), "backend"),
        (GeomLossW2ProviderConfig(diameter=0.0), "diameter"),
        (GeomLossW2ProviderConfig(reach=0.0), "reach"),
        (GeomLossW2ProviderConfig(truncate=0.0), "truncate"),
        (GeomLossW2ProviderConfig(cluster_scale=0.0), "cluster_scale"),
        (GeomLossW2ProviderConfig(target_batch_size=0), "target_batch_size"),
        (GeomLossW2ProviderConfig(target_draws_per_step=0), "target_draws_per_step"),
        (GeomLossW2ProviderConfig(target_grad_norm_clip=0.0), "target_grad_norm_clip"),
    ],
)
def test_geomloss_w2_config_validation_matrix(cfg: GeomLossW2ProviderConfig, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        cfg.validate()


@pytest.mark.skipif(importlib.util.find_spec("geomloss") is None, reason="geomloss not installed")
def test_geomloss_provider_internal_branches(monkeypatch) -> None:
    _, target = make_source_target_samples(seed=124, dim=4)
    sampler = make_target_sampler(target)
    provider = GeomLossW2TerminalGradProvider(
        cfg=GeomLossW2ProviderConfig(
            blur=0.5,
            scaling=0.9,
            backend="multiscale",
            target_draws_per_step=2,
            target_grad_norm_clip=None,
        ),
        dim=4,
    )
    provider.set_target_sampler(sampler)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    assert provider._resolved_backend == "online"
    assert any("Falling back" in str(w.message) for w in caught)

    out = provider.get_grad_target(xT=torch.randn(3, 4), step=0)
    assert out.grad_target.shape == (3, 4)
    assert out.metrics["target_grad_clip_frac"] == 0.0
    assert out.metrics["target_grad_scale_mean"] == 1.0

    with pytest.raises(ValueError, match="shape"):
        provider.get_grad_target(xT=torch.randn(3, 2), step=0)


@pytest.mark.skipif(importlib.util.find_spec("geomloss") is None, reason="geomloss not installed")
def test_geomloss_provider_error_paths(monkeypatch) -> None:
    _, target = make_source_target_samples(seed=125)
    sampler = make_target_sampler(target)
    provider = GeomLossW2TerminalGradProvider(
        cfg=GeomLossW2ProviderConfig(blur=0.5, scaling=0.9, backend="tensorized"),
        dim=2,
    )
    provider.set_target_sampler(sampler)

    with pytest.raises(RuntimeError, match="Call on_train_start"):
        provider.get_grad_target(xT=torch.randn(2, 2), step=0)

    import deep_mkv_gen.providers.provider_w2_geomloss as mod

    monkeypatch.setattr(mod, "SamplesLoss", None)
    with pytest.raises(ImportError, match="GeomLoss is not installed"):
        provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())

    monkeypatch.undo()
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())

    bad_sampler = type("BadSampler", (), {"sample": lambda self, n: torch.randn(n, 3)})()
    bad_provider = GeomLossW2TerminalGradProvider(
        cfg=GeomLossW2ProviderConfig(blur=0.5, scaling=0.9, backend="tensorized"),
        dim=2,
    )
    bad_provider.set_target_sampler(bad_sampler)
    bad_provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    with pytest.raises(ValueError, match="must return shape"):
        bad_provider._sample_target(2)

    class BadLoss:
        def __call__(self, x, y):
            del x, y
            return torch.ones(2, 2)

    provider.loss_fn = BadLoss()
    with pytest.raises(RuntimeError, match="Expected a scalar"):
        provider._single_batch_grad_and_metrics(torch.randn(2, 2))
