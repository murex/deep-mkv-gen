from __future__ import annotations

import pytest
import torch

from deep_mkv_gen.providers import (
    DSMScoreConfig,
    KLRatioScoreConfig,
    KLScoreDifferenceConfig,
)
from deep_mkv_gen.providers.kl.provider_kl import make_kl_provider
from deep_mkv_gen.providers.kl.ratio_score import RatioScoreKLTerminalGradProvider
from deep_mkv_gen.providers.kl.score_difference import ScoreDifferenceKLTerminalGradProvider

from tests.helpers import make_source_target_samples, make_target_sampler


def test_make_kl_provider_requires_target_score_cfg_for_implicit_score_difference() -> None:
    with pytest.raises(ValueError, match="target_score_cfg is required"):
        make_kl_provider(cfg=KLScoreDifferenceConfig(), dim=2)


def test_make_kl_provider_rejects_score_models_for_ratio_score() -> None:
    with pytest.raises(ValueError, match="only valid with KLScoreDifferenceConfig"):
        make_kl_provider(
            cfg=KLRatioScoreConfig(),
            dim=2,
            target_score_model=object(),
        )


def test_make_kl_provider_builds_ratio_score_provider() -> None:
    provider = make_kl_provider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            use_replay=False,
        ),
        dim=2,
    )

    assert isinstance(provider, RatioScoreKLTerminalGradProvider)


def test_make_kl_provider_calibrate_delegates_for_score_difference_only() -> None:
    _, target = make_source_target_samples(seed=92)
    sampler = make_target_sampler(target)

    ratio_provider = make_kl_provider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            use_replay=False,
        ),
        dim=2,
    )
    ratio_provider.set_target_sampler(sampler)
    assert not hasattr(ratio_provider, "calibrate")

    diff_provider = make_kl_provider(
        cfg=KLScoreDifferenceConfig(
            target_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=1,
                batch_size=4,
            ),
            mu_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=0,
                batch_size=4,
            ),
        ),
        dim=2,
    )
    diff_provider.set_target_sampler(sampler)
    info = diff_provider.calibrate(target_sampler=sampler, device="cpu")
    assert info["target_score_computed"] == 1.0


def test_make_kl_provider_implicit_mu_uses_calibrated_target_weights_and_mu_cfg() -> None:
    _, target = make_source_target_samples(seed=93)
    sampler = make_target_sampler(target)
    provider = make_kl_provider(
        cfg=KLScoreDifferenceConfig(
            target_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=1,
                batch_size=4,
                metric_key="target_score_loss",
            ),
            mu_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=7e-4,
                sigma_noise=0.2,
                calibration_epochs=0,
                batch_size=4,
                grad_clip=0.5,
                metric_key="score_mu_loss",
            ),
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)

    assert isinstance(provider, ScoreDifferenceKLTerminalGradProvider)

    provider.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 2, "grad_clip": 1.0})())

    assert provider.mu_score_model.cfg.lr == pytest.approx(7e-4)
    assert provider.mu_score_model.cfg.sigma_noise == pytest.approx(0.2)
    assert provider.mu_score_model.cfg.grad_clip == pytest.approx(0.5)
    assert provider.mu_score_model.cfg.metric_key == "score_mu_loss"
    assert provider.target_score_model.cfg.metric_key == "target_score_loss"

    target_state = provider.target_score_model.state_dict()
    mu_state = provider.mu_score_model.state_dict()
    assert target_state.keys() == mu_state.keys()
    for key in target_state:
        assert torch.allclose(target_state[key], mu_state[key])

    out = provider.get_grad_target(xT=torch.randn(4, 2), step=0)
    assert out.grad_target.shape == (4, 2)


def test_make_kl_provider_rejects_implicit_mu_cfg_architecture_mismatch() -> None:
    _, target = make_source_target_samples(seed=94)
    sampler = make_target_sampler(target)
    provider = make_kl_provider(
        cfg=KLScoreDifferenceConfig(
            target_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=1,
                batch_size=4,
            ),
            mu_score_cfg=DSMScoreConfig(
                hidden_dim=16,
                num_layers=2,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=0,
                batch_size=4,
            ),
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)

    with pytest.raises(ValueError, match="must match target_score_cfg architecture"):
        provider.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 2, "grad_clip": 1.0})())


def test_make_kl_provider_fresh_mu_init_builds_independent_mu_model() -> None:
    _, target = make_source_target_samples(seed=95)
    sampler = make_target_sampler(target)
    provider = make_kl_provider(
        cfg=KLScoreDifferenceConfig(
            mu_init="fresh",
            target_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=1e-3,
                sigma_noise=0.1,
                calibration_epochs=1,
                batch_size=4,
                metric_key="target_score_loss",
            ),
            mu_score_cfg=DSMScoreConfig(
                hidden_dim=8,
                num_layers=1,
                lr=7e-4,
                sigma_noise=0.2,
                calibration_epochs=0,
                batch_size=4,
                grad_clip=0.5,
                metric_key="score_mu_loss",
            ),
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)

    provider.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 2, "grad_clip": 1.0})())

    assert provider.mu_score_model.cfg.lr == pytest.approx(7e-4)
    assert provider.mu_score_model.cfg.sigma_noise == pytest.approx(0.2)
    assert provider.mu_score_model.cfg.grad_clip == pytest.approx(0.5)
    assert provider.mu_score_model.cfg.metric_key == "score_mu_loss"

    target_state = provider.target_score_model.state_dict()
    mu_state = provider.mu_score_model.state_dict()
    assert target_state.keys() == mu_state.keys()
    assert any(not torch.allclose(target_state[key], mu_state[key]) for key in target_state)
