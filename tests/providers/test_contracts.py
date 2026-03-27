from __future__ import annotations

import math

import pytest
import torch

from deep_mkv_gen.core import TrainConfig
from deep_mkv_gen.providers import (
    DSMScoreConfig,
    DSMScoreModel,
    KLRatioScoreConfig,
    KLScoreDifferenceConfig,
)
from deep_mkv_gen.providers.kl.provider_kl import make_kl_provider
from deep_mkv_gen.providers.kl.ratio_score import RatioScoreKLTerminalGradProvider

from tests.helpers import make_source_target_samples, make_target_sampler


def _tiny_train_cfg() -> TrainConfig:
    return TrainConfig(num_steps=2, batch_size=8, k=1.0, lr=1e-3, weight_decay=0.0, grad_clip=1.0, log_every=10)


def _tiny_dsm_score_model() -> DSMScoreModel:
    return DSMScoreModel(
        dim=2,
        cfg=DSMScoreConfig(
            hidden_dim=8,
            num_layers=1,
            lr=1e-3,
            sigma_noise=0.1,
            calibration_epochs=1,
            batch_size=8,
            log_every_epochs=1,
        ),
    )


@pytest.mark.parametrize(
    ("name", "builder"),
    [
        (
            "kl_ratio_score",
            lambda: make_kl_provider(
                cfg=KLRatioScoreConfig(
                    hidden_dim=8,
                    num_layers=1,
                    classifier_updates_per_step=1,
                    classifier_lr=1e-3,
                    use_replay=False,
                    replay_batch_size=8,
                    target_grad_norm_clip=1.0,
                ),
                dim=2,
            ),
        ),
        (
            "kl_score_difference",
            lambda: make_kl_provider(
                cfg=KLScoreDifferenceConfig(
                    score_mu_updates_per_step=1,
                    target_grad_norm_clip=1.0,
                ),
                target_score_model=_tiny_dsm_score_model(),
            ),
        ),
    ],
)
def test_provider_contracts(name, builder) -> None:
    _, target = make_source_target_samples(seed=21)
    sampler = make_target_sampler(target)
    provider = builder()
    provider.set_target_sampler(sampler)
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())

    xT = torch.randn(8, 2)
    out1 = provider.get_grad_target(xT=xT, step=0)
    out2 = provider.get_grad_target(xT=xT, step=1)

    assert out1.grad_target.shape == xT.shape
    assert out2.grad_target.shape == xT.shape
    assert torch.isfinite(out1.grad_target).all()
    assert torch.isfinite(out2.grad_target).all()
    assert all(math.isfinite(float(v)) for v in out1.metrics.values())
    assert all(math.isfinite(float(v)) for v in out2.metrics.values())


def test_kl_ratio_score_replay_warmup_returns_zero_until_ready() -> None:
    _, target = make_source_target_samples(seed=41)
    sampler = make_target_sampler(target)
    provider = RatioScoreKLTerminalGradProvider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=True,
            replay_capacity=16,
            replay_min_size=8,
            replay_batch_size=4,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())

    out_warm = provider.get_grad_target(xT=torch.randn(4, 2), step=0)
    assert torch.allclose(out_warm.grad_target, torch.zeros_like(out_warm.grad_target))
    assert out_warm.metrics["ratio_score_ready"] == 0.0
    assert out_warm.metrics["replay_ready"] == 0.0
    assert out_warm.metrics["classifier_updates_applied"] == 0.0
    assert out_warm.metrics["replay_size"] == 0.0
    assert out_warm.metrics["replay_size_after"] == 4.0

    out_still_warm = provider.get_grad_target(xT=torch.randn(4, 2), step=1)
    assert out_still_warm.metrics["ratio_score_ready"] == 0.0
    assert out_still_warm.metrics["replay_ready"] == 0.0
    assert out_still_warm.metrics["classifier_updates_applied"] == 0.0
    assert out_still_warm.metrics["replay_size"] == 4.0
    assert out_still_warm.metrics["replay_size_after"] == 8.0

    out_ready = provider.get_grad_target(xT=torch.randn(4, 2), step=2)
    assert out_ready.metrics["ratio_score_ready"] == 1.0
    assert out_ready.metrics["replay_ready"] == 1.0
    assert out_ready.metrics["classifier_updates_applied"] == 1.0
    assert out_ready.metrics["replay_size"] == 8.0
    assert out_ready.metrics["replay_size_after"] == 12.0
    assert not torch.allclose(out_ready.grad_target, torch.zeros_like(out_ready.grad_target))


def test_kl_ratio_score_replay_state_resets_on_train_start() -> None:
    _, target = make_source_target_samples(seed=42)
    sampler = make_target_sampler(target)
    provider = RatioScoreKLTerminalGradProvider(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=True,
            replay_capacity=8,
            replay_min_size=6,
            replay_batch_size=4,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    provider.get_grad_target(xT=torch.randn(4, 2), step=0)
    provider.get_grad_target(xT=torch.randn(4, 2), step=1)

    assert provider._replay_size() == 8
    assert provider._replay_full is True

    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    assert provider._replay_size() == 0
    assert provider._replay_pos == 0
    assert provider._replay_full is False
    assert provider._classifier_ready is False

    out = provider.get_grad_target(xT=torch.randn(4, 2), step=0)
    assert out.metrics["replay_size"] == 0.0
    assert out.metrics["replay_size_after"] == 4.0
    assert out.metrics["ratio_score_ready"] == 0.0
    assert torch.allclose(out.grad_target, torch.zeros_like(out.grad_target))


def test_kl_ratio_score_config_rejects_replay_min_size_above_capacity() -> None:
    cfg = KLRatioScoreConfig(replay_capacity=8, replay_min_size=9)

    with pytest.raises(ValueError, match="replay_min_size cannot exceed replay_capacity"):
        cfg.validate()
