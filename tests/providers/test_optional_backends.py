from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("geomloss")

from deep_mkv_gen.core import TrainConfig
from deep_mkv_gen.providers import (
    GeomLossW2ProviderConfig,
    HybridProviderConfig,
    KLRatioScoreConfig,
)
from deep_mkv_gen.providers.provider_hybrid import HybridTerminalGradProvider
from deep_mkv_gen.providers.provider_w2_geomloss import GeomLossW2TerminalGradProvider

from tests.helpers import make_source_target_samples, make_target_sampler


def _tiny_train_cfg() -> TrainConfig:
    return TrainConfig(num_steps=2, batch_size=8, k=1.0, lr=1e-3, weight_decay=0.0, grad_clip=1.0, log_every=10)


def test_geomloss_provider_contract() -> None:
    _, target = make_source_target_samples(seed=31)
    sampler = make_target_sampler(target)
    provider = GeomLossW2TerminalGradProvider(
        cfg=GeomLossW2ProviderConfig(
            blur=0.5,
            scaling=0.9,
            backend="tensorized",
            target_draws_per_step=1,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    out = provider.get_grad_target(xT=torch.randn(8, 2), step=0)

    assert out.grad_target.shape == (8, 2)
    assert torch.isfinite(out.grad_target).all()
    assert all(math.isfinite(float(v)) for v in out.metrics.values())


def test_geomloss_config_validation() -> None:
    with pytest.raises(ValueError, match="target_draws_per_step must be >= 1"):
        GeomLossW2ProviderConfig(target_draws_per_step=0).validate()


def test_hybrid_provider_contract() -> None:
    _, target = make_source_target_samples(seed=32)
    sampler = make_target_sampler(target)
    provider = HybridTerminalGradProvider(
        cfg=HybridProviderConfig(
            kl_weight=1.0,
            w2_weight=1.0,
            kl_cfg=KLRatioScoreConfig(
                hidden_dim=8,
                num_layers=1,
                classifier_updates_per_step=1,
                classifier_lr=1e-3,
                use_replay=False,
                replay_batch_size=8,
                target_grad_norm_clip=1.0,
            ),
            w2_cfg=GeomLossW2ProviderConfig(
                blur=0.5,
                scaling=0.9,
                backend="tensorized",
                target_draws_per_step=1,
                target_grad_norm_clip=1.0,
            ),
        ),
        dim=2,
    )
    provider.set_target_sampler(sampler)
    provider.on_train_start(device="cpu", train_cfg=_tiny_train_cfg())
    out = provider.get_grad_target(xT=torch.randn(8, 2), step=0)

    assert provider.kl_provider.cfg.target_grad_norm_clip is None
    assert provider.w2_provider.cfg.target_grad_norm_clip is None
    assert out.grad_target.shape == (8, 2)
    assert torch.isfinite(out.grad_target).all()
    assert all(math.isfinite(float(v)) for v in out.metrics.values())
