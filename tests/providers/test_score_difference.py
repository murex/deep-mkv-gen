from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deep_mkv_gen.core.protocols import ScoreComputeOutput
from deep_mkv_gen.providers.kl.score_difference import (
    KLScoreDifferenceConfig,
    ScoreDifferenceKLTerminalGradProvider,
)

from tests.helpers import make_source_target_samples, make_target_sampler


class FakeScoreModel(nn.Module):
    def __init__(self, *, score_value: float, calibration_value: float = 1.0, metric_key: str = "score_mu_loss"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(score_value)))
        self.calibration_value = float(calibration_value)
        self.metric_key = metric_key
        self.compute_calls: list[tuple[bool, bool]] = []
        self.on_train_start_calls = 0

    def __deepcopy__(self, memo):
        cloned = type(self)(
            score_value=float(self.scale.detach().item()),
            calibration_value=self.calibration_value,
            metric_key=self.metric_key,
        )
        memo[id(self)] = cloned
        return cloned

    def set_trainable(self, trainable: bool) -> None:
        self.scale.requires_grad_(bool(trainable))

    def on_train_start(self, device, train_cfg) -> None:
        del train_cfg
        self.on_train_start_calls += 1
        self.to(device)

    def compute_score(self, x, step: int, target_sampler=None, device="cpu", force: bool = False):
        del device
        self.compute_calls.append((x is None, force))
        if x is None:
            return ScoreComputeOutput(
                score=None,
                metrics={"score_computed": self.calibration_value, "force": 1.0 if force else 0.0},
            )
        score = torch.ones_like(x) * self.scale
        metrics = {}
        if self.training and self.scale.requires_grad:
            metrics[self.metric_key] = float(step + 1)
            if target_sampler is not None:
                metrics["target_sampler_seen"] = 1.0
        return ScoreComputeOutput(score=score, metrics=metrics)


class NonCopyableScoreModel(FakeScoreModel):
    def __deepcopy__(self, memo):
        raise RuntimeError("copy disabled")


class NoMetricScoreModel(FakeScoreModel):
    def compute_score(self, x, step: int, target_sampler=None, device="cpu", force: bool = False):
        out = super().compute_score(x, step, target_sampler=target_sampler, device=device, force=force)
        if x is not None:
            return ScoreComputeOutput(score=out.score, metrics={})
        return out


@pytest.mark.parametrize(
    ("cfg", "pattern"),
    [
        (KLScoreDifferenceConfig(score_mu_updates_per_step=0), "score_mu_updates_per_step"),
        (KLScoreDifferenceConfig(target_grad_norm_clip=0.0), "target_grad_norm_clip"),
        (KLScoreDifferenceConfig(mu_init="bad"), "mu_init"),
    ],
)
def test_score_difference_config_validation(cfg: KLScoreDifferenceConfig, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        cfg.validate()


def test_score_difference_init_rejects_uncopyable_implicit_mu_model() -> None:
    with pytest.raises(TypeError, match="could not be copied"):
        ScoreDifferenceKLTerminalGradProvider(
            target_score_model=NonCopyableScoreModel(score_value=1.0),
            mu_score_model=None,
            cfg=KLScoreDifferenceConfig(),
        )


def test_score_difference_calibrate_caches_and_force_refreshes() -> None:
    _, target = make_source_target_samples(seed=101)
    sampler = make_target_sampler(target)
    target_score = FakeScoreModel(score_value=1.0, calibration_value=2.0)
    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=target_score,
        mu_score_model=None,
        cfg=KLScoreDifferenceConfig(),
    )
    provider.set_target_sampler(sampler)

    info1 = provider.calibrate(target_sampler=sampler, device="cpu")
    info2 = provider.calibrate(device="cpu")
    info3 = provider.calibrate(device="cpu", force=True)

    assert info1["target_score_computed"] == 2.0
    assert info2 == info1
    assert info3["target_force"] == 1.0
    assert target_score.compute_calls.count((True, False)) == 1
    assert target_score.compute_calls.count((True, True)) == 1


def test_score_difference_recalibrates_after_target_sampler_change() -> None:
    _, target1 = make_source_target_samples(seed=109)
    _, target2 = make_source_target_samples(seed=110)
    sampler1 = make_target_sampler(target1)
    sampler2 = make_target_sampler(target2)
    target_score = FakeScoreModel(score_value=1.0)
    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=target_score,
        mu_score_model=FakeScoreModel(score_value=2.0),
        cfg=KLScoreDifferenceConfig(),
    )

    train_cfg = type("Cfg", (), {"num_steps": 4, "grad_clip": 1.0})()
    provider.set_target_sampler(sampler1)
    provider.on_train_start(device="cpu", train_cfg=train_cfg)
    provider.set_target_sampler(sampler2)
    provider.on_train_start(device="cpu", train_cfg=train_cfg)

    assert target_score.compute_calls.count((True, False)) == 2


def test_score_difference_calibrate_rejects_uncopyable_post_calibration_mu_model() -> None:
    _, target = make_source_target_samples(seed=102)
    sampler = make_target_sampler(target)
    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=FakeScoreModel(score_value=1.0),
        mu_score_model=FakeScoreModel(score_value=2.0),
        cfg=KLScoreDifferenceConfig(),
    )
    provider.set_target_sampler(sampler)
    provider._mu_score_model_explicit = False
    provider.mu_score_model = NonCopyableScoreModel(score_value=3.0)
    provider.target_score_model = NonCopyableScoreModel(score_value=1.0)

    with pytest.raises(TypeError, match="could not be copied after calibration"):
        provider.calibrate(target_sampler=sampler, device="cpu", force=True)


def test_score_difference_fresh_mu_init_requires_dsm_for_implicit_model() -> None:
    with pytest.raises(TypeError, match="only supported for implicit DSM score models"):
        ScoreDifferenceKLTerminalGradProvider(
            target_score_model=FakeScoreModel(score_value=1.0),
            mu_score_model=None,
            cfg=KLScoreDifferenceConfig(mu_init="fresh"),
        )


def test_score_difference_on_train_start_and_current_updates() -> None:
    _, target = make_source_target_samples(seed=105)
    target_score = FakeScoreModel(score_value=1.0)
    mu_score = FakeScoreModel(score_value=3.0)
    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=target_score,
        mu_score_model=mu_score,
        cfg=KLScoreDifferenceConfig(score_mu_updates_per_step=2, target_grad_norm_clip=1.0),
    )
    provider.set_target_sampler(make_target_sampler(target))
    provider._target_score_prepared = True

    train_cfg = type("Cfg", (), {"num_steps": 4, "grad_clip": 1.0})()
    provider.on_train_start(device="cpu", train_cfg=train_cfg)
    assert target_score.on_train_start_calls == 1
    assert mu_score.on_train_start_calls == 1
    assert target_score.training is False
    assert mu_score.training is True

    out = provider.get_grad_target(xT=torch.zeros(3, 2), step=2)
    assert out.grad_target.shape == (3, 2)
    assert out.metrics["score_mu_loss"] == 3.0
    assert out.metrics["score_mu_updates_applied"] == 2.0
    assert out.metrics["target_grad_clip_frac"] == 1.0
    assert out.metrics["provider_step"] == 2.0
    assert float(torch.linalg.norm(out.grad_target, dim=-1).max()) <= 1.0001


def test_score_difference_current_updates_average_metrics_and_fallback() -> None:
    _, target = make_source_target_samples(seed=103)
    sampler = make_target_sampler(target)
    target_score = FakeScoreModel(score_value=1.0)
    mu_score = FakeScoreModel(score_value=4.0, metric_key="unused_metric")
    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=target_score,
        mu_score_model=mu_score,
        cfg=KLScoreDifferenceConfig(
            score_mu_updates_per_step=2,
            target_grad_norm_clip=None,
        ),
    )
    provider.set_target_sampler(sampler)
    provider._target_score_prepared = True
    provider.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 4, "grad_clip": 1.0})())

    out = provider.get_grad_target(xT=torch.zeros(2, 2), step=1)
    assert out.metrics["unused_metric"] == 2.0
    assert out.metrics["score_mu_updates_applied"] == 2.0
    assert out.metrics["target_grad_clip_frac"] == 0.0
    assert out.metrics["target_grad_scale_mean"] == 1.0

    provider_fallback = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=target_score,
        mu_score_model=NoMetricScoreModel(score_value=4.0),
        cfg=KLScoreDifferenceConfig(
            score_mu_updates_per_step=2,
            target_grad_norm_clip=None,
        ),
    )
    provider_fallback.set_target_sampler(sampler)
    provider_fallback._target_score_prepared = True
    provider_fallback.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 4, "grad_clip": 1.0})())

    out_fallback = provider_fallback.get_grad_target(xT=torch.zeros(2, 2), step=1)
    assert out_fallback.metrics["score_mu_loss"] == 0.0
    assert out_fallback.metrics["score_mu_updates_applied"] == 2.0
    assert out_fallback.metrics["target_grad_clip_frac"] == 0.0
    assert out_fallback.metrics["target_grad_scale_mean"] == 1.0


def test_score_difference_get_grad_target_rejects_missing_scores() -> None:
    class NoneScoreModel(FakeScoreModel):
        def compute_score(self, x, step: int, target_sampler=None, device="cpu", force: bool = False):
            out = super().compute_score(x, step, target_sampler=target_sampler, device=device, force=force)
            if x is not None:
                return ScoreComputeOutput(score=None, metrics=out.metrics)
            return out

    provider = ScoreDifferenceKLTerminalGradProvider(
        target_score_model=NoneScoreModel(score_value=1.0),
        mu_score_model=FakeScoreModel(score_value=2.0),
        cfg=KLScoreDifferenceConfig(),
    )
    provider.set_target_sampler(make_target_sampler(make_source_target_samples(seed=104)[1]))
    provider._target_score_prepared = True
    provider.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"num_steps": 2, "grad_clip": 1.0})())

    with pytest.raises(RuntimeError, match="returned None score"):
        provider.get_grad_target(xT=torch.zeros(2, 2), step=0)
