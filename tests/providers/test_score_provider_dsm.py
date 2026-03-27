from __future__ import annotations

import pytest
import torch

from deep_mkv_gen.providers import DSMScoreConfig, DSMScoreModel

from tests.helpers import make_source_target_samples, make_target_sampler


@pytest.mark.parametrize(
    ("cfg", "pattern"),
    [
        (DSMScoreConfig(hidden_dim=0), "hidden_dim"),
        (DSMScoreConfig(num_layers=0), "num_layers"),
        (DSMScoreConfig(lr=0.0), "lr"),
        (DSMScoreConfig(weight_decay=-1.0), "weight_decay"),
        (DSMScoreConfig(sigma_noise=0.0), "sigma_noise"),
        (DSMScoreConfig(grad_clip=0.0), "grad_clip"),
        (DSMScoreConfig(calibration_epochs=-1), "calibration_epochs"),
        (DSMScoreConfig(batch_size=0), "batch_size"),
        (DSMScoreConfig(log_every_epochs=0), "log_every_epochs"),
    ],
)
def test_dsm_score_config_validation(cfg: DSMScoreConfig, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        cfg.validate()


def _tiny_cfg(**kwargs) -> DSMScoreConfig:
    cfg = DSMScoreConfig(
        hidden_dim=8,
        num_layers=1,
        lr=1e-3,
        sigma_noise=0.1,
        calibration_epochs=1,
        batch_size=4,
        log_every_epochs=1,
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_dsm_load_state_and_train_start_without_trainable_params() -> None:
    model = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=0))
    state = model.state_dict()
    model.load_score_state_dict(state)
    assert model._has_pretrained_score is True
    assert model._sampler_compute_done is False

    model.set_trainable(False)
    model.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"grad_clip": 1.0})())
    assert model._opt is None
    assert model._trainable_params == []

    with pytest.raises(RuntimeError, match="not initialized"):
        model._train_step(torch.zeros(2, 2))


def test_dsm_compute_score_requires_pretrained_or_sampler_and_supports_pretrained_mode() -> None:
    _, target = make_source_target_samples(seed=111)
    sampler = make_target_sampler(target)

    cold = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=0))
    with pytest.raises(ValueError, match="not pretrained"):
        cold.compute_score(x=None, step=0, target_sampler=sampler, device="cpu")

    warm = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=0))
    warm.load_score_state_dict(warm.state_dict())
    out = warm.compute_score(x=None, step=0, target_sampler=sampler, device="cpu")
    assert out.metrics["score_pretrained"] == 1.0

    trainable = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=1))
    with pytest.raises(ValueError, match="needs target_sampler"):
        trainable.compute_score(x=None, step=0, force=True, device="cpu")


def test_dsm_compute_from_sampler_and_iterator_paths() -> None:
    _, target = make_source_target_samples(seed=112)
    sampler = make_target_sampler(target)
    model = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=2, batch_size=5, log_every_epochs=2))

    out = model.compute_score(x=None, step=0, target_sampler=sampler, device="cpu")
    assert out.metrics["score_computed"] == 1.0
    assert out.metrics["score_compute_epochs"] == 2.0
    assert "score_compute_last_loss" in out.metrics

    iterator_batches = list(
        model._epoch_batch_iterator(
            sampler=sampler,
            device="cpu",
        )
    )
    assert len(iterator_batches) == 13
    assert all(batch.shape[1] == 2 for batch in iterator_batches)
    assert sum(batch.shape[0] for batch in iterator_batches) == int(target.shape[0])

    class SamplerOnly:
        def __init__(self, n: int) -> None:
            self.N = int(n)

        def sample(self, batch_size: int) -> torch.Tensor:
            return torch.randn(batch_size, 2)

    sampler_only = SamplerOnly(12)
    sampled_batches = list(model._epoch_batch_iterator(sampler=sampler_only, device="cpu"))
    assert len(sampled_batches) == 3
    assert all(batch.shape == (5, 2) for batch in sampled_batches)

    empty_sampler = make_target_sampler(torch.empty(0, 2))
    with pytest.raises(ValueError, match="at least one sample"):
        list(
            model._epoch_batch_iterator(
                sampler=empty_sampler,
                device="cpu",
            )
        )

    empty_sampler_only = SamplerOnly(0)
    with pytest.raises(ValueError, match="sampler.N must be >= 1"):
        list(model._epoch_batch_iterator(sampler=empty_sampler_only, device="cpu"))


def test_dsm_compute_from_sampler_rejects_no_trainable_parameters() -> None:
    _, target = make_source_target_samples(seed=113)
    sampler = make_target_sampler(target)
    model = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=1))
    for param in model.parameters():
        param.requires_grad_(False)
    model.set_trainable = lambda trainable: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="no trainable parameters"):
        model._compute_from_sampler(sampler=sampler, device="cpu")


def test_dsm_compute_score_trains_and_score_matching_loss_validates_sigma() -> None:
    model = DSMScoreModel(dim=2, cfg=_tiny_cfg(calibration_epochs=0, grad_clip=None))
    model.on_train_start(device="cpu", train_cfg=type("Cfg", (), {"grad_clip": 0.5})())
    out = model.compute_score(x=torch.randn(4, 2), step=0)
    assert "score_loss" in out.metrics
    assert out.score is not None and out.score.shape == (4, 2)

    with pytest.raises(ValueError, match="sigma_noise"):
        model.score_matching_loss(torch.zeros(2, 2), sigma_noise=0.0)
