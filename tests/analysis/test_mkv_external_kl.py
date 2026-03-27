from __future__ import annotations

import numpy as np
import pytest
import torch

from deep_mkv_gen.analysis import mkv_external_kl

from tests.helpers import make_source_target_samples


class TinyModel:
    def __init__(self) -> None:
        self.training = True
        self.to_calls: list[torch.device] = []
        self.eval_calls = 0
        self.sample_calls = 0
        self.sim_calls = 0

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def _sample_source(self, batch_size: int, device=None):
        self.sample_calls += 1
        return torch.ones(batch_size, 2, device=device)

    def _simulate(self, x0: torch.Tensor, antithetic: bool = False):
        del antithetic
        self.sim_calls += 1
        return x0 + 2.0, torch.zeros_like(x0), {}


def test_resolve_device_and_to_tensor() -> None:
    assert mkv_external_kl._resolve_device("cpu") == torch.device("cpu")
    assert mkv_external_kl._resolve_device(torch.device("cpu")) == torch.device("cpu")

    arr = np.zeros((2, 2), dtype=np.float64)
    ten = mkv_external_kl._to_tensor(arr, device="cpu", dtype=torch.float32)
    assert isinstance(ten, torch.Tensor)
    assert ten.dtype == torch.float32

    ten2 = mkv_external_kl._to_tensor(torch.ones(2, 2), device="cpu")
    assert isinstance(ten2, torch.Tensor)

    with pytest.raises(TypeError, match="numpy array or torch tensor"):
        mkv_external_kl._to_tensor("bad", device="cpu")  # type: ignore[arg-type]


def test_sample_terminal_uses_primary_sampling_path_and_restores_mode() -> None:
    model = TinyModel()
    out = mkv_external_kl.sample_terminal(model, num_samples=5, batch_size=2, device="cpu", seed=7)
    assert out.shape == (5, 2)
    assert model.training is True


def test_sample_terminal_falls_back_when_core_sampling_fails(monkeypatch) -> None:
    model = TinyModel()

    import deep_mkv_gen.core.sampling as sampling_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sampling_mod, "sample", boom)
    out = mkv_external_kl.sample_terminal(model, num_samples=5, batch_size=2, device="cpu", seed=9)
    assert out.shape == (5, 2)
    assert model.sample_calls == 3
    assert model.sim_calls == 3
    assert model.training is True


def test_draw_target_samples_validates_exclusive_inputs_and_supports_both_paths() -> None:
    _, target = make_source_target_samples(seed=131)
    sampler = type("Sampler", (), {"sample": lambda self, n: target[:n]})()

    with pytest.raises(ValueError, match="exactly one"):
        mkv_external_kl.draw_target_samples(num_samples=4, device="cpu")
    with pytest.raises(ValueError, match="exactly one"):
        mkv_external_kl.draw_target_samples(
            num_samples=4,
            target_sampler=sampler,
            target_sample_fn=lambda n, device=None: target[:n].numpy(),
            device="cpu",
        )

    from_sampler = mkv_external_kl.draw_target_samples(num_samples=4, target_sampler=sampler, device="cpu", seed=3)
    assert from_sampler.shape == (4, 2)

    from_fn = mkv_external_kl.draw_target_samples(
        num_samples=4,
        target_sample_fn=lambda n, device=None: target[:n].numpy(),
        device="cpu",
        seed=3,
    )
    assert from_fn.shape == (4, 2)


def test_evaluate_model_kl_to_target_aggregates_outputs(monkeypatch) -> None:
    model = TinyModel()

    def fake_sample_terminal(*args, **kwargs):
        del args, kwargs
        return torch.ones(6, 2)

    def fake_draw_target_samples(*args, **kwargs):
        del args, kwargs
        return torch.zeros(7, 2)

    class Run:
        def __init__(self, i: int):
            self.kl_hat = 0.5 + i
            self.heldout_acc = 0.8
            self.heldout_bce = 0.4
            self.n_train_per_class = 10
            self.n_eval_per_class = 4
            self.steps = 3
            self.batch_size = 5
            self.logit_mean = 0.1
            self.logit_std = 0.2
            self.near_separable_flag = 0.0

    class Repeated:
        def __init__(self):
            self.kl_mean = 1.2
            self.kl_std = 0.3
            self.kl_se = 0.2
            self.heldout_acc_mean = 0.75
            self.heldout_bce_mean = 0.55
            self.runs = [Run(0), Run(1)]

    monkeypatch.setattr(mkv_external_kl, "sample_terminal", fake_sample_terminal)
    monkeypatch.setattr(mkv_external_kl, "draw_target_samples", fake_draw_target_samples)
    monkeypatch.setattr(mkv_external_kl, "repeat_kl_classifier_from_samples", lambda **kwargs: Repeated())
    monkeypatch.setattr(mkv_external_kl, "estimate_kl_knn_from_samples", lambda **kwargs: {"kl_hat": 0.9, "k": 3.0})

    out = mkv_external_kl.evaluate_model_kl_to_target(
        model,
        num_model_samples=6,
        num_target_samples=7,
        target_sample_fn=lambda n, device=None: np.zeros((n, 2), dtype=np.float32),
        classifier_repeats=2,
        knn_k=3,
        seed=5,
    )

    assert out["num_model_samples"] == 6
    assert out["num_target_samples"] == 7
    assert out["classifier_kl_mean"] == pytest.approx(1.2)
    assert len(out["runs"]) == 2
    assert out["knn"]["kl_hat"] == pytest.approx(0.9)


def test_evaluate_model_kl_to_target_can_skip_knn(monkeypatch) -> None:
    monkeypatch.setattr(mkv_external_kl, "sample_terminal", lambda *args, **kwargs: torch.ones(4, 2))
    monkeypatch.setattr(
        mkv_external_kl,
        "draw_target_samples",
        lambda *args, **kwargs: torch.zeros(4, 2),
    )

    class Repeated:
        kl_mean = 0.7
        kl_std = 0.1
        kl_se = 0.05
        heldout_acc_mean = 0.8
        heldout_bce_mean = 0.3
        runs = []

    monkeypatch.setattr(mkv_external_kl, "repeat_kl_classifier_from_samples", lambda **kwargs: Repeated())

    out = mkv_external_kl.evaluate_model_kl_to_target(
        TinyModel(),
        num_model_samples=4,
        num_target_samples=4,
        target_sample_fn=lambda n, device=None: np.zeros((n, 2), dtype=np.float32),
        knn_k=None,
        seed=2,
    )
    assert "knn" not in out
