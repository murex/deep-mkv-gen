from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from deep_mkv_gen.core.runtime_utils import isolated_seed, preserve_model_device, preserve_model_mode

from tests.helpers import numpy_state_equal, seed_all


def test_isolated_seed_restores_torch_numpy_and_python_random_state() -> None:
    seed_all(123)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    with isolated_seed("cpu", 999):
        _ = torch.randn(8)
        _ = np.random.randn(8)
        _ = random.random()

    assert torch.equal(torch_before, torch.get_rng_state())
    assert numpy_state_equal(numpy_before, np.random.get_state())
    assert python_before == random.getstate()


def test_preserve_model_mode_restores_training_flag() -> None:
    model = nn.Linear(2, 2)
    model.train(True)

    with preserve_model_mode(model):
        model.eval()
        assert model.training is False

    assert model.training is True


def test_preserve_model_device_restores_original_device(monkeypatch) -> None:
    class TrackingModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.current_device = torch.device("cuda:0")
            self.to_calls: list[torch.device] = []

        def to(self, device):
            self.current_device = torch.device(device)
            self.to_calls.append(self.current_device)
            return self

    model = TrackingModule()

    import deep_mkv_gen.core.runtime_utils as runtime_utils

    monkeypatch.setattr(runtime_utils, "_get_model_device", lambda module: module.current_device)

    with preserve_model_device(model):
        model.to("cpu")
        assert model.current_device == torch.device("cpu")

    assert model.current_device == torch.device("cuda:0")
    assert model.to_calls == [torch.device("cpu"), torch.device("cuda:0")]
