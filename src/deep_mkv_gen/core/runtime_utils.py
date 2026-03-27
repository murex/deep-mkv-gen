"""Small runtime helpers for RNG, model mode, and device isolation."""

from __future__ import annotations

import contextlib
import random
from typing import Iterator, Union

import numpy as np
import torch
import torch.nn as nn


@contextlib.contextmanager
def preserve_rng_state(device: Union[str, torch.device] = "cpu") -> Iterator[None]:
    """Restore Python, NumPy, and torch RNG state on exit."""

    cpu_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def set_global_seed(seed: int, device: Union[str, torch.device] = "cpu") -> None:
    """Seed Python, NumPy, and torch for the current process."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    random.seed(int(seed))


@contextlib.contextmanager
def isolated_seed(device: Union[str, torch.device], seed: int) -> Iterator[None]:
    """Run a block with a temporary deterministic RNG state."""

    with preserve_rng_state(device):
        set_global_seed(int(seed), device)
        yield


@contextlib.contextmanager
def preserve_model_mode(model: nn.Module) -> Iterator[None]:
    """Restore the model training/eval mode on exit."""

    was_training = bool(model.training)
    try:
        yield
    finally:
        model.train(was_training)


def _get_model_device(model: nn.Module) -> torch.device | None:
    """Return the primary device for a single-device module, if available."""

    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return None


@contextlib.contextmanager
def preserve_model_device(model: nn.Module) -> Iterator[None]:
    """Restore the model device on exit for single-device modules."""

    original_device = _get_model_device(model)
    try:
        yield
    finally:
        if original_device is None:
            return
        current_device = _get_model_device(model)
        if current_device != original_device:
            model.to(original_device)
