"""Sampling helpers for source/target data and rollout snapshots."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from .runtime_utils import isolated_seed, preserve_model_device, preserve_model_mode

if TYPE_CHECKING:
    from .mkv_gen import DeepMKVGen


class SourceSampler:
    """Empirical sampler over a fixed source sample cloud."""

    def __init__(self, source_samples: torch.Tensor):
        if not torch.is_tensor(source_samples):
            raise TypeError("source_samples must be a torch.Tensor")
        self.source_samples = source_samples
        self.N = int(source_samples.shape[0])

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        idx = torch.randint(0, self.N, (int(batch_size),), device=self.source_samples.device)
        x0 = self.source_samples[idx]
        if device is not None and x0.device != device:
            x0 = x0.to(device)
        return x0

    def sample_with_indices(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.N, (int(batch_size),), device=self.source_samples.device)
        x0 = self.source_samples[idx]
        if device is not None and x0.device != device:
            x0 = x0.to(device)
            idx = idx.to(device)
        return x0, idx.to(torch.long)

    def __call__(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_with_indices(batch_size=batch_size, device=device)


class TargetSampler:
    """Empirical sampler over a fixed target sample cloud."""

    def __init__(self, target_samples: torch.Tensor):
        self.target_samples = target_samples
        self.N = target_samples.shape[0]

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.N, (batch_size,), device=self.target_samples.device)
        return self.target_samples[idx]


def _resolve_transport_device(
    x0: torch.Tensor,
    device: Optional[Union[str, torch.device]],
) -> torch.device:
    if device is not None:
        return torch.device(device)
    return x0.device


def _normalize_transport_input(x0: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(x0):
        raise TypeError("x0 must be a torch.Tensor")
    if x0.ndim == 1:
        return x0.unsqueeze(0), True
    if x0.ndim == 2:
        return x0, False
    raise ValueError("x0 must have shape (dim,) or (batch, dim)")


def _restore_transport_shape(x: torch.Tensor, squeeze_output: bool) -> torch.Tensor:
    if squeeze_output:
        return x.squeeze(0)
    return x


@torch.no_grad()
def transport(
    model: "DeepMKVGen",
    x0: torch.Tensor,
    *,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Transport explicit source states through the learned rollout."""

    x0_batch, squeeze_output = _normalize_transport_input(x0)
    device_t = _resolve_transport_device(x0_batch, device)
    rng_ctx = isolated_seed(device_t, int(seed)) if seed is not None else nullcontext()
    with preserve_model_mode(model), preserve_model_device(model), rng_ctx:
        model.to(device_t)
        model.eval()
        xT, _, _ = model._simulate(x0_batch.to(device_t), antithetic=False)
        return _restore_transport_shape(xT.detach().cpu(), squeeze_output)


@torch.no_grad()
def transport_with_snapshots(
    model: "DeepMKVGen",
    x0: torch.Tensor,
    snapshot_steps: Optional[List[int]] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Transport explicit source states and return selected rollout snapshots."""

    x0_batch, squeeze_output = _normalize_transport_input(x0)
    device_t = _resolve_transport_device(x0_batch, device)
    rng_ctx = isolated_seed(device_t, int(seed)) if seed is not None else nullcontext()
    with preserve_model_mode(model), preserve_model_device(model), rng_ctx:
        model.to(device_t)
        model.eval()
        if snapshot_steps is None:
            requested = [0, model.N]
        else:
            requested = sorted(set(int(s) for s in snapshot_steps))
            if len(requested) == 0:
                raise ValueError("snapshot_steps must be non-empty when provided")
        for s in requested:
            if s < 0 or s > model.N:
                raise ValueError(f"snapshot step {s} out of range [0, {model.N}]")
        _, _, _, snaps = model._rollout(
            x0_batch.to(device_t),
            antithetic=False,
            snapshot_steps=set(requested),
            collect_metrics=False,
        )
        return {step: _restore_transport_shape(value.detach().cpu(), squeeze_output) for step, value in snaps.items()}


@torch.no_grad()
def sample(
    model: "DeepMKVGen",
    num_samples: int,
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample terminal states from the model in batches."""

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rng_ctx = isolated_seed(device, int(seed)) if seed is not None else nullcontext()
    with preserve_model_mode(model), preserve_model_device(model), rng_ctx:
        model.to(device)
        model.eval()
        out = []
        n = 0
        while n < num_samples:
            b = min(batch_size, num_samples - n)
            x0 = model._sample_source(b, device=torch.device(device))
            xT, _, _ = model._simulate(x0, antithetic=False)
            out.append(xT.detach().cpu())
            n += b
        return torch.cat(out, dim=0)


@torch.no_grad()
def sample_with_snapshots(
    model: "DeepMKVGen",
    num_samples: int,
    snapshot_steps: List[int],
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
    return_source_indices: bool = False,
    seed: Optional[int] = None,
) -> Union[Dict[int, torch.Tensor], Tuple[Dict[int, torch.Tensor], torch.Tensor]]:
    """Sample trajectories and collect requested rollout snapshots."""

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(snapshot_steps) == 0:
        raise ValueError("snapshot_steps must be non-empty")

    steps = sorted(set(int(s) for s in snapshot_steps))
    rng_ctx = isolated_seed(device, int(seed)) if seed is not None else nullcontext()
    with preserve_model_mode(model), preserve_model_device(model), rng_ctx:
        model.to(device)
        model.eval()

        buckets: Dict[int, List[torch.Tensor]] = {s: [] for s in steps}
        source_idx_chunks: List[torch.Tensor] = []
        n = 0
        while n < num_samples:
            b = min(batch_size, num_samples - n)
            x0, source_idx = model._sample_source_with_indices(b, device=torch.device(device))
            snaps = transport_with_snapshots(model, x0, snapshot_steps=steps, device=torch.device(device))
            for s in steps:
                buckets[s].append(snaps[s].detach().cpu())
            if return_source_indices:
                if source_idx is None:
                    source_idx_chunks.append(torch.full((b,), -1, dtype=torch.long))
                else:
                    source_idx_chunks.append(source_idx.detach().cpu().to(torch.long))
            n += b

        out = {s: torch.cat(chunks, dim=0) for s, chunks in buckets.items()}
        if return_source_indices:
            return out, torch.cat(source_idx_chunks, dim=0)
        return out
