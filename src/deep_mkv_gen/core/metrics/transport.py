"""Transport metrics used by stage evaluation."""

from __future__ import annotations

import math
import secrets
from typing import Optional

import numpy as np
import ot as pot
import torch


def wasserstein2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute empirical W2 between two point clouds with POT."""

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape (N, d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("x and y must each contain at least one sample")
    a = np.ones(x.shape[0], dtype=np.float64) / float(x.shape[0])
    b = np.ones(y.shape[0], dtype=np.float64) / float(y.shape[0])
    M = torch.cdist(x, y).pow(2).detach().cpu().numpy()
    return float(math.sqrt(pot.emd2(a, b, M, numItermax=int(1e7))))


@torch.no_grad()
def sliced_w2_sq(
    x: torch.Tensor,
    y: torch.Tensor,
    projections: int = 128,
    seed: Optional[int] = None,
) -> float:
    """Compute empirical sliced W2 squared between two point clouds."""
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must have shape (N,d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    if projections < 1:
        raise ValueError("projections must be >= 1")

    n = min(int(x.shape[0]), int(y.shape[0]))
    if n < 1:
        raise ValueError("x and y must each contain at least one sample")

    x_eval = x[:n]
    y_eval = y[:n].to(x_eval.device)
    if not x_eval.dtype.is_floating_point:
        x_eval = x_eval.float()
    if not y_eval.dtype.is_floating_point:
        y_eval = y_eval.float()
    y_eval = y_eval.to(dtype=x_eval.dtype)

    d = int(x_eval.shape[1])
    gen = torch.Generator(device="cpu")
    if seed is None:
        gen.manual_seed(secrets.randbits(63))
    else:
        gen.manual_seed(int(seed))
    v = torch.randn(projections, d, generator=gen, dtype=torch.float32)
    v = v.to(device=x_eval.device, dtype=x_eval.dtype)

    v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-12)
    x_proj = x_eval @ v.T
    y_proj = y_eval @ v.T
    x_sorted = x_proj.sort(dim=0).values
    y_sorted = y_proj.sort(dim=0).values
    return float((x_sorted - y_sorted).pow(2).mean().item())
