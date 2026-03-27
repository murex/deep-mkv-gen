"""Stage-level evaluation records and transport summary helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .metrics import sliced_w2_sq, wasserstein2
from .runtime_utils import isolated_seed

@dataclass
class StageRecord:
    """Aggregated training and evaluation metrics for one stage."""

    stage: int
    k: float
    steps_done: int
    mean_grad_norm: float
    mean_target_grad_clip_frac: float
    mean_target_grad_preclip_norm: float
    mean_drift_clip_frac: float
    mean_drift_clip_scale: float
    w2: float
    w2_std: float
    w2_se: float
    sw2_sq: float
    sw2_sq_std: float
    kl: float
    saved_best_checkpoint: bool
    mean_controller_grad_clip_frac: float = float("nan")
    mean_controller_grad_preclip_norm: float = float("nan")
    controller_grad_clip_count: int = 0
    controller_grad_clip_total: int = 0
    provider_metrics: Dict[str, float] = field(default_factory=dict)


def mean_of(logs: Sequence[Dict[str, float]], key: str, default: float = float("nan")) -> float:
    """Return the mean value for a metric key across log records."""

    vals = [float(r[key]) for r in logs if key in r]
    if not vals:
        return float(default)
    return float(np.mean(vals))


def mean_std_se(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean, sample standard deviation, and standard error."""

    if len(values) < 1:
        raise ValueError("values must be non-empty")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
        se = float(std / math.sqrt(float(arr.size)))
    else:
        std = 0.0
        se = 0.0
    return mean, std, se


def transport_eval_stats(
    x_gen: torch.Tensor,
    target_eval_sets: Sequence[torch.Tensor],
    sw2_proj: int,
    sw2_seed: int,
) -> Dict[str, float]:
    """Aggregate W2 and sliced-W2 statistics over repeated target draws."""

    if len(target_eval_sets) < 1:
        raise ValueError("target_eval_sets must be non-empty")
    w2_vals: List[float] = []
    sw2_vals: List[float] = []
    for i, target_eval in enumerate(target_eval_sets):
        w2_vals.append(wasserstein2(x_gen, target_eval))
        sw2_vals.append(
            sliced_w2_sq(
                x_gen,
                target_eval,
                projections=sw2_proj,
                seed=int(sw2_seed + i),
            )
        )
    w2_mean, w2_std, w2_se = mean_std_se(w2_vals)
    sw2_mean, sw2_std, _ = mean_std_se(sw2_vals)
    return {
        "w2": float(w2_mean),
        "w2_std": float(w2_std),
        "w2_se": float(w2_se),
        "sw2_sq": float(sw2_mean),
        "sw2_sq_std": float(sw2_std),
    }


def make_target_eval_sets(
    target_samples: torch.Tensor,
    eval_n: int,
    repeats: int,
    generator: torch.Generator,
) -> List[torch.Tensor]:
    """Pre-draw repeated target evaluation batches with a fixed generator."""

    if eval_n < 1:
        raise ValueError("eval_n must be >= 1")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    n = int(target_samples.shape[0])
    out: List[torch.Tensor] = []
    total_needed = int(eval_n) * int(repeats)
    if total_needed <= n:
        perm = torch.randperm(n, generator=generator)
        for i in range(int(repeats)):
            lo = int(i) * int(eval_n)
            hi = lo + int(eval_n)
            out.append(target_samples[perm[lo:hi]].clone())
        return out
    for _ in range(int(repeats)):
        idx = torch.randint(0, n, (int(eval_n),), generator=generator)
        out.append(target_samples[idx].clone())
    return out
