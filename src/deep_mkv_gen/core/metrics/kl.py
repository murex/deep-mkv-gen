"""Sample-based KL estimators used by evaluation and analysis code."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn

from ..runtime_utils import isolated_seed


ArrayLike = Union[np.ndarray, torch.Tensor]


def _resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _to_tensor(x: ArrayLike, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=_resolve_device(device), dtype=dtype)
    if torch.is_tensor(x):
        return x.to(device=_resolve_device(device), dtype=dtype)
    raise TypeError("Expected a numpy array or torch tensor.")


def _set_seed(seed: int, device: Union[str, torch.device] = "auto") -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    dev = _resolve_device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class ClassifierRunResult:
    """Diagnostics from one classifier-based KL estimation run."""

    kl_hat: float
    heldout_acc: float
    heldout_bce: float
    n_train_per_class: int
    n_eval_per_class: int
    steps: int
    batch_size: int
    logit_mean: float
    logit_std: float
    near_separable_flag: float


@dataclass
class RepeatedClassifierResult:
    """Aggregate statistics over repeated classifier KL runs."""

    kl_mean: float
    kl_std: float
    kl_se: float
    heldout_acc_mean: float
    heldout_bce_mean: float
    runs: List[ClassifierRunResult]


class RatioClassifier(nn.Module):
    """MLP classifier whose optimal logit is log p(x)/q(x) under equal priors."""

    def __init__(self, dim: int, hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def estimate_kl_classifier_from_samples(
    model_samples: ArrayLike,
    target_samples: ArrayLike,
    *,
    seed: int = 0,
    device: Union[str, torch.device] = "auto",
    hidden_dim: int = 256,
    num_layers: int = 4,
    steps: int = 4000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    eval_fraction: float = 0.2,
    logit_clip: float = 30.0,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
) -> ClassifierRunResult:
    """Estimate KL(P || Q) from samples x~P and y~Q via a binary classifier."""
    with isolated_seed(device, int(seed)):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not (0.0 < eval_fraction < 0.5):
            raise ValueError("eval_fraction must lie in (0, 0.5)")
        if not (0.0 <= label_smoothing < 0.5):
            raise ValueError("label_smoothing must lie in [0, 0.5)")

        x = _to_tensor(model_samples, device=device, dtype=torch.float32)
        y = _to_tensor(target_samples, device=device, dtype=torch.float32)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("model_samples and target_samples must have shape (n, d)")
        if x.shape[1] != y.shape[1]:
            raise ValueError("model_samples and target_samples must have the same dimension")

        n = min(int(x.shape[0]), int(y.shape[0]))
        if n < 256:
            raise ValueError("Need at least 256 samples from each distribution.")
        x = x[:n]
        y = y[:n]

        _set_seed(seed, device=device)
        rng = np.random.default_rng(int(seed))

        n_eval = int(round(float(eval_fraction) * n))
        n_eval = min(max(n_eval, 128), n - 1)
        n_train = n - n_eval

        perm_x = rng.permutation(n)
        perm_y = rng.permutation(n)

        x_train = x[perm_x[:n_train]]
        x_eval = x[perm_x[n_train:]]
        y_train = y[perm_y[:n_train]]
        y_eval = y[perm_y[n_train:]]

        dev = _resolve_device(device)
        model = RatioClassifier(dim=int(x.shape[1]), hidden_dim=hidden_dim, num_layers=num_layers).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        bce = nn.BCEWithLogitsLoss()

        pos_label = 1.0 - float(label_smoothing)
        neg_label = 0.0 + float(label_smoothing)

        model.train()
        for _ in range(int(steps)):
            idx_pos = torch.randint(0, n_train, (int(batch_size),), device=dev)
            idx_neg = torch.randint(0, n_train, (int(batch_size),), device=dev)
            x_all = torch.cat([x_train[idx_pos], y_train[idx_neg]], dim=0)
            labels = torch.cat(
                [
                    torch.full((int(batch_size),), pos_label, device=dev),
                    torch.full((int(batch_size),), neg_label, device=dev),
                ],
                dim=0,
            )
            logits = model(x_all)
            loss = bce(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_x = model(x_eval)
            logits_x = torch.nan_to_num(logits_x, nan=0.0, posinf=60.0, neginf=-60.0)
            if logit_clip is not None and float(logit_clip) > 0:
                logits_x = logits_x.clamp(-float(logit_clip), float(logit_clip))
            kl_hat = float(logits_x.mean().item())

            eval_inputs = torch.cat([x_eval, y_eval], dim=0)
            eval_labels = torch.cat(
                [
                    torch.ones(n_eval, device=dev),
                    torch.zeros(n_eval, device=dev),
                ],
                dim=0,
            )
            eval_logits = model(eval_inputs)
            heldout_acc = float(((eval_logits > 0).float() == eval_labels).float().mean().item())
            heldout_bce = float(bce(eval_logits, eval_labels).item())

        return ClassifierRunResult(
            kl_hat=float(kl_hat),
            heldout_acc=float(heldout_acc),
            heldout_bce=float(heldout_bce),
            n_train_per_class=int(n_train),
            n_eval_per_class=int(n_eval),
            steps=int(steps),
            batch_size=int(batch_size),
            logit_mean=float(logits_x.mean().item()),
            logit_std=float(logits_x.std(unbiased=False).item()),
            near_separable_flag=1.0 if heldout_acc >= 0.98 else 0.0,
        )


def repeat_kl_classifier_from_samples(
    model_samples: ArrayLike,
    target_samples: ArrayLike,
    *,
    repeats: int = 3,
    seed: int = 0,
    **kwargs: Any,
) -> RepeatedClassifierResult:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    runs: List[ClassifierRunResult] = []
    for i in range(int(repeats)):
        runs.append(
            estimate_kl_classifier_from_samples(
                model_samples=model_samples,
                target_samples=target_samples,
                seed=int(seed + 1009 * i),
                **kwargs,
            )
        )

    kl_vals = np.asarray([r.kl_hat for r in runs], dtype=np.float64)
    acc_vals = np.asarray([r.heldout_acc for r in runs], dtype=np.float64)
    bce_vals = np.asarray([r.heldout_bce for r in runs], dtype=np.float64)

    kl_mean = float(kl_vals.mean())
    kl_std = float(kl_vals.std(ddof=1)) if kl_vals.size > 1 else 0.0
    kl_se = float(kl_std / math.sqrt(float(kl_vals.size))) if kl_vals.size > 1 else 0.0

    return RepeatedClassifierResult(
        kl_mean=float(kl_mean),
        kl_std=float(kl_std),
        kl_se=float(kl_se),
        heldout_acc_mean=float(acc_vals.mean()),
        heldout_bce_mean=float(bce_vals.mean()),
        runs=runs,
    )


def _kth_neighbor_distance(
    query: torch.Tensor,
    ref: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
    exclude_self: bool,
) -> torch.Tensor:
    out: List[torch.Tensor] = []
    n = int(query.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        q = query[start:end]
        dists = torch.cdist(q, ref, p=2)
        if exclude_self:
            if ref.shape[0] != query.shape[0]:
                raise ValueError("exclude_self=True requires query and ref to have the same size")
            rows = torch.arange(end - start, device=dists.device)
            cols = torch.arange(start, end, device=dists.device)
            dists[rows, cols] = float("inf")
        out.append(torch.topk(dists, k=int(k), largest=False).values[:, -1])
    return torch.cat(out, dim=0)


def estimate_kl_knn_from_samples(
    model_samples: ArrayLike,
    target_samples: ArrayLike,
    *,
    k: int = 5,
    chunk_size: int = 1024,
) -> dict[str, float]:
    """kNN estimate of KL(P || Q) from samples x~P and y~Q."""
    x = _to_tensor(model_samples, device="cpu", dtype=torch.float64)
    y = _to_tensor(target_samples, device="cpu", dtype=torch.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("model_samples and target_samples must have shape (n, d)")
    if x.shape[1] != y.shape[1]:
        raise ValueError("model_samples and target_samples must have the same dimension")

    n, d = int(x.shape[0]), int(x.shape[1])
    m = int(y.shape[0])
    if n <= 1:
        raise ValueError("Need at least two model samples for the kNN KL estimate.")
    if k < 1 or k >= n or k > m:
        raise ValueError("Require 1 <= k < n_model and k <= n_target.")

    eps_k = _kth_neighbor_distance(x, x, k=int(k), chunk_size=int(chunk_size), exclude_self=True).clamp_min(1e-12)
    nu_k = _kth_neighbor_distance(x, y, k=int(k), chunk_size=int(chunk_size), exclude_self=False).clamp_min(1e-12)

    kl_hat = math.log(m / (n - 1.0)) + d * float(torch.log(nu_k / eps_k).mean().item())
    return {
        "kl_hat": float(kl_hat),
        "k": float(k),
        "chunk_size": float(chunk_size),
        "eps_k_median": float(eps_k.median().item()),
        "nu_k_median": float(nu_k.median().item()),
    }
