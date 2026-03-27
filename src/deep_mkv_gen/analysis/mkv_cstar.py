"""Standalone KL and c-star estimators used outside the training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..core.mkv_eval import isolated_seed


@dataclass
class KLResult:
    """Result container for KL and c-star estimators."""

    method: str
    kl_hat: float
    c_star_hat: float
    n_target_used: int
    n_baseline_used: int
    details: Dict[str, float]


class RatioClassifier(nn.Module):
    """Small MLP used to estimate log density ratio via binary classification."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def log_normal_diag(x: torch.Tensor, mean: torch.Tensor, var_diag: torch.Tensor) -> torch.Tensor:
    """Evaluate a diagonal Gaussian log density."""

    d = x.shape[-1]
    log_det = torch.log(var_diag).sum()
    quad = ((x - mean) ** 2 / var_diag).sum(dim=-1)
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + quad)


def sample_gamma(
    n: int,
    mean: np.ndarray,
    std_scalar: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the Gaussian baseline used by the estimators."""

    return mean[None, :] + std_scalar * rng.standard_normal((n, mean.shape[0]))


def estimate_kl_classifier(
    target: np.ndarray,
    gamma_mean: np.ndarray,
    gamma_std: float,
    seed: int,
    device: str,
    hidden_dim: int = 256,
    num_layers: int = 4,
    steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    eval_fraction: float = 0.2,
    logit_clip: float = 25.0,
) -> KLResult:
    """Estimate KL with a learned density-ratio classifier."""

    rng = np.random.default_rng(seed)
    n = target.shape[0]
    n_eval = int(round(eval_fraction * n))
    n_eval = min(max(n_eval, 128), n - 1)
    n_train = n - n_eval

    perm = rng.permutation(n)
    x_train = target[perm[:n_train]]
    x_eval = target[perm[n_train:]]

    y_train = sample_gamma(n_train, gamma_mean, gamma_std, rng)
    y_eval = sample_gamma(n_eval, gamma_mean, gamma_std, rng)

    torch_device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device if device != "auto" else "cpu")
    model = RatioClassifier(dim=target.shape[1], hidden_dim=hidden_dim, num_layers=num_layers).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train).to(torch_device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch_device, dtype=torch.float32)
    x_eval_t = torch.from_numpy(x_eval).to(torch_device, dtype=torch.float32)
    y_eval_t = torch.from_numpy(y_eval).to(torch_device, dtype=torch.float32)

    model.train()
    for _ in range(steps):
        idx_pos = torch.randint(0, n_train, (batch_size,), device=torch_device)
        idx_neg = torch.randint(0, n_train, (batch_size,), device=torch_device)
        x_all = torch.cat([x_train_t[idx_pos], y_train_t[idx_neg]], dim=0)
        labels = torch.cat(
            [torch.ones(batch_size, device=torch_device), torch.zeros(batch_size, device=torch_device)],
            dim=0,
        )
        logits = model(x_all)
        loss = bce(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_target = model(x_eval_t)
        logits_target = torch.nan_to_num(logits_target, nan=0.0, posinf=60.0, neginf=-60.0)
        if logit_clip > 0:
            logits_target = logits_target.clamp(-logit_clip, logit_clip)
        kl_hat = float(logits_target.mean().item())

        eval_x = torch.cat([x_eval_t, y_eval_t], dim=0)
        eval_labels = torch.cat(
            [torch.ones(n_eval, device=torch_device), torch.zeros(n_eval, device=torch_device)],
            dim=0,
        )
        eval_logits = model(eval_x)
        eval_acc = float(((eval_logits > 0).float() == eval_labels).float().mean().item())
        eval_bce = float(bce(eval_logits, eval_labels).item())

    return KLResult(
        method="classifier",
        kl_hat=kl_hat,
        c_star_hat=float("nan"),
        n_target_used=n,
        n_baseline_used=n,
        details={
            "heldout_bce": eval_bce,
            "heldout_acc": eval_acc,
            "near_separable_flag": 1.0 if eval_acc >= 0.98 else 0.0,
            "n_train": float(n_train),
            "n_eval": float(n_eval),
            "logit_clip": float(logit_clip),
            "logit_target_mean": float(logits_target.mean().item()),
            "logit_target_std": float(logits_target.std(unbiased=False).item()),
        },
    )


def _kth_neighbor_distance(
    query: torch.Tensor,
    ref: torch.Tensor,
    k: int,
    chunk_size: int,
    exclude_self: bool,
) -> torch.Tensor:
    n = query.shape[0]
    out: List[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = query[start:end]
        dists = torch.cdist(q, ref, p=2)
        if exclude_self:
            if ref.shape[0] != query.shape[0]:
                raise ValueError("exclude_self=True requires query and ref to have same size")
            rows = torch.arange(end - start, device=dists.device)
            cols = torch.arange(start, end, device=dists.device)
            dists[rows, cols] = float("inf")
        out.append(torch.topk(dists, k=k, largest=False).values[:, -1])
    return torch.cat(out, dim=0)


def estimate_kl_knn(
    target: np.ndarray,
    baseline: np.ndarray,
    k: int = 5,
    chunk_size: int = 512,
) -> KLResult:
    """Estimate KL with a kNN plug-in estimator."""

    n, d = target.shape
    m = baseline.shape[0]
    if n <= 1:
        raise ValueError("Need n >= 2 target samples for kNN KL estimate")
    if k < 1:
        raise ValueError("knn k must be >= 1")
    if k >= n:
        raise ValueError(f"knn k must be < n_target_used ({n})")
    if k > m:
        raise ValueError(f"knn k must be <= n_baseline_used ({m})")

    x = torch.from_numpy(target).to(dtype=torch.float64)
    y = torch.from_numpy(baseline).to(dtype=torch.float64)
    eps_k = _kth_neighbor_distance(query=x, ref=x, k=k, chunk_size=chunk_size, exclude_self=True).clamp_min(1e-12)
    nu_k = _kth_neighbor_distance(query=x, ref=y, k=k, chunk_size=chunk_size, exclude_self=False).clamp_min(1e-12)

    kl_hat = math.log(m / (n - 1.0)) + d * float(torch.log(nu_k / eps_k).mean().item())
    return KLResult(
        method="knn",
        kl_hat=kl_hat,
        c_star_hat=float("nan"),
        n_target_used=n,
        n_baseline_used=m,
        details={
            "k": float(k),
            "chunk_size": float(chunk_size),
            "eps_k_median": float(eps_k.median().item()),
            "nu_k_median": float(nu_k.median().item()),
        },
    )


def estimate_cstar_classifier(
    target: np.ndarray,
    gamma_mean: np.ndarray,
    gamma_std: float,
    sigma: float,
    seed: int,
    device: str,
    hidden_dim: int = 256,
    num_layers: int = 4,
    steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    eval_fraction: float = 0.2,
    logit_clip: float = 25.0,
) -> KLResult:
    """Estimate c-star by scaling the classifier KL estimate."""

    res = estimate_kl_classifier(
        target=target,
        gamma_mean=gamma_mean,
        gamma_std=gamma_std,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eval_fraction=eval_fraction,
        logit_clip=logit_clip,
    )
    res.c_star_hat = float((sigma ** 2) * res.kl_hat)
    return res


def estimate_cstar_knn(
    target: np.ndarray,
    baseline: np.ndarray,
    sigma: float,
    k: int = 5,
    chunk_size: int = 512,
) -> KLResult:
    """Estimate c-star by scaling the kNN KL estimate."""

    res = estimate_kl_knn(target=target, baseline=baseline, k=k, chunk_size=chunk_size)
    res.c_star_hat = float((sigma ** 2) * res.kl_hat)
    return res


def estimate_cstar_mc(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    sigma: float,
    T: float,
    n_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    sample_target_fn: Callable[..., torch.Tensor],
    log_target_prob_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, float]:
    if n_samples < 1:
        raise ValueError("cstar n_samples must be >= 1")
    if batch_size < 1:
        raise ValueError("cstar batch_size must be >= 1")

    dev = torch.device(device)
    mean_d = mean.to(dev).reshape(1, -1)
    std_d = std.to(dev).reshape(1, -1)
    mu0_n = (-mean_d / std_d).squeeze(0)
    gamma_var_diag = (1.0 / (std_d.squeeze(0) ** 2)) + float(sigma * sigma * T)

    acc = 0.0
    seen = 0
    with isolated_seed(device, int(seed)):
        while seen < n_samples:
            b = min(batch_size, n_samples - seen)
            x = sample_target_fn(b, device=dev)
            x_n = (x - mean_d) / std_d
            log_rho = log_target_prob_fn(x).to(dev)
            if log_rho.ndim != 1 or log_rho.shape[0] != b:
                raise ValueError("log_target_prob_fn must return a 1D tensor of shape (batch_size,)")
            log_gamma = log_normal_diag(x_n, mu0_n, gamma_var_diag)
            acc += float((log_rho - log_gamma).sum().item())
            seen += b

    kl_rho_gamma = acc / float(n_samples)
    c_star = float(sigma * sigma * kl_rho_gamma)
    out = {
        "kl_rho_gamma": float(kl_rho_gamma),
        "c_star": float(c_star),
    }
    for i, v in enumerate(mu0_n.tolist(), start=1):
        out[f"gamma_mean_x{i}"] = float(v)
    for i, v in enumerate(gamma_var_diag.tolist(), start=1):
        out[f"gamma_var_x{i}"] = float(v)
    return out
