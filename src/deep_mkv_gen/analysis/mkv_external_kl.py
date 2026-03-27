"""External KL evaluation helpers for trained models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from ..core.metrics import (
    ClassifierRunResult,
    RepeatedClassifierResult,
    estimate_kl_classifier_from_samples,
    estimate_kl_knn_from_samples,
    repeat_kl_classifier_from_samples,
)
from ..core.runtime_utils import isolated_seed, preserve_model_mode


ArrayLike = Union[np.ndarray, torch.Tensor]


def _resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Normalize a device specifier."""

    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _to_tensor(x: ArrayLike, device: Union[str, torch.device], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert NumPy or torch inputs to a tensor on the requested device."""

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=_resolve_device(device), dtype=dtype)
    if torch.is_tensor(x):
        return x.to(device=_resolve_device(device), dtype=dtype)
    raise TypeError("Expected a numpy array or torch tensor.")


@torch.no_grad()
def sample_terminal(
    model: Any,
    *,
    num_samples: int,
    batch_size: int = 2048,
    device: Union[str, torch.device] = "auto",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample terminal states from a model without mutating RNG or mode."""

    dev = _resolve_device(device)
    rng_ctx = isolated_seed(dev, int(seed)) if seed is not None else nullcontext()
    with preserve_model_mode(model), rng_ctx:
        try:
            from ..core.sampling import sample as model_sample  # type: ignore
            return model_sample(
                model,
                num_samples=int(num_samples),
                batch_size=int(batch_size),
                device=dev,
            ).cpu()
        except Exception:
            pass

        model.to(dev)
        model.eval()
        out: List[torch.Tensor] = []
        seen = 0
        while seen < int(num_samples):
            b = min(int(batch_size), int(num_samples) - seen)
            x0 = model._sample_source(b, device=dev)
            xT, _, _ = model._simulate(x0, antithetic=False)
            out.append(xT.detach().cpu())
            seen += b
        return torch.cat(out, dim=0)


@torch.no_grad()
def draw_target_samples(
    *,
    num_samples: int,
    target_sampler: Optional[Any] = None,
    target_sample_fn: Optional[Callable[..., ArrayLike]] = None,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Draw target samples from either a sampler or a callback."""

    rng_ctx = isolated_seed(device, int(seed)) if seed is not None else nullcontext()
    with rng_ctx:
        if (target_sampler is None) == (target_sample_fn is None):
            raise ValueError("Provide exactly one of target_sampler or target_sample_fn.")
        if target_sample_fn is not None:
            samples = target_sample_fn(int(num_samples), device=_resolve_device(device))
            return _to_tensor(samples, device="cpu", dtype=torch.float32).cpu()
        samples = target_sampler.sample(int(num_samples))
        return _to_tensor(samples, device="cpu", dtype=torch.float32).cpu()


def evaluate_model_kl_to_target(
    model: Any,
    *,
    num_model_samples: int,
    num_target_samples: int,
    target_sampler: Optional[Any] = None,
    target_sample_fn: Optional[Callable[..., ArrayLike]] = None,
    model_batch_size: int = 2048,
    target_batch_size: int = 2048,
    device: Union[str, torch.device] = "auto",
    classifier_steps: int = 4000,
    classifier_batch_size: int = 1024,
    classifier_hidden_dim: int = 256,
    classifier_num_layers: int = 4,
    classifier_repeats: int = 3,
    classifier_lr: float = 1e-3,
    classifier_weight_decay: float = 1e-5,
    classifier_eval_fraction: float = 0.2,
    classifier_logit_clip: float = 30.0,
    classifier_grad_clip: float = 1.0,
    knn_k: Optional[int] = 5,
    knn_chunk_size: int = 1024,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate terminal KL with classifier and optional kNN estimates."""

    model_samples = sample_terminal(
        model,
        num_samples=int(num_model_samples),
        batch_size=int(model_batch_size),
        device=device,
        seed=int(seed),
    )
    target_samples = draw_target_samples(
        num_samples=int(num_target_samples),
        target_sampler=target_sampler,
        target_sample_fn=target_sample_fn,
        device="cpu",
        seed=int(seed + 17),
    )

    clf = repeat_kl_classifier_from_samples(
        model_samples=model_samples,
        target_samples=target_samples,
        repeats=int(classifier_repeats),
        seed=int(seed),
        device=device,
        hidden_dim=int(classifier_hidden_dim),
        num_layers=int(classifier_num_layers),
        steps=int(classifier_steps),
        batch_size=int(classifier_batch_size),
        lr=float(classifier_lr),
        weight_decay=float(classifier_weight_decay),
        eval_fraction=float(classifier_eval_fraction),
        logit_clip=float(classifier_logit_clip),
        grad_clip=float(classifier_grad_clip),
    )

    out: Dict[str, Any] = {
        "num_model_samples": int(num_model_samples),
        "num_target_samples": int(num_target_samples),
        "classifier_kl_mean": float(clf.kl_mean),
        "classifier_kl_std": float(clf.kl_std),
        "classifier_kl_se": float(clf.kl_se),
        "classifier_acc_mean": float(clf.heldout_acc_mean),
        "classifier_bce_mean": float(clf.heldout_bce_mean),
        "classifier_repeats": int(classifier_repeats),
        "classifier_steps": int(classifier_steps),
        "classifier_batch_size": int(classifier_batch_size),
        "runs": [
            {
                "kl_hat": float(r.kl_hat),
                "heldout_acc": float(r.heldout_acc),
                "heldout_bce": float(r.heldout_bce),
                "n_train_per_class": int(r.n_train_per_class),
                "n_eval_per_class": int(r.n_eval_per_class),
                "steps": int(r.steps),
                "batch_size": int(r.batch_size),
                "logit_mean": float(r.logit_mean),
                "logit_std": float(r.logit_std),
                "near_separable_flag": float(r.near_separable_flag),
            }
            for r in clf.runs
        ],
    }

    if knn_k is not None:
        out["knn"] = estimate_kl_knn_from_samples(
            model_samples=model_samples,
            target_samples=target_samples,
            k=int(knn_k),
            chunk_size=int(knn_chunk_size),
        )

    return out
