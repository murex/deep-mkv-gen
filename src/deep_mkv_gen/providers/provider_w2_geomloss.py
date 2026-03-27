"""W2^2 terminal gradients from a GeomLoss Sinkhorn objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import warnings

import torch

from ..core.protocols import TargetSamplerLike, TerminalGradOutput

try:  # pragma: no cover - depends on user environment
    from geomloss import SamplesLoss
except Exception:  # pragma: no cover - depends on user environment
    SamplesLoss = None  # type: ignore[assignment]


@dataclass
class GeomLossW2ProviderConfig:
    """Configuration for the GeomLoss-based W2 provider."""

    blur: float = 0.2
    scaling: float = 0.9
    debias: bool = True
    backend: str = "auto"
    diameter: Optional[float] = None
    reach: Optional[float] = None
    truncate: float = 5.0
    cluster_scale: Optional[float] = None
    verbose: bool = False

    target_batch_size: Optional[int] = None
    target_draws_per_step: int = 1

    target_grad_norm_clip: Optional[float] = 10.0

    def validate(self) -> None:
        if float(self.blur) <= 0.0:
            raise ValueError("blur must be > 0")
        if not (0.0 < float(self.scaling) < 1.0):
            raise ValueError("scaling must be in (0, 1)")
        if self.backend not in {"auto", "tensorized", "online", "multiscale"}:
            raise ValueError("backend must be one of {'auto', 'tensorized', 'online', 'multiscale'}")
        if self.diameter is not None and float(self.diameter) <= 0.0:
            raise ValueError("diameter must be > 0 when provided")
        if self.reach is not None and float(self.reach) <= 0.0:
            raise ValueError("reach must be > 0 when provided")
        if float(self.truncate) <= 0.0:
            raise ValueError("truncate must be > 0")
        if self.cluster_scale is not None and float(self.cluster_scale) <= 0.0:
            raise ValueError("cluster_scale must be > 0 when provided")
        if self.target_batch_size is not None and int(self.target_batch_size) < 1:
            raise ValueError("target_batch_size must be >= 1 when provided")
        if int(self.target_draws_per_step) < 1:
            raise ValueError("target_draws_per_step must be >= 1")
        if self.target_grad_norm_clip is not None and float(self.target_grad_norm_clip) <= 0.0:
            raise ValueError("target_grad_norm_clip must be > 0 when provided")


class GeomLossW2TerminalGradProvider:
    """W2^2 terminal-gradient provider backed by GeomLoss."""

    def __init__(
        self,
        cfg: GeomLossW2ProviderConfig,
        dim: int = 2,
    ):
        self.cfg = cfg
        self.target_sampler: Optional[TargetSamplerLike] = None
        self.dim = dim
        self.device: Optional[torch.device] = None
        self.total_steps: Optional[int] = None
        self.loss_fn = None
        self._resolved_backend: Optional[str] = None

    def set_target_sampler(self, target_sampler: TargetSamplerLike) -> None:
        self.target_sampler = target_sampler

    def _require_geomloss(self) -> None:
        if SamplesLoss is None:
            raise ImportError(
                "GeomLoss is not installed. Install it with `pip install geomloss`. "
                "For the 'online' and 'multiscale' backends you will typically also "
                "want `pykeops`."
            )

    def _resolve_backend(self) -> str:
        backend = str(self.cfg.backend)
        if backend not in {"auto", "tensorized", "online", "multiscale"}:
            raise ValueError(
                "backend must be one of {'auto', 'tensorized', 'online', 'multiscale'}"
            )
        if backend == "multiscale" and self.dim > 3:
            warnings.warn(
                "GeomLoss backend='multiscale' is intended for dimensions <= 3. "
                "Falling back to backend='online'."
            )
            backend = "online"
        return backend

    def on_train_start(self, device: Union[str, torch.device], train_cfg: Any) -> None:
        """Resolve the GeomLoss backend and build the Sinkhorn loss object."""

        self.cfg.validate()
        self._require_geomloss()
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before on_train_start(...).")
        self.total_steps = getattr(train_cfg, "num_steps", None)
        self.device = torch.device(device)
        self._resolved_backend = self._resolve_backend()

        self.loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=float(self.cfg.blur),
            reach=self.cfg.reach,
            diameter=self.cfg.diameter,
            scaling=float(self.cfg.scaling),
            truncate=float(self.cfg.truncate),
            cluster_scale=self.cfg.cluster_scale,
            debias=bool(self.cfg.debias),
            potentials=False,
            verbose=bool(self.cfg.verbose),
            backend=self._resolved_backend,
        )

    def _sample_target(self, n: int) -> torch.Tensor:
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before sampling targets.")
        y = self.target_sampler.sample(n)
        if y.ndim != 2 or y.shape[1] != self.dim:
            raise ValueError(
                f"target_sampler.sample({n}) must return shape (n, {self.dim})"
            )
        return y.to(self.device)

    def _single_batch_grad_and_metrics(
        self,
        xT_det: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Return one GeomLoss gradient estimate and its diagnostics."""

        if self.loss_fn is None:
            raise RuntimeError("Call on_train_start(...) before get_grad_target(...).")

        n = int(xT_det.shape[0])
        m = int(self.cfg.target_batch_size or n)
        if n < 1 or m < 1:
            raise ValueError("Batch sizes must be positive.")

        y = self._sample_target(m)
        x = xT_det.clone().to(self.device).requires_grad_(True)

        raw_loss = self.loss_fn(x, y)
        if raw_loss.ndim != 0:
            raw_loss = raw_loss.squeeze()
            if raw_loss.ndim != 0:
                raise RuntimeError(
                    f"Expected a scalar GeomLoss value, got shape {tuple(raw_loss.shape)}"
                )

        grad_scalar = torch.autograd.grad(raw_loss, x, create_graph=False)[0]

        measure_factor = float(n)
        cost_factor = 2.0

        grad_target = (measure_factor * cost_factor) * grad_scalar.detach()

        raw_value = float(raw_loss.detach().item())
        metrics = {
            "geomloss_value_raw": raw_value,
            "geomloss_value_times_2": 2.0 * raw_value,
            "geomloss_batch_size_x": float(n),
            "geomloss_batch_size_y": float(m),
            "geomloss_debias": 1.0 if self.cfg.debias else 0.0,
            "geomloss_measure_factor": measure_factor,
            "geomloss_cost_factor": cost_factor,
        }
        return grad_target, metrics

    def get_grad_target(
        self,
        xT: torch.Tensor,
        step: int,
    ) -> TerminalGradOutput:
        """Average GeomLoss gradients and apply optional clipping."""

        if self.device is None:
            raise RuntimeError("Call on_train_start(...) before get_grad_target(...).")
        if xT.ndim != 2 or xT.shape[1] != self.dim:
            raise ValueError(f"xT must have shape (B, {self.dim})")

        xT_det = xT.detach()
        metrics: Dict[str, float] = {}
        num_draws = max(1, int(self.cfg.target_draws_per_step))

        grad_sum = torch.zeros_like(xT_det, device=self.device)
        metrics_accum: Dict[str, float] = {}

        for _ in range(num_draws):
            grad_i, metrics_i = self._single_batch_grad_and_metrics(xT_det=xT_det)
            grad_sum += grad_i
            for k, v in metrics_i.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)

        grad_target = grad_sum / float(num_draws)
        metrics.update({k: v / float(num_draws) for k, v in metrics_accum.items()})

        norms = torch.linalg.norm(grad_target, dim=-1, keepdim=True).clamp_min(1e-12)
        metrics["target_grad_preclip_norm_mean"] = float(norms.mean().item())

        if self.cfg.target_grad_norm_clip is not None:
            clip = float(self.cfg.target_grad_norm_clip)
            scale = torch.clamp(clip / norms, max=1.0)
            grad_target = grad_target * scale
            metrics["target_grad_clip_frac"] = float(
                (scale.squeeze(-1) < (1.0 - 1e-6)).float().mean().item()
            )
            metrics["target_grad_scale_mean"] = float(scale.mean().item())
        else:
            metrics["target_grad_clip_frac"] = 0.0
            metrics["target_grad_scale_mean"] = 1.0

        post_norm_mean = float(torch.linalg.norm(grad_target, dim=-1).mean().item())
        metrics["target_grad_postclip_norm_mean"] = post_norm_mean
        metrics["grad_norm_mean"] = post_norm_mean
        metrics["provider_step"] = float(step)

        return TerminalGradOutput(
            grad_target=grad_target.detach(),
            metrics=metrics,
        )
