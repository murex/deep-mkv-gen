"""Hybrid terminal gradients from weighted KL and W2^2 branches."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch

from ..core.protocols import TargetSamplerLike, TerminalGradOutput

from .kl import (
    KLRatioScoreConfig,
    KLScoreDifferenceConfig,
    make_kl_provider,
)
from .provider_w2_geomloss import (
    GeomLossW2ProviderConfig,
    GeomLossW2TerminalGradProvider,
)


@dataclass
class HybridProviderConfig:
    """Configuration for the weighted KL plus W2^2 provider."""

    kl_weight: float = 1.0
    w2_weight: float = 1.0

    target_grad_norm_clip: Optional[float] = 10.0

    kl_cfg: KLRatioScoreConfig | KLScoreDifferenceConfig = field(default_factory=KLScoreDifferenceConfig)
    w2_cfg: GeomLossW2ProviderConfig = field(default_factory=GeomLossW2ProviderConfig)

    def validate(self) -> None:
        if float(self.kl_weight) < 0.0:
            raise ValueError("kl_weight must be >= 0")
        if float(self.w2_weight) < 0.0:
            raise ValueError("w2_weight must be >= 0")
        if self.target_grad_norm_clip is not None and float(self.target_grad_norm_clip) <= 0.0:
            raise ValueError("target_grad_norm_clip must be > 0 when provided")
        self.kl_cfg.validate()
        self.w2_cfg.validate()


class HybridTerminalGradProvider:
    """Terminal-gradient provider that sums KL and W2^2 branches."""

    def __init__(
        self,
        cfg: HybridProviderConfig,
        dim: int = 2,
    ):
        self.cfg = cfg
        self.target_sampler: Optional[TargetSamplerLike] = None
        self.dim = dim
        self.device: Optional[torch.device] = None
        self.total_steps: Optional[int] = None
        self._kl_cfg_runtime = self._make_kl_cfg_runtime(cfg.kl_cfg)
        self._w2_cfg_runtime = self._make_w2_cfg_runtime(cfg.w2_cfg)

        self.kl_provider = make_kl_provider(cfg=self._kl_cfg_runtime, dim=dim)
        self.w2_provider = GeomLossW2TerminalGradProvider(
            cfg=self._w2_cfg_runtime,
            dim=dim,
        )

    def set_target_sampler(self, target_sampler: TargetSamplerLike) -> None:
        self.target_sampler = target_sampler
        self.kl_provider.set_target_sampler(target_sampler)
        self.w2_provider.set_target_sampler(target_sampler)

    def on_train_start(self, device: Union[str, torch.device], train_cfg: Any) -> None:
        """Initialize both inner providers for a new training phase."""

        self.cfg.validate()
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before on_train_start(...).")
        self.device = torch.device(device)
        self.total_steps = getattr(train_cfg, "num_steps", None)
        self.kl_provider.on_train_start(device=device, train_cfg=train_cfg)
        self.w2_provider.on_train_start(device=device, train_cfg=train_cfg)

    @staticmethod
    def _prefixed(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {f"{prefix}{k}": float(v) for k, v in metrics.items()}

    @staticmethod
    def _make_kl_cfg_runtime(
        cfg: KLRatioScoreConfig | KLScoreDifferenceConfig,
    ) -> KLRatioScoreConfig | KLScoreDifferenceConfig:
        runtime_cfg = copy.deepcopy(cfg)
        if isinstance(runtime_cfg, KLScoreDifferenceConfig):
            runtime_cfg.target_grad_norm_clip = None
        else:
            runtime_cfg.target_grad_norm_clip = None
        return runtime_cfg

    @staticmethod
    def _make_w2_cfg_runtime(cfg: GeomLossW2ProviderConfig) -> GeomLossW2ProviderConfig:
        runtime_cfg = copy.deepcopy(cfg)
        runtime_cfg.target_grad_norm_clip = None
        return runtime_cfg

    def get_grad_target(
        self,
        xT: torch.Tensor,
        step: int,
    ) -> TerminalGradOutput:
        """Combine KL and W2 terminal fields and clip once at the top level."""

        if self.device is None:
            raise RuntimeError("Call on_train_start(...) before get_grad_target(...).")
        if xT.ndim != 2 or xT.shape[1] != self.dim:
            raise ValueError(f"xT must have shape (B, {self.dim})")

        xT_det = xT.detach()
        metrics: Dict[str, float] = {
            "hybrid_kl_weight": float(self.cfg.kl_weight),
            "hybrid_w2_weight": float(self.cfg.w2_weight),
            "hybrid_kl_uses_score_difference": (
                1.0 if isinstance(self.cfg.kl_cfg, KLScoreDifferenceConfig) else 0.0
            ),
            "hybrid_kl_uses_ratio_score": (
                1.0 if isinstance(self.cfg.kl_cfg, KLRatioScoreConfig) else 0.0
            ),
            "provider_step": float(step),
        }

        out_kl = self.kl_provider.get_grad_target(xT=xT, step=step)
        grad_kl = out_kl.grad_target.to(xT_det.device)
        metrics.update(self._prefixed(out_kl.metrics, "kl_"))

        out_w2 = self.w2_provider.get_grad_target(xT=xT, step=step)
        grad_w2 = out_w2.grad_target.to(xT_det.device)
        metrics.update(self._prefixed(out_w2.metrics, "w2_"))

        kl_norm = torch.linalg.norm(grad_kl, dim=-1)
        metrics["hybrid_kl_component_norm_mean"] = float(kl_norm.mean().item())
        metrics["hybrid_kl_component_norm_std"] = float(kl_norm.std(unbiased=False).item())

        w2_norm = torch.linalg.norm(grad_w2, dim=-1)
        metrics["hybrid_w2_component_norm_mean"] = float(w2_norm.mean().item())
        metrics["hybrid_w2_component_norm_std"] = float(w2_norm.std(unbiased=False).item())

        grad_target = float(self.cfg.kl_weight) * grad_kl + float(self.cfg.w2_weight) * grad_w2

        kl_weighted_norm = torch.linalg.norm(float(self.cfg.kl_weight) * grad_kl, dim=-1)
        w2_weighted_norm = torch.linalg.norm(float(self.cfg.w2_weight) * grad_w2, dim=-1)
        metrics["hybrid_kl_weighted_norm_mean"] = float(kl_weighted_norm.mean().item())
        metrics["hybrid_w2_weighted_norm_mean"] = float(w2_weighted_norm.mean().item())

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

        post_norm = torch.linalg.norm(grad_target, dim=-1)
        post_norm_mean = float(post_norm.mean().item())
        metrics["target_grad_postclip_norm_mean"] = post_norm_mean
        metrics["grad_norm_mean"] = post_norm_mean

        return TerminalGradOutput(
            grad_target=grad_target.detach(),
            metrics=metrics,
        )
