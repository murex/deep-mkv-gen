"""Public provider-spec surface and internal provider factory."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..core.protocols import ScoreProvider, TerminalGradProvider
from .kl import KLRatioScoreConfig, KLScoreDifferenceConfig
from .kl.provider_kl import make_kl_provider
from .provider_hybrid import HybridProviderConfig, HybridTerminalGradProvider
from .provider_w2_geomloss import GeomLossW2ProviderConfig, GeomLossW2TerminalGradProvider


@dataclass
class KLProviderSpec:
    """Public spec for constructing a KL provider inside ``fit``."""

    cfg: KLRatioScoreConfig | KLScoreDifferenceConfig
    dim: int = 2
    target_score_model: Optional[ScoreProvider] = None
    mu_score_model: Optional[ScoreProvider] = None

    def validate(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("dim must be >= 1")
        self.cfg.validate()
        if isinstance(self.cfg, KLRatioScoreConfig):
            if self.target_score_model is not None or self.mu_score_model is not None:
                raise ValueError(
                    "target_score_model and mu_score_model are only supported with KLScoreDifferenceConfig."
                )


@dataclass
class W2ProviderSpec:
    """Public spec for constructing a GeomLoss W2 provider inside ``fit``."""

    cfg: GeomLossW2ProviderConfig = field(default_factory=GeomLossW2ProviderConfig)
    dim: int = 2

    def validate(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("dim must be >= 1")
        self.cfg.validate()


@dataclass
class HybridProviderSpec:
    """Public spec for constructing a hybrid KL+W2 provider inside ``fit``."""

    cfg: HybridProviderConfig = field(default_factory=HybridProviderConfig)
    dim: int = 2

    def validate(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("dim must be >= 1")
        self.cfg.validate()


ProviderSpec = KLProviderSpec | W2ProviderSpec | HybridProviderSpec


def build_provider(spec: ProviderSpec) -> TerminalGradProvider:
    """Build the concrete provider implementation for a public provider spec."""

    if isinstance(spec, KLProviderSpec):
        spec.validate()
        return make_kl_provider(
            cfg=copy.deepcopy(spec.cfg),
            dim=int(spec.dim),
            target_score_model=spec.target_score_model,
            mu_score_model=spec.mu_score_model,
        )

    if isinstance(spec, W2ProviderSpec):
        spec.validate()
        return GeomLossW2TerminalGradProvider(
            cfg=copy.deepcopy(spec.cfg),
            dim=int(spec.dim),
        )

    if isinstance(spec, HybridProviderSpec):
        spec.validate()
        return HybridTerminalGradProvider(
            cfg=copy.deepcopy(spec.cfg),
            dim=int(spec.dim),
        )

    raise TypeError(f"Unsupported provider spec type: {type(spec).__name__}")
