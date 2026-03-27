"""Factory for the concrete KL provider variants."""

from __future__ import annotations

import copy
from typing import Optional

from ...core.protocols import ScoreProvider
from .ratio_score import KLRatioScoreConfig, RatioScoreKLTerminalGradProvider
from .score_difference import KLScoreDifferenceConfig, ScoreDifferenceKLTerminalGradProvider
from .score_provider_dsm import DSMScoreModel


def make_kl_provider(
    cfg: KLRatioScoreConfig | KLScoreDifferenceConfig,
    dim: int = 2,
    *,
    target_score_model: Optional[ScoreProvider] = None,
    mu_score_model: Optional[ScoreProvider] = None,
) -> RatioScoreKLTerminalGradProvider | ScoreDifferenceKLTerminalGradProvider:
    """Instantiate the KL provider that matches the given config."""

    if isinstance(cfg, KLScoreDifferenceConfig):
        cfg.validate()
        if target_score_model is None:
            if cfg.target_score_cfg is None:
                raise ValueError(
                    "KLScoreDifferenceConfig.target_score_cfg is required when "
                    "target_score_model is not provided."
                )
            target_score_model = DSMScoreModel(
                dim=int(dim),
                cfg=copy.deepcopy(cfg.target_score_cfg),
            )
        return ScoreDifferenceKLTerminalGradProvider(
            target_score_model=target_score_model,
            mu_score_model=mu_score_model,
            cfg=cfg,
        )

    if isinstance(cfg, KLRatioScoreConfig):
        cfg.validate()
        if target_score_model is not None or mu_score_model is not None:
            raise ValueError(
                "target_score_model and mu_score_model are only valid with KLScoreDifferenceConfig."
            )
        return RatioScoreKLTerminalGradProvider(cfg=cfg, dim=int(dim))

    raise TypeError(f"Unsupported KL config type: {type(cfg).__name__}")
