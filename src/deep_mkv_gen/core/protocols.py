"""Shared runtime protocols and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Protocol, Union

import torch
import torch.nn as nn


@dataclass
class TerminalGradOutput:
    """Terminal field returned by a provider for the current xT batch."""

    grad_target: torch.Tensor
    metrics: Dict[str, float]


@dataclass
class ScoreComputeOutput:
    """Score value and training metrics returned by a score model."""

    score: Optional[torch.Tensor]
    metrics: Dict[str, float]


class TerminalGradProvider(Protocol):
    """Runtime contract for terminal-gradient providers."""

    def set_target_sampler(self, target_sampler: "TargetSamplerLike") -> None:
        ...

    def on_train_start(self, device: Union[str, torch.device], train_cfg: Any) -> None:
        ...

    def get_grad_target(self, xT: torch.Tensor, step: int) -> TerminalGradOutput:
        ...


class TargetSamplerLike(Protocol):
    """Minimal target-sampler interface used by providers and evaluators."""

    N: int

    def sample(self, batch_size: int) -> torch.Tensor:
        ...


class ScoreProvider(Protocol):
    """Contract for trainable score backends used by KL providers."""

    def to(self, device: Union[str, torch.device]):
        ...

    def train(self, mode: bool = True):
        ...

    def eval(self):
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        ...

    def set_trainable(self, trainable: bool) -> None:
        ...

    def on_train_start(self, device: Union[str, torch.device], train_cfg: object) -> None:
        ...

    def compute_score(
        self,
        x: Optional[torch.Tensor],
        step: int,
        target_sampler: Optional[TargetSamplerLike] = None,
        device: Union[str, torch.device] = "cpu",
        force: bool = False,
    ) -> ScoreComputeOutput:
        ...
