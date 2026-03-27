"""Mutable run state and final run result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..mkv_eval import StageRecord


@dataclass
class RunResult:
    """Final records, summary, and best-checkpoint location for a run."""

    records: List[StageRecord]
    summary: Dict[str, Any]
    best_checkpoint_path: Optional[Path]


@dataclass
class RunState:
    """Mutable orchestration state accumulated during a run."""

    stage: int = 0
    steps_done: int = 0
    records: List[StageRecord] = field(default_factory=list)
    best_w2: float = float("inf")
    best_w2_se: float = float("nan")
    best_sw2: float = float("inf")
    best_kl: float = float("inf")
    best_k_w2: Optional[float] = None
    best_stage_w2: Optional[int] = None
    best_step_w2: Optional[int] = None
    checkpoint_metric_name: Optional[str] = None
    best_checkpoint_metric_value: float = float("inf")
    best_checkpoint_path: Optional[Path] = None
    best_checkpoint_stage: Optional[int] = None
    best_checkpoint_step: Optional[int] = None
    best_model_state_dict: Optional[Dict[str, torch.Tensor]] = None
