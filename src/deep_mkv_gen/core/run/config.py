"""Configuration objects for staged training runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..mkv_eval import StageRecord


@dataclass
class EvalConfig:
    """Evaluation settings used at each stage boundary."""

    eval_kl_source: str = "batch"
    eval_kl_n: int = 2048
    eval_n: int = 10_000
    sw2_proj: int = 512
    eval_w2_repeats: int = 1

    def validate(self) -> None:
        if self.eval_kl_source not in ("batch", "sample"):
            raise ValueError("eval_kl_source must be one of {'batch', 'sample'}")
        if self.eval_n < 1:
            raise ValueError("eval_n must be >= 1")
        if self.eval_kl_n < 2:
            raise ValueError("eval_kl_n must be >= 2")
        if self.sw2_proj < 1:
            raise ValueError("sw2_proj must be >= 1")
        if self.eval_w2_repeats < 1:
            raise ValueError("eval_w2_repeats must be >= 1")


@dataclass
class CheckpointConfig:
    """Best-model tracking and optional checkpoint persistence."""

    save_best_checkpoint: bool = False
    keep_best_in_memory: bool = True
    restore_best_at_end: bool = True
    best_metric: str = "w2"
    outdir: Optional[Path] = None
    best_checkpoint_path: Optional[Path] = None

    def validate(self) -> None:
        if self.best_metric not in ("w2", "sw2", "kl"):
            raise ValueError("best_metric must be one of {'w2', 'sw2', 'kl'}")
        if self.save_best_checkpoint and self.outdir is None and self.best_checkpoint_path is None:
            raise ValueError("checkpoint outdir or best_checkpoint_path is required when save_best_checkpoint=True")


@dataclass
class KConfig:
    """Scalar k used throughout a run."""

    k: float = 6.0

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be > 0")


@dataclass
class RunHooks:
    """Run-level callbacks."""

    on_stage_record: Optional[Callable[[StageRecord], None]] = None


@dataclass
class RunConfig:
    """Top-level configuration for ``fit``."""

    device: str
    train_seed: int
    data_seed: int
    total_steps: int
    eval_every: int
    k_config: KConfig
    tag: Optional[str] = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    mode: str = "continuous"
    verbose: bool = False
    model_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 1024
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    hooks: RunHooks = field(default_factory=RunHooks)

    def validate(self) -> None:
        if self.mode not in ("continuous", "refinement"):
            raise ValueError("mode must be one of {'continuous', 'refinement'}")
        if self.model_lr <= 0:
            raise ValueError("model_lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        self.eval.validate()
        self.checkpoint.validate()
        self.k_config.validate()


def resolve_run_tag(cfg: RunConfig) -> str:
    """Return the printable run tag."""

    return str(cfg.tag) if cfg.tag is not None else "run"
