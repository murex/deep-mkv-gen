"""Best-model tracking and optional checkpoint persistence."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..mkv_gen import DeepMKVGen
from ..mkv_eval import StageRecord
from .config import CheckpointConfig, RunConfig, resolve_run_tag
from .state import RunState


def clone_model_state_dict(model: DeepMKVGen) -> dict[str, torch.Tensor]:
    """Clone the model state dict onto CPU for safe storage."""

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def resolve_best_checkpoint_path(cfg: RunConfig, ckpt_cfg: CheckpointConfig) -> Path:
    """Resolve the path used for the best checkpoint file."""

    if ckpt_cfg.best_checkpoint_path is not None:
        return ckpt_cfg.best_checkpoint_path
    if ckpt_cfg.outdir is None:
        raise ValueError("checkpoint outdir is required when no best_checkpoint_path is provided")
    return ckpt_cfg.outdir / "checkpoints" / f"{resolve_run_tag(cfg)}_best_{ckpt_cfg.best_metric}.pt"


def metric_value(
    metric_name: str,
    *,
    w2_val: float,
    sw2_val: float,
    kl_val: float,
) -> float:
    """Select the scalar value used for best-model comparison."""

    if metric_name == "w2":
        return float(w2_val)
    if metric_name == "sw2":
        return float(sw2_val)
    return float(kl_val)


def maybe_save_best_checkpoint(
    *,
    cfg: RunConfig,
    state: RunState,
    model: DeepMKVGen,
    record: StageRecord,
) -> bool:
    """Update the best-model trackers and optionally persist a checkpoint."""

    ckpt_cfg = cfg.checkpoint
    if state.checkpoint_metric_name != ckpt_cfg.best_metric:
        state.checkpoint_metric_name = ckpt_cfg.best_metric
        state.best_checkpoint_metric_value = float("inf")
        state.best_checkpoint_path = None
        state.best_checkpoint_stage = None
        state.best_checkpoint_step = None
        state.best_model_state_dict = None

    if not ckpt_cfg.save_best_checkpoint and not ckpt_cfg.keep_best_in_memory:
        return False

    value = metric_value(
        ckpt_cfg.best_metric,
        w2_val=record.w2,
        sw2_val=record.sw2_sq,
        kl_val=record.kl,
    )
    if not math.isfinite(value):
        return False
    if value >= state.best_checkpoint_metric_value - 1e-12:
        return False

    if ckpt_cfg.keep_best_in_memory:
        state.best_model_state_dict = clone_model_state_dict(model)

    if ckpt_cfg.save_best_checkpoint:
        best_ckpt_path = resolve_best_checkpoint_path(cfg, ckpt_cfg)
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tag": resolve_run_tag(cfg),
                "stage": int(record.stage),
                "step": int(record.steps_done),
                "k": float(record.k),
                "w2_mean": float(record.w2),
                "w2_se": float(record.w2_se),
                "sw2_sq": float(record.sw2_sq),
                "kl": float(record.kl),
                "checkpoint_metric": ckpt_cfg.best_metric,
                "checkpoint_metric_value": float(value),
                "train_seed": int(cfg.train_seed),
                "data_seed": int(cfg.data_seed),
                "model_state_dict": model.state_dict(),
            },
            best_ckpt_path,
        )
        state.best_checkpoint_path = best_ckpt_path
    else:
        state.best_checkpoint_path = None

    state.best_checkpoint_metric_value = float(value)
    state.best_checkpoint_stage = int(record.stage)
    state.best_checkpoint_step = int(record.steps_done)
    return True


def maybe_restore_best_model(cfg: RunConfig, state: RunState, model: DeepMKVGen) -> bool:
    """Restore the best in-memory model state when configured."""

    ckpt_cfg = cfg.checkpoint
    if not ckpt_cfg.restore_best_at_end:
        return False
    if state.best_model_state_dict is None:
        return False
    model.load_state_dict(state.best_model_state_dict)
    return True
