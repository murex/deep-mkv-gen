"""Final summary construction for completed runs."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict

from .checkpoint import resolve_best_checkpoint_path
from .config import RunConfig, resolve_run_tag
from .state import RunState


def build_run_summary(
    *,
    cfg: RunConfig,
    state: RunState,
    source_size: int | None,
    target_size: int | None,
    validation_source_size: int | None,
    validation_target_size: int | None,
) -> Dict[str, Any]:
    """Build the serializable run summary returned to callers."""

    final_ckpt_cfg = cfg.checkpoint
    resolved_best_checkpoint_path = (
        str(resolve_best_checkpoint_path(cfg, final_ckpt_cfg))
        if final_ckpt_cfg.save_best_checkpoint
        else None
    )
    summary = {
        "tag": resolve_run_tag(cfg),
        "device": cfg.device,
        "mode": cfg.mode,
        "seed": int(cfg.train_seed),
        "train_seed": int(cfg.train_seed),
        "data_seed": int(cfg.data_seed),
        "k": float(cfg.k_config.k),
        "total_steps": int(cfg.total_steps),
        "steps_done": int(state.steps_done),
        "eval_every": int(cfg.eval_every),
        "eval_kl_source": cfg.eval.eval_kl_source,
        "eval_kl_n": int(cfg.eval.eval_kl_n),
        "eval_w2_repeats": int(cfg.eval.eval_w2_repeats),
        "model_lr": float(cfg.model_lr),
        "weight_decay": float(cfg.weight_decay),
        "grad_clip": float(cfg.grad_clip),
        "batch_size": int(cfg.batch_size),
        "best_w2": None if not math.isfinite(state.best_w2) else float(state.best_w2),
        "best_w2_se": float(state.best_w2_se) if math.isfinite(state.best_w2_se) else None,
        "best_k_w2": None if state.best_k_w2 is None else float(state.best_k_w2),
        "best_stage_w2": None if state.best_stage_w2 is None else int(state.best_stage_w2),
        "best_step_w2": None if state.best_step_w2 is None else int(state.best_step_w2),
        "best_sw2_sq": None if not math.isfinite(state.best_sw2) else float(state.best_sw2),
        "best_kl": None if not math.isfinite(state.best_kl) else float(state.best_kl),
        "best_checkpoint_path": None if state.best_checkpoint_path is None else str(state.best_checkpoint_path),
        "save_best_checkpoint": bool(final_ckpt_cfg.save_best_checkpoint),
        "keep_best_in_memory": bool(final_ckpt_cfg.keep_best_in_memory),
        "restore_best_at_end": bool(final_ckpt_cfg.restore_best_at_end),
        "checkpoint_metric": final_ckpt_cfg.best_metric,
        "best_checkpoint_metric_value": (
            None if not math.isfinite(state.best_checkpoint_metric_value)
            else float(state.best_checkpoint_metric_value)
        ),
        "best_checkpoint_stage": None if state.best_checkpoint_stage is None else int(state.best_checkpoint_stage),
        "best_checkpoint_step": None if state.best_checkpoint_step is None else int(state.best_checkpoint_step),
        "resolved_best_checkpoint_path": resolved_best_checkpoint_path,
        "source_size": None if source_size is None else int(source_size),
        "target_size": None if target_size is None else int(target_size),
        "validation_source_size": None if validation_source_size is None else int(validation_source_size),
        "validation_target_size": None if validation_target_size is None else int(validation_target_size),
        "records": [asdict(record) for record in state.records],
    }
    return summary
