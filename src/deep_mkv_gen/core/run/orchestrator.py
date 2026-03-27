"""High-level run orchestration for training, evaluation, and checkpoints."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from ...providers.provider_spec import ProviderSpec, build_provider
from ..mkv_eval import StageRecord
from ..mkv_gen import DeepMKVGen
from ..protocols import TargetSamplerLike, TerminalGradProvider
from ..runtime_utils import set_global_seed
from ..sampling import SourceSampler, TargetSampler
from ..mkv_train import TrainConfig, TrainEvalEvent, TrainHooks, TrainRuntime, train
from .checkpoint import maybe_restore_best_model, maybe_save_best_checkpoint
from .config import RunConfig, RunHooks, resolve_run_tag
from .evaluate import StageEvaluator
from .state import RunResult, RunState
from .summary import build_run_summary


def _emit_run_log(cfg: RunConfig, message: str) -> None:
    if cfg.verbose:
        print(message, flush=True)


def _resolve_provider(
    *,
    provider_spec: Optional[ProviderSpec],
    provider: Optional[TerminalGradProvider],
) -> TerminalGradProvider:
    if (provider_spec is None) == (provider is None):
        raise ValueError("Provide exactly one of provider_spec or provider.")
    if provider is not None:
        return provider
    return build_provider(provider_spec)


def _resolve_source_size(
    *,
    source_samples: Optional[torch.Tensor],
    source_sampler: object | None,
) -> int | None:
    if source_samples is not None:
        return int(source_samples.shape[0])
    if source_sampler is not None and hasattr(source_sampler, "N"):
        return int(getattr(source_sampler, "N"))
    return None


def _resolve_target_inputs(
    *,
    target_samples: Optional[torch.Tensor],
    target_sampler: Optional[TargetSamplerLike],
) -> tuple[TargetSamplerLike, torch.Tensor | None, int | None]:
    if (target_samples is None) == (target_sampler is None):
        raise ValueError("Provide exactly one of target_samples or target_sampler.")
    if target_samples is not None:
        return TargetSampler(target_samples), target_samples, int(target_samples.shape[0])
    if not hasattr(target_sampler, "sample") or not callable(getattr(target_sampler, "sample")):
        raise TypeError("target_sampler must define a callable sample(batch_size) method.")
    if not hasattr(target_sampler, "N"):
        raise TypeError("target_sampler must expose N.")
    return target_sampler, None, int(target_sampler.N) if hasattr(target_sampler, "N") else None


def _resolve_optional_source_eval_inputs(
    *,
    source_samples: Optional[torch.Tensor],
    source_sampler: object | None,
) -> tuple[object | None, torch.Tensor | None, int | None]:
    if source_samples is None and source_sampler is None:
        return None, None, None
    if (source_samples is None) == (source_sampler is None):
        raise ValueError("Provide exactly one of source_val_samples or source_val_sampler.")
    if source_samples is not None:
        return SourceSampler(source_samples), source_samples, int(source_samples.shape[0])
    if not callable(source_sampler) and not (
        hasattr(source_sampler, "sample") and callable(getattr(source_sampler, "sample"))
    ):
        raise TypeError("source_val_sampler must be callable or define sample(batch_size, device=None)")
    return source_sampler, None, _resolve_source_size(source_samples=None, source_sampler=source_sampler)


def _resolve_optional_target_eval_inputs(
    *,
    target_samples: Optional[torch.Tensor],
    target_sampler: Optional[TargetSamplerLike],
) -> tuple[TargetSamplerLike | None, torch.Tensor | None, int | None]:
    if target_samples is None and target_sampler is None:
        return None, None, None
    if (target_samples is None) == (target_sampler is None):
        raise ValueError("Provide exactly one of target_val_samples or target_val_sampler.")
    return _resolve_target_inputs(target_samples=target_samples, target_sampler=target_sampler)


def _attach_source_to_model(
    *,
    model: DeepMKVGen,
    source_samples: Optional[torch.Tensor],
    source_sampler: object | None,
) -> int | None:
    if (source_samples is None) == (source_sampler is None):
        raise ValueError("Provide exactly one of source_samples or source_sampler.")
    if not hasattr(model, "_set_source"):
        raise TypeError("fit(...) requires model._set_source(...) to configure the source distribution.")
    setter = getattr(model, "_set_source")
    if source_samples is not None:
        setter(source_samples=source_samples)
    else:
        setter(source_sampler=source_sampler)
    return _resolve_source_size(source_samples=source_samples, source_sampler=source_sampler)


def _make_train_cfg(cfg: RunConfig, *, num_steps: int) -> TrainConfig:
    return TrainConfig(
        num_steps=int(num_steps),
        batch_size=int(cfg.batch_size),
        k=float(cfg.k_config.k),
        lr=float(cfg.model_lr),
        weight_decay=float(cfg.weight_decay),
        grad_clip=float(cfg.grad_clip),
        log_every=max(1, int(num_steps)),
    )


def _sync_train_cfg(train_cfg: TrainConfig, cfg: RunConfig) -> None:
    train_cfg.batch_size = int(cfg.batch_size)
    train_cfg.k = float(cfg.k_config.k)
    train_cfg.lr = float(cfg.model_lr)
    train_cfg.weight_decay = float(cfg.weight_decay)
    train_cfg.grad_clip = float(cfg.grad_clip)
    train_cfg.validate()


def _update_best_metric_trackers(state: RunState, record: StageRecord) -> None:
    if record.w2 < state.best_w2 - 1e-12:
        state.best_w2 = float(record.w2)
        state.best_w2_se = float(record.w2_se)
        state.best_k_w2 = float(record.k)
        state.best_stage_w2 = int(record.stage)
        state.best_step_w2 = int(record.steps_done)
    if record.sw2_sq < state.best_sw2 - 1e-12:
        state.best_sw2 = float(record.sw2_sq)
    if record.kl < state.best_kl - 1e-12:
        state.best_kl = float(record.kl)


def _handle_eval_event(
    *,
    cfg: RunConfig,
    state: RunState,
    evaluator: StageEvaluator,
    event: TrainEvalEvent,
    hooks: RunHooks,
    model: DeepMKVGen,
) -> None:
    cfg.validate()
    state.steps_done = int(event.global_step)
    stage_id = int(state.stage + 1)
    record = evaluator.evaluate_stage(
        stage=stage_id,
        steps_done=int(state.steps_done),
        train_logs=event.logs,
    )
    _update_best_metric_trackers(state, record)
    record.saved_best_checkpoint = maybe_save_best_checkpoint(
        cfg=cfg,
        state=state,
        model=model,
        record=record,
    )
    state.stage = int(stage_id)
    state.records.append(record)

    _emit_run_log(
        cfg,
        f"[{resolve_run_tag(cfg)}] mode={cfg.mode} stage={record.stage} step={record.steps_done} k={record.k:.4f} "
        f"W2={record.w2:.4f} SW2^2={record.sw2_sq:.4f} KL={record.kl:.4f} "
        f"best_ckpt={'1' if record.saved_best_checkpoint else '0'} metric={cfg.checkpoint.best_metric}",
    )
    if hooks.on_stage_record is not None:
        hooks.on_stage_record(record)
    _sync_train_cfg(event.cfg, cfg)


def _run_continuous(
    *,
    cfg: RunConfig,
    state: RunState,
    model: DeepMKVGen,
    provider: TerminalGradProvider,
    hooks: RunHooks,
    evaluator: StageEvaluator,
) -> None:
    train_cfg = _make_train_cfg(cfg, num_steps=int(cfg.total_steps))

    def on_eval(event: TrainEvalEvent) -> None:
        _handle_eval_event(
            cfg=cfg,
            state=state,
            evaluator=evaluator,
            event=event,
            hooks=hooks,
            model=model,
        )

    runtime = TrainRuntime(global_step=int(state.steps_done))
    train(
        model=model,
        cfg=train_cfg,
        grad_provider=provider,
        device=cfg.device,
        runtime=runtime,
        hooks=TrainHooks(on_eval=on_eval),
        eval_every=lambda: int(cfg.eval_every),
        reset_state=True,
    )
    state.steps_done = int(runtime.global_step)


def _run_refinement(
    *,
    cfg: RunConfig,
    state: RunState,
    model: DeepMKVGen,
    provider: TerminalGradProvider,
    hooks: RunHooks,
    evaluator: StageEvaluator,
) -> None:
    while state.steps_done < int(cfg.total_steps):
        cfg.validate()
        remaining_steps = int(cfg.total_steps) - int(state.steps_done)
        stage_steps = min(int(cfg.eval_every), remaining_steps)
        if stage_steps < 1:
            break

        train_cfg = _make_train_cfg(cfg, num_steps=stage_steps)

        def on_eval(event: TrainEvalEvent) -> None:
            _handle_eval_event(
                cfg=cfg,
                state=state,
                evaluator=evaluator,
                event=event,
                hooks=hooks,
                model=model,
            )

        runtime = TrainRuntime(global_step=int(state.steps_done))
        train(
            model=model,
            cfg=train_cfg,
            grad_provider=provider,
            device=cfg.device,
            runtime=runtime,
            hooks=TrainHooks(on_eval=on_eval),
            eval_every=int(stage_steps),
            reset_state=True,
        )
        state.steps_done = int(runtime.global_step)


def fit(
    cfg: RunConfig,
    model: DeepMKVGen,
    *,
    provider_spec: Optional[ProviderSpec] = None,
    provider: Optional[TerminalGradProvider] = None,
    source_samples: Optional[torch.Tensor] = None,
    source_sampler: object | None = None,
    target_samples: Optional[torch.Tensor] = None,
    target_sampler: Optional[TargetSamplerLike] = None,
    source_val_samples: Optional[torch.Tensor] = None,
    source_val_sampler: object | None = None,
    target_val_samples: Optional[torch.Tensor] = None,
    target_val_sampler: Optional[TargetSamplerLike] = None,
) -> RunResult:
    """Fit a model with periodic evaluation and best-checkpoint tracking."""

    cfg.validate()
    set_global_seed(int(cfg.train_seed), cfg.device)

    hooks = cfg.hooks
    state = RunState()
    source_size = _attach_source_to_model(
        model=model,
        source_samples=source_samples,
        source_sampler=source_sampler,
    )
    resolved_target_sampler, resolved_target_samples, target_size = _resolve_target_inputs(
        target_samples=target_samples,
        target_sampler=target_sampler,
    )
    resolved_eval_source_sampler, resolved_eval_source_samples, eval_source_size = _resolve_optional_source_eval_inputs(
        source_samples=source_val_samples,
        source_sampler=source_val_sampler,
    )
    resolved_eval_target_sampler, resolved_eval_target_samples, eval_target_size = _resolve_optional_target_eval_inputs(
        target_samples=target_val_samples,
        target_sampler=target_val_sampler,
    )
    provider_obj = _resolve_provider(provider_spec=provider_spec, provider=provider)
    provider_obj.set_target_sampler(resolved_target_sampler)
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=resolved_target_sampler if resolved_eval_target_sampler is None else resolved_eval_target_sampler,
        target_samples=resolved_target_samples if resolved_eval_target_samples is None else resolved_eval_target_samples,
        source_sampler=resolved_eval_source_sampler,
        source_samples=resolved_eval_source_samples,
    )

    _emit_run_log(
        cfg,
        f"[{resolve_run_tag(cfg)}] run mode={cfg.mode}: k={cfg.k_config.k:.4f}, total_steps={cfg.total_steps}, "
        f"eval_every={cfg.eval_every}",
    )

    if cfg.mode == "continuous":
        _run_continuous(
            cfg=cfg,
            state=state,
            model=model,
            provider=provider_obj,
            hooks=hooks,
            evaluator=evaluator,
        )
    else:
        _run_refinement(
            cfg=cfg,
            state=state,
            model=model,
            provider=provider_obj,
            hooks=hooks,
            evaluator=evaluator,
        )

    maybe_restore_best_model(cfg, state, model)

    summary = build_run_summary(
        cfg=cfg,
        state=state,
        source_size=source_size,
        target_size=target_size,
        validation_source_size=eval_source_size,
        validation_target_size=eval_target_size,
    )

    if state.best_checkpoint_path is not None:
        _emit_run_log(
            cfg,
            f"[{resolve_run_tag(cfg)}] best checkpoint: {state.best_checkpoint_path} "
            f"(metric={cfg.checkpoint.best_metric}, stage={state.best_checkpoint_stage}, "
            f"step={state.best_checkpoint_step}, value={state.best_checkpoint_metric_value:.4f})",
        )
    if math.isfinite(state.best_w2):
        best_k = float("nan") if state.best_k_w2 is None else float(state.best_k_w2)
        _emit_run_log(
            cfg,
            f"[{resolve_run_tag(cfg)}] best W2={state.best_w2:.4f} at k={best_k:.4f}, "
            f"stage={state.best_stage_w2}, step={state.best_step_w2}",
        )

    return RunResult(
        records=state.records,
        summary=summary,
        best_checkpoint_path=state.best_checkpoint_path,
    )
