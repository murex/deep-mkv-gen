"""Low-level training loop for the deep MKV model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .protocols import TerminalGradProvider


@dataclass
class TrainConfig:
    """Optimizer and batching settings for one training phase."""

    num_steps: int = 6_000
    batch_size: int = 512
    k: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 100

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")


@dataclass
class TrainRuntime:
    """Mutable state carried across training phases."""

    optimizer: Optional[torch.optim.Optimizer] = None
    provider_initialized: bool = False
    global_step: int = 0
    phase_step: int = 0
    warned_odd_batch_sizes: set[int] = field(default_factory=set)


@dataclass
class TrainEvalEvent:
    """Payload emitted when the low-level loop reaches an eval boundary."""

    global_step: int
    phase_step: int
    logs: List[Dict[str, float]]
    cfg: TrainConfig
    runtime: TrainRuntime


@dataclass
class TrainHooks:
    """Optional callbacks for the low-level training loop."""

    on_eval: Optional[Callable[[TrainEvalEvent], None]] = None


def _sample_antithetic_x0(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size % 2 == 0:
        return model._sample_source(batch_size, device=device)
    if batch_size == 1:
        # Degenerate case: duplicate the sample so we can still form one
        # antithetic pair instead of failing on batch_size=1.
        x0 = model._sample_source(1, device=device)
        return torch.cat([x0, x0], dim=0)
    return model._sample_source(batch_size - 1, device=device)


def _make_train_log_record(
    *,
    global_step: int,
    cfg: TrainConfig,
    x0: torch.Tensor,
    info: Dict[str, torch.Tensor],
    provider_metrics: Dict[str, float],
    loss: torch.Tensor,
    preclip_norm: Union[torch.Tensor, float],
) -> Dict[str, float]:
    preclip_norm_f = float(preclip_norm.item() if torch.is_tensor(preclip_norm) else preclip_norm)
    rec = {
        "step": float(global_step),
        "k": float(cfg.k),
        "effective_batch_size": float(x0.shape[0]),
        "lr": float(cfg.lr),
        "loss": float(loss.item()),
        "energy": float(info["energy"].item()),
        "controller_grad_preclip_norm": preclip_norm_f,
        "controller_grad_clip_frac": float(preclip_norm_f > float(cfg.grad_clip)),
    }
    for mk in (
        "drift_clip_frac",
        "drift_clip_scale_mean",
        "drift_pre_norm_mean",
        "drift_post_norm_mean",
    ):
        if mk in info:
            rec[mk] = float(info[mk].item())
    for mk, mv in provider_metrics.items():
        rec[mk] = float(mv)
    return rec


def _maybe_warn_odd_batch_size(cfg: TrainConfig, runtime: TrainRuntime) -> None:
    if cfg.batch_size % 2 == 0 or cfg.batch_size in runtime.warned_odd_batch_sizes:
        return
    adjusted_batch_size = 2 if cfg.batch_size == 1 else cfg.batch_size - 1
    print(
        f"[train] antithetic low-level training adjusted odd batch_size={cfg.batch_size} "
        f"to effective_batch_size={adjusted_batch_size}"
    )
    runtime.warned_odd_batch_sizes.add(int(cfg.batch_size))


def _sync_optimizer_hparams(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(cfg.lr)
        group["weight_decay"] = float(cfg.weight_decay)


def _resolve_eval_every(
    eval_every: Optional[Union[int, Callable[[], int]]],
) -> Optional[int]:
    if eval_every is None:
        return None
    value = int(eval_every() if callable(eval_every) else eval_every)
    if value < 1:
        raise ValueError("eval_every must be >= 1")
    return value


def _initialize_runtime(
    *,
    model: nn.Module,
    cfg: TrainConfig,
    grad_provider: TerminalGradProvider,
    device: Union[str, torch.device],
    runtime: TrainRuntime,
    reset_state: bool,
) -> torch.device:
    cfg.validate()
    device_t = torch.device(device)
    model.to(device_t)
    model.train()

    if reset_state:
        runtime.phase_step = 0
        runtime.optimizer = None
        runtime.provider_initialized = False

    if runtime.optimizer is None:
        runtime.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    _sync_optimizer_hparams(runtime.optimizer, cfg)

    if not runtime.provider_initialized:
        grad_provider.on_train_start(device=device_t, train_cfg=cfg)
        runtime.provider_initialized = True

    return device_t


def train(
    model: nn.Module,
    cfg: TrainConfig,
    grad_provider: TerminalGradProvider,
    device: Union[str, torch.device] = "cpu",
    *,
    runtime: Optional[TrainRuntime] = None,
    hooks: Optional[TrainHooks] = None,
    eval_every: Optional[Union[int, Callable[[], int]]] = None,
    reset_state: bool = True,
) -> List[Dict[str, float]]:
    """Train the model against a terminal-gradient provider."""

    runtime = runtime or TrainRuntime()
    hooks = hooks or TrainHooks()
    device_t = _initialize_runtime(
        model=model,
        cfg=cfg,
        grad_provider=grad_provider,
        device=device,
        runtime=runtime,
        reset_state=reset_state,
    )

    logs: List[Dict[str, float]] = []
    interval_logs: List[Dict[str, float]] = []
    for step in range(cfg.num_steps):
        cfg.validate()
        _maybe_warn_odd_batch_size(cfg, runtime)
        if runtime.optimizer is None:
            raise RuntimeError("Training runtime optimizer was not initialized.")
        _sync_optimizer_hparams(runtime.optimizer, cfg)

        x0 = _sample_antithetic_x0(model, cfg.batch_size, device=device_t)
        xT, yT, info = model._simulate(x0, antithetic=True)

        provider_out = grad_provider.get_grad_target(xT=xT, step=runtime.phase_step)
        score = provider_out.grad_target
        loss = ((yT / cfg.k) - score).pow(2).sum(dim=-1).mean()

        runtime.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        runtime.optimizer.step()

        runtime.global_step += 1
        runtime.phase_step += 1

        step_rec = _make_train_log_record(
            global_step=runtime.global_step,
            cfg=cfg,
            x0=x0,
            info=info,
            provider_metrics=provider_out.metrics,
            loss=loss,
            preclip_norm=preclip_norm,
        )
        interval_logs.append(step_rec)

        should_log = (runtime.global_step % cfg.log_every == 0) or (step == 0)
        if should_log:
            logs.append(step_rec)

        current_eval_every = _resolve_eval_every(eval_every)
        should_eval = False
        if current_eval_every is not None:
            should_eval = (runtime.global_step % current_eval_every == 0) or (step == cfg.num_steps - 1)

        if should_eval and hooks.on_eval is not None:
            hooks.on_eval(
                TrainEvalEvent(
                    global_step=int(runtime.global_step),
                    phase_step=int(runtime.phase_step),
                    logs=list(interval_logs),
                    cfg=cfg,
                    runtime=runtime,
                )
            )
            interval_logs.clear()

    return logs
