from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from deep_mkv_gen.core import (
    CheckpointConfig,
    DeepMKVGenConfig,
    EvalConfig,
    KConfig,
    DeepMKVGen,
    RunConfig,
    SourceSampler,
    StageRecord,
    TargetSampler,
    TerminalGradOutput,
    TrainConfig,
)

_AUTO_TAG = object()


def seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    random.seed(int(seed))


def numpy_state_equal(left: tuple, right: tuple) -> bool:
    if left[0] != right[0]:
        return False
    if not np.array_equal(left[1], right[1]):
        return False
    return left[2:] == right[2:]


def empirical_sampler(samples: torch.Tensor) -> Callable[[int, torch.device | None], torch.Tensor]:
    def _sampler(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        idx = torch.randint(0, int(samples.shape[0]), (int(batch_size),))
        out = samples[idx]
        return out.to(device) if device is not None else out

    return _sampler


def indexed_empirical_sampler(
    samples: torch.Tensor,
) -> Callable[[int, torch.device | None], tuple[torch.Tensor, torch.Tensor]]:
    def _sampler(batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, int(samples.shape[0]), (int(batch_size),))
        out = samples[idx]
        if device is not None:
            out = out.to(device)
            idx = idx.to(device)
        return out, idx

    return _sampler


def make_source_sampler(source_samples: torch.Tensor) -> SourceSampler:
    return SourceSampler(source_samples)


def make_source_target_samples(
    *,
    n_source: int = 64,
    n_target: int = 64,
    dim: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    source = torch.randn(n_source, dim, generator=gen)
    target = torch.randn(n_target, dim, generator=gen) + 1.0
    return source, target


def make_empirical_model(
    *,
    source_samples: torch.Tensor | None = None,
    dim: int = 2,
    T: float = 1.0,
    N: int = 4,
    sigma: float = 0.3,
    drift_clip: float | None = None,
    indexed: bool = False,
) -> DeepMKVGen:
    if source_samples is not None:
        dim = int(source_samples.shape[1])
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=dim,
            T=T,
            N=N,
            sigma=sigma,
            drift_clip=drift_clip,
            z_hidden_dim=8,
            z_time_embed_dim=8,
            z_num_layers=1,
            z_structure="diag",
            z_time_encoding="scalar",
        ),
    )
    if source_samples is not None:
        model._set_source(source_samples=source_samples)
    return model


class ZeroGradProvider:
    def __init__(self) -> None:
        self.on_train_start_calls = 0
        self.seen_steps: list[int] = []
        self.target_sampler = None

    def set_target_sampler(self, target_sampler) -> None:
        self.target_sampler = target_sampler

    def on_train_start(self, device, train_cfg) -> None:
        self.on_train_start_calls += 1
        self.device = device
        self.last_train_cfg = train_cfg

    def get_grad_target(self, xT: torch.Tensor, step: int) -> TerminalGradOutput:
        self.seen_steps.append(int(step))
        grad_norm = float(torch.linalg.norm(xT, dim=-1).mean().item()) if xT.numel() > 0 else 0.0
        metrics = {
            "classifier_acc": 0.5,
            "classifier_loss": 1.0,
            "grad_norm_mean": grad_norm,
            "target_grad_clip_frac": 0.0,
            "target_grad_preclip_norm_mean": grad_norm,
        }
        return TerminalGradOutput(grad_target=torch.zeros_like(xT), metrics=metrics)


class ToyTrainModel(nn.Module):
    def __init__(self, dim: int = 2, initial_weight: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.weight = nn.Parameter(torch.full((self.dim,), float(initial_weight)))
        self.batch_history: list[int] = []
        self.source_samples: torch.Tensor | None = None
        self.source_sampler = None

    def _set_source(
        self,
        *,
        source_samples: torch.Tensor | None = None,
        source_sampler=None,
    ) -> None:
        if (source_samples is None) == (source_sampler is None):
            raise ValueError("Provide exactly one of source_samples or source_sampler.")
        if source_samples is not None:
            self.source_samples = source_samples

            def _sampler(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
                idx = torch.randint(0, int(source_samples.shape[0]), (int(batch_size),))
                out = source_samples[idx]
                return out.to(device) if device is not None else out

            self.source_sampler = _sampler
            return

        if not callable(source_sampler) and not (
            hasattr(source_sampler, "sample") and callable(getattr(source_sampler, "sample"))
        ):
            raise TypeError("source_sampler must be callable or define sample(batch_size, device=None)")
        self.source_samples = None
        self.source_sampler = source_sampler

    def _sample_source(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        self.batch_history.append(int(batch_size))
        if self.source_sampler is not None:
            if hasattr(self.source_sampler, "sample") and callable(getattr(self.source_sampler, "sample")):
                out = self.source_sampler.sample(int(batch_size), device=device)
            else:
                out = self.source_sampler(int(batch_size), device)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out
        return torch.ones(int(batch_size), self.dim, device=device)

    def _simulate(self, x0: torch.Tensor, antithetic: bool = False):
        del antithetic
        xT = x0 * self.weight[None, :]
        yT = self.weight[None, :].expand_as(x0)
        zero = torch.zeros((), device=x0.device)
        one = torch.ones((), device=x0.device)
        info = {
            "energy": 0.5 * self.weight.pow(2).sum(),
            "drift_clip_frac": zero,
            "drift_clip_scale_mean": one,
            "drift_pre_norm_mean": torch.linalg.norm(self.weight),
            "drift_post_norm_mean": torch.linalg.norm(self.weight),
        }
        return xT, yT, info

def make_run_config(
    *,
    outdir: Path,
    mode: str = "continuous",
    total_steps: int = 4,
    eval_every: int = 2,
    best_metric: str = "w2",
    save_best_checkpoint: bool = False,
    tag: object = _AUTO_TAG,
) -> RunConfig:
    resolved_tag = f"test-{mode}" if tag is _AUTO_TAG else tag
    return RunConfig(
        device="cpu",
        tag=resolved_tag,
        train_seed=123,
        data_seed=456,
        total_steps=int(total_steps),
        eval_every=int(eval_every),
        eval=EvalConfig(
            eval_kl_source="batch",
            eval_kl_n=8,
            eval_n=8,
            sw2_proj=8,
            eval_w2_repeats=1,
        ),
        k_config=KConfig(k=1.0),
        mode=mode,
        model_lr=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        batch_size=4,
        checkpoint=CheckpointConfig(
            save_best_checkpoint=bool(save_best_checkpoint),
            best_metric=best_metric,
            outdir=outdir,
        ),
    )


def make_stage_record(
    *,
    stage: int = 1,
    steps_done: int = 1,
    k: float = 1.0,
    w2: float = 1.0,
    sw2_sq: float = 1.0,
    kl: float = 1.0,
) -> StageRecord:
    return StageRecord(
        stage=int(stage),
        k=float(k),
        steps_done=int(steps_done),
        mean_grad_norm=1.0,
        mean_target_grad_clip_frac=0.0,
        mean_target_grad_preclip_norm=1.0,
        mean_drift_clip_frac=0.0,
        mean_drift_clip_scale=1.0,
        w2=float(w2),
        w2_std=0.0,
        w2_se=0.0,
        sw2_sq=float(sw2_sq),
        sw2_sq_std=0.0,
        kl=float(kl),
        saved_best_checkpoint=False,
        provider_metrics={"classifier_acc": 0.5, "classifier_loss": 1.0},
    )


def make_small_train_cfg(*, num_steps: int = 2, batch_size: int = 4) -> TrainConfig:
    return TrainConfig(
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        k=1.0,
        lr=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        log_every=10,
    )


def make_target_sampler(target_samples: torch.Tensor) -> TargetSampler:
    return TargetSampler(target_samples)


def make_regression_model(source_samples: torch.Tensor | None = None) -> DeepMKVGen:
    model = make_empirical_model(source_samples=source_samples, dim=2, N=4, sigma=0.2)
    with torch.no_grad():
        model.y0.fill_(0.0)
    return model


def make_provider_spec_case(kind: str):
    if kind == "kl_ratio":
        from deep_mkv_gen.providers import KLProviderSpec, KLRatioScoreConfig

        return KLProviderSpec(
            cfg=KLRatioScoreConfig(
                hidden_dim=32,
                num_layers=2,
                classifier_updates_per_step=3,
                classifier_lr=5e-4,
                use_replay=False,
                replay_batch_size=16,
                target_grad_norm_clip=1.0,
            ),
            dim=2,
        )

    if kind == "kl_difference":
        from deep_mkv_gen.providers import (
            DSMScoreConfig,
            KLProviderSpec,
            KLScoreDifferenceConfig,
        )

        return KLProviderSpec(
            cfg=KLScoreDifferenceConfig(
                score_mu_updates_per_step=2,
                target_grad_norm_clip=1.0,
                target_score_cfg=DSMScoreConfig(
                    hidden_dim=16,
                    num_layers=2,
                    lr=5e-4,
                    sigma_noise=0.2,
                    calibration_epochs=2,
                    batch_size=16,
                ),
                mu_score_cfg=DSMScoreConfig(
                    hidden_dim=16,
                    num_layers=2,
                    lr=5e-4,
                    sigma_noise=0.2,
                    calibration_epochs=0,
                    batch_size=16,
                ),
            ),
            dim=2,
        )

    if kind == "w2":
        from deep_mkv_gen.providers import GeomLossW2ProviderConfig, W2ProviderSpec

        return W2ProviderSpec(
            cfg=GeomLossW2ProviderConfig(
                blur=0.5,
                scaling=0.9,
                backend="tensorized",
                target_draws_per_step=2,
                target_grad_norm_clip=1.0,
            ),
            dim=2,
        )

    if kind == "hybrid_ratio":
        from deep_mkv_gen.providers import (
            GeomLossW2ProviderConfig,
            HybridProviderConfig,
            HybridProviderSpec,
            KLRatioScoreConfig,
        )

        return HybridProviderSpec(
            cfg=HybridProviderConfig(
                kl_weight=1.0,
                w2_weight=1.0,
                kl_cfg=KLRatioScoreConfig(
                    hidden_dim=32,
                    num_layers=2,
                    classifier_updates_per_step=3,
                    classifier_lr=5e-4,
                    use_replay=False,
                    replay_batch_size=16,
                    target_grad_norm_clip=1.0,
                ),
                w2_cfg=GeomLossW2ProviderConfig(
                    blur=0.5,
                    scaling=0.9,
                    backend="tensorized",
                    target_draws_per_step=2,
                    target_grad_norm_clip=1.0,
                ),
            ),
            dim=2,
        )

    if kind == "hybrid_difference":
        from deep_mkv_gen.providers import (
            DSMScoreConfig,
            GeomLossW2ProviderConfig,
            HybridProviderConfig,
            HybridProviderSpec,
            KLScoreDifferenceConfig,
        )

        return HybridProviderSpec(
            cfg=HybridProviderConfig(
                kl_weight=1.0,
                w2_weight=1.0,
                kl_cfg=KLScoreDifferenceConfig(
                    score_mu_updates_per_step=2,
                    target_grad_norm_clip=1.0,
                    target_score_cfg=DSMScoreConfig(
                        hidden_dim=16,
                        num_layers=2,
                        lr=5e-4,
                        sigma_noise=0.2,
                        calibration_epochs=2,
                        batch_size=16,
                    ),
                    mu_score_cfg=DSMScoreConfig(
                        hidden_dim=16,
                        num_layers=2,
                        lr=5e-4,
                        sigma_noise=0.2,
                        calibration_epochs=0,
                        batch_size=16,
                    ),
                ),
                w2_cfg=GeomLossW2ProviderConfig(
                    blur=0.5,
                    scaling=0.9,
                    backend="tensorized",
                    target_draws_per_step=2,
                    target_grad_norm_clip=1.0,
                ),
            ),
            dim=2,
        )

    raise ValueError(f"Unknown provider case: {kind}")
