"""Stage evaluation and record construction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..metrics import estimate_kl_knn_from_samples
from ..mkv_gen import DeepMKVGen
from ..mkv_eval import (
    StageRecord,
    isolated_seed,
    make_target_eval_sets,
    mean_of,
    transport_eval_stats,
)
from ..protocols import TargetSamplerLike
from ..runtime_utils import preserve_model_device, preserve_model_mode
from ..sampling import sample
from .config import EvalConfig, RunConfig


GENERIC_TRAIN_LOG_KEYS = {
    "step",
    "k",
    "effective_batch_size",
    "lr",
    "loss",
    "energy",
    "controller_grad_preclip_norm",
    "controller_grad_clip_frac",
    "drift_clip_frac",
    "drift_clip_scale_mean",
    "drift_pre_norm_mean",
    "drift_post_norm_mean",
    "grad_norm_mean",
    "target_grad_clip_frac",
    "target_grad_preclip_norm_mean",
    "target_grad_postclip_norm_mean",
    "target_grad_scale_mean",
    "provider_step",
}


def count_metric_entries(logs: List[Dict[str, float]], key: str) -> int:
    """Count how many log records contain a metric key."""

    return sum(1 for rec in logs if key in rec)


def sum_metric(logs: List[Dict[str, float]], key: str) -> float:
    """Sum a metric across log records."""

    total = 0.0
    for rec in logs:
        if key in rec:
            total += float(rec[key])
    return total


def collect_provider_metric_means(logs: List[Dict[str, float]]) -> Dict[str, float]:
    """Average non-generic training metrics into provider_metrics."""

    provider_keys = sorted({key for rec in logs for key in rec if key not in GENERIC_TRAIN_LOG_KEYS})
    return {key: mean_of(logs, key) for key in provider_keys}


def make_stage_record(
    *,
    stage: int,
    k: float,
    steps_done: int,
    train_logs: List[Dict[str, float]],
    w2_val: float,
    w2_std: float,
    w2_se: float,
    sw2_val: float,
    sw2_std: float,
    kl: float,
) -> StageRecord:
    """Build a StageRecord from training logs and evaluation metrics."""

    cclip_total = count_metric_entries(train_logs, "controller_grad_clip_frac")
    cclip_count = int(round(sum_metric(train_logs, "controller_grad_clip_frac")))
    return StageRecord(
        stage=int(stage),
        k=float(k),
        steps_done=int(steps_done),
        mean_grad_norm=mean_of(train_logs, "grad_norm_mean"),
        mean_target_grad_clip_frac=mean_of(train_logs, "target_grad_clip_frac"),
        mean_target_grad_preclip_norm=mean_of(train_logs, "target_grad_preclip_norm_mean"),
        mean_drift_clip_frac=mean_of(train_logs, "drift_clip_frac"),
        mean_drift_clip_scale=mean_of(train_logs, "drift_clip_scale_mean"),
        w2=float(w2_val),
        w2_std=float(w2_std),
        w2_se=float(w2_se),
        sw2_sq=float(sw2_val),
        sw2_sq_std=float(sw2_std),
        kl=float(kl),
        saved_best_checkpoint=False,
        mean_controller_grad_clip_frac=mean_of(train_logs, "controller_grad_clip_frac"),
        mean_controller_grad_preclip_norm=mean_of(train_logs, "controller_grad_preclip_norm"),
        controller_grad_clip_count=int(cclip_count),
        controller_grad_clip_total=int(cclip_total),
        provider_metrics=collect_provider_metric_means(train_logs),
    )


class StageEvaluator:
    """Evaluate the current model and turn results into StageRecord values."""

    def __init__(
        self,
        *,
        cfg: RunConfig,
        model: DeepMKVGen,
        target_sampler: TargetSamplerLike,
        target_samples: torch.Tensor | None = None,
        source_sampler: object | None = None,
        source_samples: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.target_sampler = target_sampler
        self.target_samples = target_samples
        self.source_sampler = source_sampler
        self.source_samples = source_samples
        self._target_eval_sets_cache: Dict[Tuple[int, int, int], List[torch.Tensor]] = {}
        self._source_eval_batch_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _source_eval_batch(self, eval_cfg: EvalConfig) -> torch.Tensor:
        cache_key = (int(self.cfg.data_seed), int(eval_cfg.eval_n))
        if cache_key not in self._source_eval_batch_cache:
            if self.source_samples is None:
                raise RuntimeError("_source_eval_batch requires source_samples.")
            source_eval_gen = torch.Generator(device="cpu")
            source_eval_gen.manual_seed(int(self.cfg.data_seed + 101))
            n = int(self.source_samples.shape[0])
            if int(eval_cfg.eval_n) <= n:
                idx = torch.randperm(n, generator=source_eval_gen)[: int(eval_cfg.eval_n)]
            else:
                idx = torch.randint(0, n, (int(eval_cfg.eval_n),), generator=source_eval_gen)
            self._source_eval_batch_cache[cache_key] = self.source_samples[idx].clone()
        return self._source_eval_batch_cache[cache_key].clone()

    def _draw_source_batch(self, *, num_samples: int, seed: int) -> torch.Tensor:
        if self.source_samples is not None:
            return self._source_eval_batch(self.cfg.eval)
        if self.source_sampler is None:
            raise RuntimeError("_draw_source_batch requires source_samples or source_sampler.")

        with isolated_seed(self.cfg.device, int(seed)):
            if hasattr(self.source_sampler, "sample") and callable(getattr(self.source_sampler, "sample")):
                batch = self.source_sampler.sample(int(num_samples), device=torch.device(self.cfg.device))
            else:
                batch = self.source_sampler(int(num_samples), torch.device(self.cfg.device))
        if isinstance(batch, (tuple, list)):
            if len(batch) < 1:
                raise ValueError("source_sampler tuple/list output must contain x0 as the first element.")
            batch = batch[0]
        if not torch.is_tensor(batch):
            raise TypeError("source_sampler must return a torch.Tensor or (x0, idx).")
        if batch.ndim != 2 or int(batch.shape[1]) != int(self.model.dim):
            raise ValueError(f"source_sampler must return shape (n,{int(self.model.dim)}).")
        return batch.detach().cpu().clone()

    def _draw_target_batch(self, *, num_samples: int, seed: int) -> torch.Tensor:
        with isolated_seed(self.cfg.device, int(seed)):
            batch = self.target_sampler.sample(int(num_samples))
        if not torch.is_tensor(batch):
            raise TypeError("target_sampler.sample(...) must return a torch.Tensor")
        return batch.detach().cpu().clone()

    def _target_eval_sets(self, eval_cfg: EvalConfig, *, step_id: int) -> List[torch.Tensor]:
        if self.target_samples is None:
            return [
                self._draw_target_batch(
                    num_samples=int(eval_cfg.eval_n),
                    seed=int(self.cfg.data_seed + 103 + 1_000_003 * int(step_id) + i),
                )
                for i in range(int(eval_cfg.eval_w2_repeats))
            ]
        cache_key = (int(self.cfg.data_seed), int(eval_cfg.eval_n), int(eval_cfg.eval_w2_repeats))
        if cache_key not in self._target_eval_sets_cache:
            target_eval_gen = torch.Generator(device="cpu")
            target_eval_gen.manual_seed(int(self.cfg.data_seed + 103))
            self._target_eval_sets_cache[cache_key] = make_target_eval_sets(
                target_samples=self.target_samples,
                eval_n=int(eval_cfg.eval_n),
                repeats=int(eval_cfg.eval_w2_repeats),
                generator=target_eval_gen,
            )
        return self._target_eval_sets_cache[cache_key]

    def _transport_eval_source(self, x0: torch.Tensor, *, seed: int) -> torch.Tensor:
        with isolated_seed(self.cfg.device, int(seed)):
            with preserve_model_mode(self.model), preserve_model_device(self.model):
                self.model.to(self.cfg.device)
                self.model.eval()
                xT, _, _ = self.model._simulate(x0.to(self.cfg.device), antithetic=False)
        return xT.detach().cpu()

    def _sample_generated_from_eval_source(self, eval_cfg: EvalConfig, *, steps_done: int) -> torch.Tensor:
        draw_seed = int(self.cfg.train_seed + 2_000_003 * int(steps_done))
        transport_seed = int(self.cfg.train_seed + 2_000_029 * int(steps_done))
        x0 = self._draw_source_batch(num_samples=int(eval_cfg.eval_n), seed=draw_seed)
        return self._transport_eval_source(x0, seed=transport_seed)

    def _sample_model_for_kl(
        self,
        eval_cfg: EvalConfig,
        *,
        step_id: int,
        x_gen: torch.Tensor,
    ) -> torch.Tensor:
        eval_seed = int(self.cfg.train_seed + 1_000_003 * int(step_id))
        if eval_cfg.eval_kl_source == "batch":
            x_mu = x_gen
            n = int(x_mu.shape[0])
            if 0 < int(eval_cfg.eval_kl_n) < n:
                with isolated_seed(self.cfg.device, eval_seed):
                    idx = torch.randperm(n)[: int(eval_cfg.eval_kl_n)]
                x_mu = x_mu[idx]
            return x_mu

        if self.source_samples is not None or self.source_sampler is not None:
            x0 = self._draw_source_batch(num_samples=int(eval_cfg.eval_kl_n), seed=eval_seed)
            return self._transport_eval_source(x0, seed=int(eval_seed + 17))

        with isolated_seed(self.cfg.device, eval_seed):
            with preserve_model_mode(self.model):
                x_mu_n = sample(
                    self.model,
                    num_samples=int(eval_cfg.eval_kl_n),
                    batch_size=min(2048, int(eval_cfg.eval_kl_n)),
                    device=self.cfg.device,
                ).cpu()
        return x_mu_n

    def _sample_target_for_kl(self, *, num_samples: int, step_id: int) -> torch.Tensor:
        eval_seed = int(self.cfg.data_seed + 1_000_033 * int(step_id))
        if self.target_samples is not None:
            with isolated_seed(self.cfg.device, eval_seed):
                idx = torch.randint(0, int(self.target_samples.shape[0]), (int(num_samples),))
            return self.target_samples[idx].clone()
        return self._draw_target_batch(num_samples=int(num_samples), seed=eval_seed)

    def _estimate_external_kl(self, eval_cfg: EvalConfig, *, step_id: int, x_gen: torch.Tensor) -> float:
        x_mu = self._sample_model_for_kl(eval_cfg, step_id=step_id, x_gen=x_gen)
        n_model = int(x_mu.shape[0])
        if n_model < 2:
            return float("nan")
        x_target = self._sample_target_for_kl(num_samples=min(int(eval_cfg.eval_kl_n), n_model), step_id=step_id)
        n_target = int(x_target.shape[0])
        if n_target < 1:
            return float("nan")
        knn_k = min(5, n_model - 1, n_target)
        if knn_k < 1:
            return float("nan")
        result = estimate_kl_knn_from_samples(
            model_samples=x_mu,
            target_samples=x_target,
            k=int(knn_k),
        )
        return float(result["kl_hat"])

    def _sample_generated(self, eval_cfg: EvalConfig, *, steps_done: int) -> torch.Tensor:
        if self.source_samples is not None or self.source_sampler is not None:
            return self._sample_generated_from_eval_source(eval_cfg, steps_done=steps_done)
        with isolated_seed(self.cfg.device, int(self.cfg.train_seed + 2_000_003 * int(steps_done))):
            with preserve_model_mode(self.model):
                x_gen_n = sample(
                    self.model,
                    num_samples=int(eval_cfg.eval_n),
                    batch_size=min(2048, int(eval_cfg.eval_n)),
                    device=self.cfg.device,
                ).cpu()
        return x_gen_n

    def evaluate_stage(
        self,
        *,
        stage: int,
        steps_done: int,
        train_logs: List[Dict[str, float]],
    ) -> StageRecord:
        """Evaluate one stage boundary and return its aggregated record."""

        eval_cfg = self.cfg.eval
        x_gen_n = self._sample_generated(eval_cfg, steps_done=steps_done)
        tstats = transport_eval_stats(
            x_gen_n,
            self._target_eval_sets(eval_cfg, step_id=steps_done),
            int(eval_cfg.sw2_proj),
            int(self.cfg.train_seed),
        )
        kl = self._estimate_external_kl(eval_cfg, step_id=steps_done, x_gen=x_gen_n)
        return make_stage_record(
            stage=int(stage),
            k=float(self.cfg.k_config.k),
            steps_done=int(steps_done),
            train_logs=train_logs,
            w2_val=float(tstats["w2"]),
            w2_std=float(tstats["w2_std"]),
            w2_se=float(tstats["w2_se"]),
            sw2_val=float(tstats["sw2_sq"]),
            sw2_std=float(tstats["sw2_sq_std"]),
            kl=float(kl),
        )
