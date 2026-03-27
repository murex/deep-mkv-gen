"""Denoising score-matching backend used by the KL providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn

from ...core.protocols import ScoreComputeOutput, TargetSamplerLike


@dataclass
class DSMScoreConfig:
    """Architecture and optimization settings for a DSM score model."""

    hidden_dim: int = 128
    num_layers: int = 3

    lr: float = 2e-4
    weight_decay: float = 0.0
    sigma_noise: float = 0.2
    grad_clip: Optional[float] = None
    metric_key: str = "score_loss"

    calibration_epochs: int = 0
    batch_size: int = 256
    log_every_epochs: int = 1

    def validate(self) -> None:
        if int(self.hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1")
        if int(self.num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be > 0")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.sigma_noise is None or float(self.sigma_noise) <= 0.0:
            raise ValueError("sigma_noise must be > 0")
        if self.grad_clip is not None and float(self.grad_clip) <= 0.0:
            raise ValueError("grad_clip must be > 0 when provided")
        if int(self.calibration_epochs) < 0:
            raise ValueError("calibration_epochs must be >= 0")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if int(self.log_every_epochs) < 1:
            raise ValueError("log_every_epochs must be >= 1")


class DSMScoreModel(nn.Module):
    """MLP score provider trained with denoising score matching."""

    def __init__(
        self,
        dim: int,
        cfg: Optional[DSMScoreConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else DSMScoreConfig()
        self.cfg.validate()

        self.dim = int(dim)
        h = int(self.cfg.hidden_dim)
        in_dim = int(dim)
        layers: list[nn.Module] = []
        for i in range(max(1, int(self.cfg.num_layers))):
            layers.append(nn.Linear(in_dim if i == 0 else h, h))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(h, int(dim)))
        self.net = nn.Sequential(*layers)

        self._opt: Optional[torch.optim.Optimizer] = None
        self._grad_clip_runtime: Optional[float] = None
        self._trainable_params: List[nn.Parameter] = []

        self._sampler_compute_done = False
        self._sampler_compute_info: Dict[str, float] = {}
        self._sampler: Optional[TargetSamplerLike] = None
        self._has_pretrained_score = False

    def _refresh_cfg(self) -> None:
        self.cfg.validate()

    def score_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def score_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[override]
        incompatible = super().load_state_dict(state_dict, *args, **kwargs)
        if len(state_dict) > 0:
            self._has_pretrained_score = True
            self._sampler_compute_done = False
        return incompatible

    def load_score_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)

    def set_trainable(self, trainable: bool) -> None:
        for p in self.parameters():
            p.requires_grad_(bool(trainable))

    def on_train_start(self, device: Union[str, torch.device], train_cfg: object) -> None:
        """Move the model to device and initialize its optimizer."""

        self._refresh_cfg()
        self.to(device)
        self.train()

        params = [p for p in self.parameters() if p.requires_grad]
        if len(params) == 0:
            self._trainable_params = []
            self._opt = None
            self._grad_clip_runtime = None
            return

        self._trainable_params = params
        self._opt = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        if self.cfg.grad_clip is not None:
            self._grad_clip_runtime = float(self.cfg.grad_clip)
        else:
            gc = getattr(train_cfg, "grad_clip", None)
            self._grad_clip_runtime = float(gc) if gc is not None else None

    def compute_score(
        self,
        x: Optional[torch.Tensor],
        step: int,
        target_sampler: Optional[TargetSamplerLike] = None,
        device: Union[str, torch.device] = "cpu",
        force: bool = False,
    ) -> ScoreComputeOutput:
        """Optionally calibrate the model and then score or train on x."""

        del step
        self._refresh_cfg()
        metrics: Dict[str, float] = {}

        compute_requested = False
        if target_sampler is not None:
            self._sampler = target_sampler
            compute_requested = True
        if force:
            self._sampler_compute_done = False
            compute_requested = True

        if compute_requested and not self._sampler_compute_done:
            if self.cfg.calibration_epochs < 1:
                if not self._has_pretrained_score:
                    raise ValueError(
                        "Target score calibration requested with calibration_epochs=0, "
                        "but score weights are not pretrained. "
                        "Load pretrained score weights or set calibration_epochs > 0."
                    )
                self._sampler_compute_info = {"score_pretrained": 1.0}
                self._sampler_compute_done = True
            else:
                if self._sampler is None:
                    raise ValueError(
                        "Score model needs target_sampler for sampler-based compute. "
                        "Pass target_sampler to compute_score(...)."
                    )
                logs = self._compute_from_sampler(sampler=self._sampler, device=device)
                last_loss = float(logs[-1]["score_compute_loss"]) if len(logs) > 0 else float("nan")
                self._sampler_compute_info = {
                    "score_computed": 1.0,
                    "score_compute_epochs": float(self.cfg.calibration_epochs),
                    "score_compute_last_loss": last_loss,
                }
                self._sampler_compute_done = True
            metrics.update(self._sampler_compute_info)

        if x is not None and self.training and self._opt is not None:
            has_trainable = any(p.requires_grad for p in self.parameters())
            if has_trainable:
                metrics.update(self._train_step(x=x))

        score_value = self.score(x) if x is not None else None
        return ScoreComputeOutput(score=score_value, metrics=metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def _train_step(self, x: torch.Tensor) -> Dict[str, float]:
        if self._opt is None:
            raise RuntimeError("Score model not initialized. Call on_train_start() first.")

        x_det = x.detach()
        loss = self._loss_for_batch(x_det)

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if self._grad_clip_runtime is not None:
            nn.utils.clip_grad_norm_(self._trainable_params, self._grad_clip_runtime)
        self._opt.step()
        self._has_pretrained_score = True
        return {self.cfg.metric_key: float(loss.item())}

    def _compute_from_sampler(
        self,
        sampler: TargetSamplerLike,
        device: Union[str, torch.device],
    ) -> List[Dict[str, float]]:
        """Calibrate the score model from target samples."""

        self.to(device)
        self.train()
        self.set_trainable(True)

        params = [p for p in self.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("Score model has no trainable parameters for sampler-based compute.")

        opt = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        logs: List[Dict[str, float]] = []
        global_step = 0
        for epoch in range(self.cfg.calibration_epochs):
            running_loss = 0.0
            batch_iter = self._epoch_batch_iterator(
                sampler=sampler,
                device=device,
            )
            n_batches = 0
            for x_batch in batch_iter:
                loss = self._loss_for_batch(x_batch.detach())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                opt.step()

                running_loss += float(loss.item())
                global_step += 1
                n_batches += 1

            if n_batches < 1:
                raise RuntimeError("Score compute epoch produced zero batches.")

            rec = {
                "epoch": float(epoch + 1),
                "step": float(global_step),
                "score_compute_loss": float(running_loss / float(n_batches)),
            }
            if (
                ((epoch + 1) % self.cfg.log_every_epochs == 0)
                or (epoch == 0)
                or (epoch + 1 == self.cfg.calibration_epochs)
            ):
                logs.append(rec)
        self._has_pretrained_score = True
        return logs

    def _epoch_batch_iterator(
        self,
        *,
        sampler: TargetSamplerLike,
        device: Union[str, torch.device],
    ) -> Iterator[torch.Tensor]:
        samples = getattr(sampler, "target_samples", None)
        if torch.is_tensor(samples):
            n_samples = int(samples.shape[0])
            if n_samples < 1:
                raise ValueError("sampler.target_samples must contain at least one sample")
            perm = torch.randperm(n_samples, device=samples.device)
            for start in range(0, n_samples, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                yield samples[idx].to(device)
            return

        n_samples = int(getattr(sampler, "N"))
        if n_samples < 1:
            raise ValueError("sampler.N must be >= 1")
        steps_per_epoch = max(1, math.ceil(float(n_samples) / float(self.cfg.batch_size)))
        for _ in range(steps_per_epoch):
            yield sampler.sample(self.cfg.batch_size).to(device)

    def score_matching_loss(self, x_clean: torch.Tensor, sigma_noise: Optional[float]) -> torch.Tensor:
        """Compute the denoising score-matching loss for one batch."""

        if sigma_noise is None or sigma_noise <= 0:
            raise ValueError("sigma_noise must be positive for DSM score training.")
        eps = torch.randn_like(x_clean)
        x_tilde = x_clean + eps * float(sigma_noise)
        pred = self.score(x_tilde)
        target = -eps / float(sigma_noise)
        return 0.5 * (pred - target).pow(2).sum(dim=-1).mean() * (float(sigma_noise) ** 2)

    def _loss_for_batch(self, x_clean: torch.Tensor) -> torch.Tensor:
        return self.score_matching_loss(x_clean, sigma_noise=self.cfg.sigma_noise)
