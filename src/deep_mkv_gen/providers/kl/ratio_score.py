"""KL terminal gradients from a classifier-based ratio score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from ...core.protocols import TargetSamplerLike, TerminalGradOutput


@dataclass
class KLRatioScoreConfig:
    """Configuration for the classifier-based KL ratio estimator."""

    hidden_dim: int = 256
    num_layers: int = 4

    classifier_lr: float = 1e-3
    classifier_updates_per_step: int = 5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    replay_capacity: int = 32768
    replay_min_size: int = 512
    replay_batch_size: int = 512
    use_replay: bool = True
    replay_recent_frac: float = 0.0
    replay_mix_current: float = 0.0

    uncertainty_is_beta: float = 0.0

    target_grad_norm_clip: Optional[float] = 10.0

    classifier_batch_size: Optional[int] = None

    spectral_norm: bool = False

    def validate(self) -> None:
        if int(self.hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1")
        if int(self.num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if float(self.classifier_lr) <= 0.0:
            raise ValueError("classifier_lr must be > 0")
        if int(self.classifier_updates_per_step) < 1:
            raise ValueError("classifier_updates_per_step must be >= 1")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if float(self.grad_clip) < 0.0:
            raise ValueError("grad_clip must be >= 0")
        if int(self.replay_capacity) < 1:
            raise ValueError("replay_capacity must be >= 1")
        if int(self.replay_min_size) < 1:
            raise ValueError("replay_min_size must be >= 1")
        if int(self.replay_min_size) > int(self.replay_capacity):
            raise ValueError("replay_min_size cannot exceed replay_capacity")
        if int(self.replay_batch_size) < 1:
            raise ValueError("replay_batch_size must be >= 1")
        if self.classifier_batch_size is not None and int(self.classifier_batch_size) < 1:
            raise ValueError("classifier_batch_size must be >= 1 when provided")
        if not (0.0 <= float(self.replay_recent_frac) <= 1.0):
            raise ValueError("replay_recent_frac must be in [0, 1]")
        if not (0.0 <= float(self.replay_mix_current) <= 1.0):
            raise ValueError("replay_mix_current must be in [0, 1]")
        if not (0.0 <= float(self.uncertainty_is_beta) <= 1.0):
            raise ValueError("uncertainty_is_beta must be in [0, 1]")
        if self.target_grad_norm_clip is not None and float(self.target_grad_norm_clip) <= 0.0:
            raise ValueError("target_grad_norm_clip must be > 0 when provided")


class RatioScoreClassifier(nn.Module):
    """MLP binary classifier for the ratio-score estimate."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int, spectral_norm: bool = False):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.extend([linear, nn.SiLU()])
            in_dim = hidden_dim
        final = nn.Linear(hidden_dim, 1)
        if spectral_norm:
            final = nn.utils.spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RatioScoreKLTerminalGradProvider:
    """KL terminal gradient via classifier-based ratio-score estimation."""

    def __init__(
        self,
        cfg: KLRatioScoreConfig,
        dim: int = 2,
    ):
        self.cfg = cfg
        self.target_sampler: Optional[TargetSamplerLike] = None
        self.dim = dim
        self.classifier = RatioScoreClassifier(
            dim,
            cfg.hidden_dim,
            cfg.num_layers,
            spectral_norm=cfg.spectral_norm,
        )
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.device: Optional[torch.device] = None
        self.total_steps: Optional[int] = None
        self._classifier_ready: bool = False

        self._replay: Optional[torch.Tensor] = None
        self._replay_pos: int = 0
        self._replay_full: bool = False

    def set_target_sampler(self, target_sampler: TargetSamplerLike) -> None:
        self.target_sampler = target_sampler

    def on_train_start(self, device: Union[str, torch.device], train_cfg) -> None:
        """Reset optimizer and replay state for a new training phase."""

        self.cfg.validate()
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before on_train_start(...).")
        self.total_steps = getattr(train_cfg, "num_steps", None)
        self.device = torch.device(device)
        self.classifier = self.classifier.to(device)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.cfg.classifier_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self._classifier_ready = False
        self._replay = None
        self._replay_pos = 0
        self._replay_full = False
        if self.cfg.use_replay:
            self._replay = torch.empty(
                self.cfg.replay_capacity, self.dim,
                device="cpu", dtype=torch.float32,
            )

    def _replay_size(self) -> int:
        if not self.cfg.use_replay or self._replay is None:
            return 0
        if self._replay_full:
            return self.cfg.replay_capacity
        return self._replay_pos

    def _add_to_replay(self, xT: torch.Tensor) -> None:
        if not self.cfg.use_replay or self._replay is None:
            return
        data = xT.detach().cpu()
        n = data.shape[0]
        cap = self.cfg.replay_capacity
        if self._replay_pos + n <= cap:
            self._replay[self._replay_pos : self._replay_pos + n] = data
            self._replay_pos += n
        else:
            first = cap - self._replay_pos
            self._replay[self._replay_pos :] = data[:first]
            remaining = n - first
            if remaining > 0:
                self._replay[:remaining] = data[first : first + remaining]
            self._replay_pos = remaining % cap
            self._replay_full = True
        if self._replay_pos >= cap:
            self._replay_pos = self._replay_pos % cap
            self._replay_full = True

    def _sample_replay(self, n: int) -> torch.Tensor:
        size = self._replay_size()
        if size < 1:
            raise RuntimeError("Replay is empty.")
        recent_frac = float(self.cfg.replay_recent_frac)
        if recent_frac <= 0.0:
            idx = torch.randint(0, size, (n,))
            return self._replay[idx].to(self.device)

        recent_frac = min(1.0, max(0.0, recent_frac))
        window = max(1, int(size * recent_frac))
        if not self._replay_full:
            lo = max(0, size - window)
            idx = torch.randint(lo, size, (n,))
            return self._replay[idx].to(self.device)

        cap = self.cfg.replay_capacity
        start = (self._replay_pos - window) % cap
        offs = torch.randint(0, window, (n,))
        idx = (start + offs) % cap
        return self._replay[idx].to(self.device)

    def _sample_current_mu(self, xT_det: torch.Tensor, n: int) -> torch.Tensor:
        bsz = int(xT_det.shape[0])
        if bsz < 1 or n < 1:
            return xT_det[:0].to(self.device)
        if n == bsz:
            return xT_det.to(self.device)

        beta = min(1.0, max(0.0, float(self.cfg.uncertainty_is_beta)))
        if beta <= 0.0:
            if n < bsz:
                idx = torch.randperm(bsz, device=xT_det.device)[:n]
            else:
                idx = torch.randint(0, bsz, (n,), device=xT_det.device)
            return xT_det[idx].to(self.device)

        with torch.no_grad():
            logits = self.classifier(xT_det.to(self.device))
            probs = torch.sigmoid(logits)
            uncertainty = 1.0 - torch.abs(2.0 * probs - 1.0)
            u_sum = float(uncertainty.sum().item())
            if u_sum <= 1e-12:
                if n < bsz:
                    idx = torch.randperm(bsz, device=xT_det.device)[:n]
                else:
                    idx = torch.randint(0, bsz, (n,), device=xT_det.device)
            else:
                p_uniform = torch.full_like(uncertainty, 1.0 / float(bsz))
                p_hard = uncertainty / uncertainty.sum()
                p = (1.0 - beta) * p_uniform + beta * p_hard
                idx = torch.multinomial(p, num_samples=n, replacement=(n > bsz))
        return xT_det[idx].to(self.device)

    def get_grad_target(
        self,
        xT: torch.Tensor,
        step: int,
    ) -> TerminalGradOutput:
        """Update the classifier and return the current ratio-score field."""

        xT_det = xT.detach()
        metrics: Dict[str, float] = {}
        cls_grad_preclip_sum = 0.0
        cls_grad_clip_hits = 0.0
        cls_grad_clip_count = 0.0
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before get_grad_target(...).")

        if self.cfg.use_replay:
            replay_size_before = self._replay_size()
            replay_ready = replay_size_before >= self.cfg.replay_min_size
        else:
            replay_size_before = 0
            replay_ready = True

        updates_applied = 0
        if replay_ready:
            self.classifier.train()
            total_loss = 0.0
            total_acc = 0.0
            n_updates = self.cfg.classifier_updates_per_step

            for _ in range(n_updates):
                bs = self.cfg.classifier_batch_size or self.cfg.replay_batch_size

                if self.cfg.use_replay:
                    mix_current = min(1.0, max(0.0, float(self.cfg.replay_mix_current)))
                    n_cur = int(round(bs * mix_current))
                    n_cur = min(max(n_cur, 0), bs)
                    n_rep = bs - n_cur
                    mu_parts = []
                    if n_rep > 0:
                        mu_parts.append(self._sample_replay(n_rep))
                    if n_cur > 0:
                        mu_cur = self._sample_current_mu(xT_det=xT_det, n=n_cur)
                        if mu_cur.shape[0] > 0:
                            mu_parts.append(mu_cur)
                    if len(mu_parts) == 0:
                        mu_batch = self._sample_replay(bs)
                    elif len(mu_parts) == 1:
                        mu_batch = mu_parts[0]
                    else:
                        mu_batch = torch.cat(mu_parts, dim=0)
                        perm = torch.randperm(mu_batch.shape[0], device=mu_batch.device)
                        mu_batch = mu_batch[perm]
                else:
                    mu_batch = self._sample_current_mu(xT_det=xT_det, n=bs)
                    bs = int(mu_batch.shape[0])

                rho_batch = self.target_sampler.sample(bs).to(self.device)

                x_all = torch.cat([mu_batch, rho_batch], dim=0)
                labels = torch.cat([
                    torch.ones((mu_batch.shape[0],), device=self.device),
                    torch.zeros((rho_batch.shape[0],), device=self.device),
                ])

                logits = self.classifier(x_all)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip > 0:
                    preclip_norm = nn.utils.clip_grad_norm_(
                        self.classifier.parameters(), self.cfg.grad_clip
                    )
                    preclip_norm_f = float(preclip_norm.item())
                    cls_grad_preclip_sum += preclip_norm_f
                    cls_grad_clip_count += 1.0
                    if preclip_norm_f > float(self.cfg.grad_clip):
                        cls_grad_clip_hits += 1.0
                self.optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_acc += ((logits > 0).float() == labels).float().mean().item()
                updates_applied += 1

            metrics["classifier_loss"] = total_loss / n_updates
            metrics["classifier_acc"] = total_acc / n_updates
            if cls_grad_clip_count > 0:
                metrics["classifier_grad_preclip_norm_mean"] = (
                    cls_grad_preclip_sum / cls_grad_clip_count
                )
                metrics["classifier_grad_clip_frac"] = (
                    cls_grad_clip_hits / cls_grad_clip_count
                )
            self._classifier_ready = True

        metrics["classifier_updates_applied"] = float(updates_applied)
        metrics["ratio_score_ready"] = 1.0 if self._classifier_ready else 0.0

        if not self._classifier_ready:
            self._add_to_replay(xT_det)
            grad_target = torch.zeros_like(xT_det, device=xT_det.device)
            metrics["target_grad_preclip_norm_mean"] = 0.0
            metrics["target_grad_clip_frac"] = 0.0
            metrics["target_grad_scale_mean"] = 1.0
            metrics["target_grad_postclip_norm_mean"] = 0.0
            metrics["grad_norm_mean"] = 0.0
            metrics["provider_step"] = float(step)
            metrics["replay_size"] = float(replay_size_before)
            metrics["replay_size_before"] = float(replay_size_before)
            metrics["replay_size_after"] = float(self._replay_size())
            metrics["is_uncertainty_beta"] = float(self.cfg.uncertainty_is_beta)
            metrics["replay_ready"] = 1.0 if replay_ready else 0.0
            return TerminalGradOutput(grad_target=grad_target, metrics=metrics)

        self.classifier.eval()
        x = xT_det.clone().to(self.device).requires_grad_(True)
        logit = self.classifier(x)
        grad_target = torch.autograd.grad(logit.sum(), x, create_graph=False)[0].detach()

        norms = torch.linalg.norm(grad_target, dim=-1, keepdim=True).clamp_min(1e-12)
        metrics["target_grad_preclip_norm_mean"] = float(norms.mean().item())

        if self.cfg.target_grad_norm_clip is not None:
            clip = float(self.cfg.target_grad_norm_clip)
            scale = torch.clamp(clip / norms, max=1.0)
            grad_target = grad_target * scale
            metrics["target_grad_clip_frac"] = float(
                (scale.squeeze(-1) < (1.0 - 1e-6)).float().mean().item()
            )
            metrics["target_grad_scale_mean"] = float(scale.mean().item())
        else:
            metrics["target_grad_clip_frac"] = 0.0
            metrics["target_grad_scale_mean"] = 1.0

        post_norm = torch.linalg.norm(grad_target, dim=-1)
        post_norm_mean = float(post_norm.mean().item())
        metrics["target_grad_postclip_norm_mean"] = post_norm_mean
        metrics["grad_norm_mean"] = post_norm_mean
        metrics["provider_step"] = float(step)
        metrics["replay_size"] = float(replay_size_before)
        metrics["replay_size_before"] = float(replay_size_before)
        self._add_to_replay(xT_det)
        metrics["replay_size_after"] = float(self._replay_size())
        metrics["replay_ready"] = 1.0 if replay_ready else 0.0
        metrics["is_uncertainty_beta"] = float(self.cfg.uncertainty_is_beta)

        return TerminalGradOutput(grad_target=grad_target, metrics=metrics)
