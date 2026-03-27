"""KL terminal gradients from the score-difference decomposition."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import torch

from ...core.protocols import ScoreProvider, TargetSamplerLike, TerminalGradOutput
from .score_provider_dsm import DSMScoreConfig, DSMScoreModel


@dataclass
class KLScoreDifferenceConfig:
    """Configuration for the score-difference KL estimator."""

    score_mu_updates_per_step: int = 1
    target_grad_norm_clip: Optional[float] = None
    mu_init: Literal["copy_target", "fresh"] = "copy_target"
    target_score_cfg: Optional[DSMScoreConfig] = None
    mu_score_cfg: Optional[DSMScoreConfig] = None

    def validate(self) -> None:
        if int(self.score_mu_updates_per_step) < 1:
            raise ValueError("score_mu_updates_per_step must be >= 1")
        if self.target_grad_norm_clip is not None and float(self.target_grad_norm_clip) <= 0.0:
            raise ValueError("target_grad_norm_clip must be > 0 when provided")
        if self.mu_init not in ("copy_target", "fresh"):
            raise ValueError("mu_init must be one of: copy_target, fresh")
        if self.target_score_cfg is not None:
            self.target_score_cfg.validate()
        if self.mu_score_cfg is not None:
            self.mu_score_cfg.validate()


class ScoreDifferenceKLTerminalGradProvider:
    """KL terminal gradients built from s_mu - s_rho."""

    def __init__(
        self,
        target_score_model: ScoreProvider,
        cfg: KLScoreDifferenceConfig,
        mu_score_model: Optional[ScoreProvider] = None,
    ):
        self.target_score_model = target_score_model
        self.cfg = cfg
        self._mu_score_model_explicit = mu_score_model is not None
        if mu_score_model is None:
            self.mu_score_model = self._build_implicit_mu_score_model()
        else:
            self.mu_score_model = mu_score_model
        self.target_sampler: Optional[TargetSamplerLike] = None
        self.device: Optional[torch.device] = None
        self._target_score_prepared = False
        self.calibration_info: Dict[str, float] = {}

    def set_target_sampler(self, target_sampler: TargetSamplerLike) -> None:
        if self.target_sampler is not target_sampler:
            self._target_score_prepared = False
            self.calibration_info = {}
        self.target_sampler = target_sampler

    def _build_fresh_implicit_mu_score_model(self) -> ScoreProvider:
        if not isinstance(self.target_score_model, DSMScoreModel):
            raise TypeError(
                "mu_init='fresh' is only supported for implicit DSM score models. "
                "Pass mu_score_model explicitly for custom score providers."
            )
        if self.cfg.mu_score_cfg is not None:
            mu_cfg = copy.deepcopy(self.cfg.mu_score_cfg)
        else:
            mu_cfg = copy.deepcopy(self.target_score_model.cfg)
        mu_cfg.validate()
        return DSMScoreModel(dim=int(self.target_score_model.dim), cfg=mu_cfg)

    def _build_implicit_mu_score_model(self) -> ScoreProvider:
        if self.cfg.mu_init == "fresh":
            return self._build_fresh_implicit_mu_score_model()
        try:
            return copy.deepcopy(self.target_score_model)
        except Exception as exc:  # pragma: no cover
            raise TypeError(
                "mu_score_model is None and target_score_model could not be copied. "
                "Pass mu_score_model explicitly."
            ) from exc

    def _apply_mu_score_cfg_to_implicit_copy(self) -> None:
        if self.cfg.mu_init != "copy_target":
            return
        if self.cfg.mu_score_cfg is None:
            return
        if not isinstance(self.target_score_model, object) or not hasattr(self.target_score_model, "cfg"):
            raise TypeError("Implicit mu score config override requires a score model with a cfg attribute.")
        if not hasattr(self.mu_score_model, "cfg"):
            raise TypeError("Implicit mu score config override requires a mu score model with a cfg attribute.")

        target_cfg = getattr(self.target_score_model, "cfg")
        mu_cfg = copy.deepcopy(self.cfg.mu_score_cfg)
        if not hasattr(target_cfg, "hidden_dim") or not hasattr(target_cfg, "num_layers"):
            raise TypeError("Implicit mu score config override requires score cfgs with hidden_dim and num_layers.")
        if int(mu_cfg.hidden_dim) != int(target_cfg.hidden_dim) or int(mu_cfg.num_layers) != int(target_cfg.num_layers):
            raise ValueError(
                "mu_score_cfg must match target_score_cfg architecture when mu_score_model is implicit "
                f"(hidden_dim/num_layers). Got target=({int(target_cfg.hidden_dim)}, {int(target_cfg.num_layers)}) "
                f"and mu=({int(mu_cfg.hidden_dim)}, {int(mu_cfg.num_layers)})."
            )
        mu_cfg.validate()
        self.mu_score_model.cfg = mu_cfg

    def calibrate(
        self,
        target_sampler: Optional[TargetSamplerLike] = None,
        device: Union[str, torch.device] = "cpu",
        force: bool = False,
    ) -> Dict[str, float]:
        """Prepare the frozen target score model."""

        if target_sampler is not None:
            self.target_sampler = target_sampler

        if not force and self._target_score_prepared:
            return dict(self.calibration_info)

        info = self.target_score_model.compute_score(
            x=None,
            step=0,
            target_sampler=self.target_sampler,  # type: ignore[arg-type]
            device=device,
            force=force,
        ).metrics

        if not self._mu_score_model_explicit:
            if self.cfg.mu_init == "copy_target":
                try:
                    self.mu_score_model = copy.deepcopy(self.target_score_model)
                except Exception as exc:  # pragma: no cover
                    raise TypeError(
                        "mu_score_model is implicit, but target_score_model could not be copied after calibration."
                    ) from exc
            else:
                self.mu_score_model = self._build_fresh_implicit_mu_score_model()
            self._apply_mu_score_cfg_to_implicit_copy()

        self.calibration_info = {f"target_{k}": float(v) for k, v in info.items()}
        self._target_score_prepared = True
        return dict(self.calibration_info)

    def on_train_start(self, device: Union[str, torch.device], train_cfg: object) -> None:
        """Prepare target and mu score models for a training phase."""

        self.cfg.validate()
        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before on_train_start(...).")
        self.device = torch.device(device)

        if not self._target_score_prepared:
            self.calibrate(device=device, force=False)

        self.target_score_model.on_train_start(device=device, train_cfg=train_cfg)
        self.mu_score_model.on_train_start(device=device, train_cfg=train_cfg)

        self.target_score_model.set_trainable(False)
        self.target_score_model.eval()
        self.mu_score_model.set_trainable(True)
        self.mu_score_model.train()

    def _update_mu_score_from_current(self, *, xT: torch.Tensor, step: int) -> Dict[str, float]:
        if self.cfg.score_mu_updates_per_step == 1:
            metrics = {
                k: float(v)
                for k, v in self.mu_score_model.compute_score(x=xT, step=step).metrics.items()
            }
        else:
            metric_sums: Dict[str, float] = {}
            for _ in range(self.cfg.score_mu_updates_per_step):
                rec = self.mu_score_model.compute_score(x=xT, step=step).metrics
                for k, v in rec.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metrics = {k: (v / float(self.cfg.score_mu_updates_per_step)) for k, v in metric_sums.items()}
        if len(metrics) == 0:
            metrics = {"score_mu_loss": 0.0}
        metrics["score_mu_updates_applied"] = float(self.cfg.score_mu_updates_per_step)
        return metrics

    def get_grad_target(
        self,
        xT: torch.Tensor,
        step: int,
    ) -> TerminalGradOutput:
        """Update the mu score and return s_mu - s_rho at xT."""

        if self.target_sampler is None:
            raise RuntimeError("Call set_target_sampler(...) before get_grad_target(...).")
        xT_det = xT.detach()
        metrics = self._update_mu_score_from_current(xT=xT_det, step=step)

        with torch.no_grad():
            mu_was_training = self.mu_score_model.training
            self.mu_score_model.eval()
            mu_score = self.mu_score_model.compute_score(x=xT_det, step=step).score
            if mu_was_training:
                self.mu_score_model.train()

            rho_score = self.target_score_model.compute_score(x=xT_det, step=step).score
            if mu_score is None or rho_score is None:
                raise RuntimeError("score_model.compute_score returned None score for x input.")
            grad_score = mu_score - rho_score
            preclip_norm = torch.linalg.norm(grad_score, dim=-1, keepdim=True).clamp_min(1e-12)
            metrics["target_grad_preclip_norm_mean"] = float(preclip_norm.mean().item())
            if self.cfg.target_grad_norm_clip is not None:
                clip = float(self.cfg.target_grad_norm_clip)
                scale = torch.clamp(clip / preclip_norm, max=1.0)
                grad_score = grad_score * scale
                metrics["target_grad_clip_frac"] = float(
                    (scale.squeeze(-1) < (1.0 - 1e-6)).float().mean().item()
                )
                metrics["target_grad_scale_mean"] = float(scale.mean().item())
            else:
                metrics["target_grad_clip_frac"] = 0.0
                metrics["target_grad_scale_mean"] = 1.0

        post_norm = torch.linalg.norm(grad_score, dim=-1)
        post_norm_mean = float(post_norm.mean().item())
        metrics["target_grad_postclip_norm_mean"] = post_norm_mean
        metrics["grad_norm_mean"] = post_norm_mean
        metrics["provider_step"] = float(step)

        return TerminalGradOutput(grad_target=grad_score, metrics=metrics)
