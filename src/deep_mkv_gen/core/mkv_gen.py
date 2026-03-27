"""Core deep MKV model and rollout logic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .time_features import SinusoidalTimeEmbedding
from .sampling import SourceSampler


SourceSamplerFn = Callable[
    [int, Optional[torch.device]],
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
]


@dataclass
class DeepMKVGenConfig:
    """Configuration for the deep MKV controller and rollout."""

    dim: int
    T: float
    N: int
    sigma: float
    drift_clip: Optional[float] = None
    drift_clip_type: str = "softnorm"
    time_scheme: str = "uniform"
    time_grid: Optional[torch.Tensor] = None
    z_hidden_dim: int = 128
    z_time_embed_dim: int = 32
    z_num_layers: int = 3
    z_structure: str = "diag"
    z_time_encoding: str = "sinusoidal"
    y0_mode: str = "constant"
    y0_hidden_dim: int = 128
    y0_num_layers: int = 1

    def validate(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("dim must be >= 1")
        if float(self.T) <= 0:
            raise ValueError("T must be > 0")
        if int(self.N) < 1:
            raise ValueError("N must be >= 1")
        if float(self.sigma) < 0:
            raise ValueError("sigma must be >= 0")
        if self.drift_clip_type not in ("softnorm", "tanh"):
            raise ValueError("drift_clip_type must be 'softnorm' or 'tanh'")
        if self.y0_mode not in ("constant", "network"):
            raise ValueError("y0_mode must be 'constant' or 'network'")
        if self.time_grid is not None and self.time_grid.shape != (int(self.N) + 1,):
            raise ValueError("time_grid must have shape (N+1)")


def make_time_grid(T: float, N: int, scheme: str = "cosine", device=None, dtype=torch.float32) -> torch.Tensor:
    """Build the rollout time grid."""
    i = torch.arange(N + 1, device=device, dtype=dtype)
    s = i / N
    if scheme == "uniform":
        return T * s
    if scheme == "sqrt":
        return T * torch.sqrt(s)
    if scheme == "cosine":
        return T * 0.5 * (1.0 - torch.cos(math.pi * s))
    raise ValueError(f"Unknown scheme={scheme}")


class ZNet(nn.Module):
    """Predict the diffusion-control term Z(t, x)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        num_layers: int = 3,
        structure: str = "diag",
        time_encoding: str = "sinusoidal",
        time_horizon: float = 1.0,
    ):
        super().__init__()
        if structure not in ("full", "diag"):
            raise ValueError("structure must be 'full' or 'diag'")
        if time_encoding not in ("sinusoidal", "scalar"):
            raise ValueError("time_encoding must be 'sinusoidal' or 'scalar'")
        if time_horizon <= 0:
            raise ValueError("time_horizon must be > 0")
        self.dim = int(dim)
        self.structure = structure
        self.time_encoding = time_encoding
        self.time_horizon = float(time_horizon)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim) if time_encoding == "sinusoidal" else None

        in_dim = self.dim + (time_embed_dim if time_encoding == "sinusoidal" else 1)
        h = int(hidden_dim)
        layers: List[nn.Module] = []
        for i in range(max(1, int(num_layers))):
            layers.append(nn.Linear(in_dim if i == 0 else h, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())

        out_dim = self.dim * self.dim if structure == "full" else self.dim
        layers.append(nn.Linear(h, out_dim))
        self.net = nn.Sequential(*layers)

        # Small final init helps early rollout stability.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, d = x.shape
        if t.dim() == 0:
            t_in = t.expand(B)
        else:
            t_in = t.view(-1)
            if t_in.shape[0] != B:
                t_in = t_in.expand(B)
        if self.time_encoding == "sinusoidal":
            te = self.time_embed(t_in)
        else:
            te = (t_in.float() / self.time_horizon)[:, None]
        out = self.net(torch.cat([x, te], dim=-1))
        if self.structure == "full":
            return out.view(B, d, d)
        return out.view(B, d)


class Y0Net(nn.Module):
    """Predict a sample-dependent initial control."""

    def __init__(self, dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        h = int(hidden_dim)
        layers: List[nn.Module] = []
        for i in range(max(1, int(num_layers))):
            layers.append(nn.Linear(dim if i == 0 else h, h))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(h, dim))
        self.net = nn.Sequential(*layers)

        # Start close to constant-Y0 behavior.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMKVGen(nn.Module):
    """Deep MKV model with learnable initial control and Z network."""

    def __init__(
        self,
        cfg: DeepMKVGenConfig,
        source_samples: Optional[torch.Tensor] = None,
        source_sampler: Optional[SourceSampler | SourceSamplerFn] = None,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.dim = int(cfg.dim)
        self.T = float(cfg.T)
        self.N = int(cfg.N)
        self.sigma = float(cfg.sigma)
        self.z_net = ZNet(
            dim=self.dim,
            hidden_dim=int(cfg.z_hidden_dim),
            time_embed_dim=int(cfg.z_time_embed_dim),
            num_layers=int(cfg.z_num_layers),
            structure=cfg.z_structure,
            time_encoding=cfg.z_time_encoding,
            time_horizon=self.T,
        )
        self.drift_clip = cfg.drift_clip
        self.drift_clip_type = cfg.drift_clip_type
        self.source_samples: Optional[torch.Tensor] = None
        self.source_sampler: Optional[SourceSampler | SourceSamplerFn] = None
        if source_samples is not None and source_sampler is not None:
            raise ValueError("Provide at most one of source_samples or source_sampler.")
        if source_samples is not None or source_sampler is not None:
            self._set_source(source_samples=source_samples, source_sampler=source_sampler)

        time_grid = cfg.time_grid
        if time_grid is None:
            time_grid = make_time_grid(self.T, self.N, scheme=cfg.time_scheme)
        if time_grid.shape != (self.N + 1,):
            raise ValueError("time_grid must have shape (N+1,)")
        self.register_buffer("time_grid", time_grid)
        self.register_buffer("dt", time_grid[1:] - time_grid[:-1])

        if cfg.y0_mode == "constant":
            self.y0_net = None
            self.y0 = nn.Parameter(torch.zeros(self.dim))
        else:
            self.y0_net = Y0Net(
                dim=self.dim,
                hidden_dim=int(cfg.y0_hidden_dim),
                num_layers=int(cfg.y0_num_layers),
            )
            self.register_parameter("y0", None)

    def _set_source(
        self,
        *,
        source_samples: Optional[torch.Tensor] = None,
        source_sampler: Optional[SourceSampler | SourceSamplerFn] = None,
    ) -> None:
        """Attach the source distribution used by training and sampling."""

        if (source_samples is None) == (source_sampler is None):
            raise ValueError("Provide exactly one of source_samples or source_sampler.")
        if source_samples is not None:
            if not torch.is_tensor(source_samples):
                raise TypeError("source_samples must be a torch.Tensor")
            if source_samples.ndim != 2 or int(source_samples.shape[1]) != self.dim:
                raise ValueError(f"source_samples must have shape (n,{self.dim})")
            self.source_samples = source_samples
            self.source_sampler = SourceSampler(source_samples)
            return

        if not callable(source_sampler) and not (
            hasattr(source_sampler, "sample") and callable(getattr(source_sampler, "sample"))
        ):
            raise TypeError("source_sampler must be callable or define sample(batch_size, device=None)")
        self.source_samples = None
        self.source_sampler = source_sampler

    def _call_source_sampler(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        with_indices: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        if self.source_sampler is None:
            raise RuntimeError(
                "Source distribution is not configured. Provide source_samples or source_sampler to fit(...), "
                "or call _set_source(...)."
            )
        sampler = self.source_sampler
        if with_indices and hasattr(sampler, "sample_with_indices") and callable(getattr(sampler, "sample_with_indices")):
            return sampler.sample_with_indices(batch_size, device=device)
        if hasattr(sampler, "sample") and callable(getattr(sampler, "sample")):
            return sampler.sample(batch_size, device=device)
        return sampler(batch_size, device)

    @torch.no_grad()
    def _sample_source(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Draw initial states without exposing source indices."""

        x0, _ = self._sample_source_with_indices(batch_size=batch_size, device=device)
        return x0

    @torch.no_grad()
    def _sample_source_with_indices(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Draw initial states and optional source indices."""

        out = self._call_source_sampler(batch_size=batch_size, device=device, with_indices=True)
        idx: Optional[torch.Tensor] = None
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError("source_sampler tuple/list output must be (x0, idx)")
            x0, idx = out
        else:
            x0 = out

        if not torch.is_tensor(x0):
            raise TypeError("source_sampler must return a torch.Tensor")
        if x0.shape != (batch_size, self.dim):
            raise ValueError(f"source_sampler must return shape ({batch_size},{self.dim}), got {tuple(x0.shape)}")
        if device is not None and x0.device != device:
            x0 = x0.to(device)

        if idx is not None:
            if not torch.is_tensor(idx):
                raise TypeError("source_sampler idx must be a torch.Tensor")
            idx = idx.reshape(-1)
            if idx.numel() != batch_size:
                raise ValueError(f"source_sampler idx must have {batch_size} entries, got {idx.numel()}")
            if device is not None and idx.device != device:
                idx = idx.to(device)
            idx = idx.to(torch.long)

        return x0, idx

    def _apply_drift_clip(self, Y: torch.Tensor) -> torch.Tensor:
        """Apply the configured drift clipping rule."""

        if self.drift_clip is None:
            return Y
        c = float(self.drift_clip)
        if self.drift_clip_type == "tanh":
            return c * torch.tanh(Y / c)
        if self.drift_clip_type == "softnorm":
            norm = torch.linalg.norm(Y, dim=-1, keepdim=True)
            scale = 1.0 / (1.0 + norm / c)
            return Y * scale
        raise ValueError(f"Unknown drift_clip_type={self.drift_clip_type}")

    def _initial_y(self, x0: torch.Tensor) -> torch.Tensor:
        B, d = x0.shape
        if d != self.dim:
            raise ValueError("x0 has wrong dimension")
        if self.y0_net is None:
            return self.y0[None, :].expand(B, d)
        y = self.y0_net(x0)
        if y.shape != (B, d):
            raise ValueError(f"y0_net must return shape ({B},{d}), got {tuple(y.shape)}")
        return y

    def _brownian_increment(
        self,
        batch_size: int,
        dim: int,
        sqrt_dt: torch.Tensor,
        device: torch.device,
        antithetic: bool,
    ) -> torch.Tensor:
        if antithetic:
            if batch_size % 2 != 0:
                raise ValueError("batch_size must be even with antithetic=True")
            half = batch_size // 2
            dW_half = torch.randn(half, dim, device=device) * sqrt_dt
            return torch.cat([dW_half, -dW_half], dim=0)
        return torch.randn(batch_size, dim, device=device) * sqrt_dt

    def _update_y(self, Y: torch.Tensor, Z: torch.Tensor, dW: torch.Tensor) -> torch.Tensor:
        if Z.dim() == 3:
            return Y + torch.bmm(Z, dW.unsqueeze(-1)).squeeze(-1)
        if Z.dim() == 2:
            return Y + Z * dW
        raise RuntimeError("Z has unexpected shape")

    def _rollout(
        self,
        x0: torch.Tensor,
        antithetic: bool = False,
        snapshot_steps: Optional[Set[int]] = None,
        collect_metrics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[int, torch.Tensor]]:
        """Run the discrete rollout and optionally collect metrics and snapshots."""

        device = x0.device
        B, d = x0.shape
        if d != self.dim:
            raise ValueError("x0 has wrong dimension")

        X = x0
        Y = self._initial_y(x0)
        snapshots: Dict[int, torch.Tensor] = {}
        if snapshot_steps is not None and 0 in snapshot_steps:
            snapshots[0] = X.detach().clone()

        energy = torch.zeros((), device=device)
        drift_clip_frac_acc = torch.zeros((), device=device)
        drift_scale_acc = torch.zeros((), device=device)
        drift_pre_norm_acc = torch.zeros((), device=device)
        drift_post_norm_acc = torch.zeros((), device=device)

        for n in range(self.N):
            t_n = self.time_grid[n].to(device)
            dt_n = self.dt[n].to(device)
            sqrt_dt = torch.sqrt(dt_n)
            dW = self._brownian_increment(B, d, sqrt_dt, device, antithetic)

            Z = self.z_net(t_n, X)
            Y_eff = self._apply_drift_clip(Y)
            if collect_metrics:
                pre_norm = torch.linalg.norm(Y, dim=-1)
                post_norm = torch.linalg.norm(Y_eff, dim=-1)
                if self.drift_clip is not None:
                    c = float(self.drift_clip)
                    drift_clip_frac_acc = drift_clip_frac_acc + (pre_norm > c).float().mean()
                    step_scale = torch.where(
                        pre_norm > 1e-12,
                        post_norm / pre_norm,
                        torch.ones_like(pre_norm),
                    )
                    drift_scale_acc = drift_scale_acc + step_scale.mean()
                else:
                    drift_scale_acc = drift_scale_acc + torch.ones((), device=device)
                drift_pre_norm_acc = drift_pre_norm_acc + pre_norm.mean()
                drift_post_norm_acc = drift_post_norm_acc + post_norm.mean()
            X = X - Y_eff * dt_n + self.sigma * dW
            Y = self._update_y(Y, Z, dW)
            if collect_metrics:
                energy = energy + 0.5 * Y_eff.pow(2).sum(dim=-1).mean() * dt_n
            if snapshot_steps is not None and (n + 1) in snapshot_steps:
                snapshots[n + 1] = X.detach().clone()

        if not collect_metrics:
            return X, Y, {}, snapshots

        denom = float(max(1, self.N))
        return X, Y, {
            "energy": energy,
            "drift_clip_frac": drift_clip_frac_acc / denom,
            "drift_clip_scale_mean": drift_scale_acc / denom,
            "drift_pre_norm_mean": drift_pre_norm_acc / denom,
            "drift_post_norm_mean": drift_post_norm_acc / denom,
        }, snapshots

    def _simulate(self, x0: torch.Tensor, antithetic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run a rollout and return terminal state, terminal control, and metrics."""

        X, Y, metrics, _ = self._rollout(x0, antithetic=antithetic, collect_metrics=True)
        return X, Y, metrics
