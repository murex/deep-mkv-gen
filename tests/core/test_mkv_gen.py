from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deep_mkv_gen.core import (
    DeepMKVGen,
    DeepMKVGenConfig,
    Y0Net,
    ZNet,
    isolated_seed,
    make_time_grid,
    transport_with_snapshots,
)

from tests.helpers import make_empirical_model, make_source_target_samples


def _tiny_znet(*, structure: str = "diag", time_encoding: str = "scalar") -> ZNet:
    return ZNet(
        dim=2,
        hidden_dim=8,
        time_embed_dim=8,
        num_layers=1,
        structure=structure,
        time_encoding=time_encoding,
        time_horizon=1.0,
    )


def _tiny_cfg(
    *,
    dim: int = 2,
    T: float = 1.0,
    N: int = 4,
    sigma: float = 0.3,
    drift_clip: float | None = None,
    z_structure: str = "diag",
    z_time_encoding: str = "scalar",
    y0_mode: str = "constant",
) -> DeepMKVGenConfig:
    return DeepMKVGenConfig(
        dim=dim,
        T=T,
        N=N,
        sigma=sigma,
        drift_clip=drift_clip,
        z_hidden_dim=8,
        z_time_embed_dim=8,
        z_num_layers=1,
        z_structure=z_structure,
        z_time_encoding=z_time_encoding,
        y0_mode=y0_mode,
        y0_hidden_dim=8,
        y0_num_layers=1,
    )


def _zero_sampler(batch_size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros(batch_size, 2, device=device)


def test_simulate_returns_expected_shapes_and_metrics() -> None:
    source, _ = make_source_target_samples(seed=3)
    model = make_empirical_model(source_samples=source)
    x0 = model._sample_source(6, device=torch.device("cpu"))
    xT, yT, info = model._simulate(x0, antithetic=False)

    assert xT.shape == (6, 2)
    assert yT.shape == (6, 2)
    assert set(info) == {
        "energy",
        "drift_clip_frac",
        "drift_clip_scale_mean",
        "drift_pre_norm_mean",
        "drift_post_norm_mean",
    }


def test_simulate_antithetic_requires_even_batch_size() -> None:
    source, _ = make_source_target_samples(seed=4)
    model = make_empirical_model(source_samples=source)
    x0 = torch.zeros(5, 2)
    with pytest.raises(ValueError, match="even"):
        model._simulate(x0, antithetic=True)


def test_drift_clipping_reduces_post_clip_norm() -> None:
    source, _ = make_source_target_samples(seed=5)
    model = make_empirical_model(source_samples=source, drift_clip=0.1)
    with torch.no_grad():
        model.y0.fill_(100.0)

    x0 = torch.zeros(4, 2)
    _, _, info = model._simulate(x0, antithetic=False)

    assert float(info["drift_post_norm_mean"]) < float(info["drift_pre_norm_mean"])
    assert float(info["drift_post_norm_mean"]) <= 0.1001


def test_simulate_matches_final_snapshot_under_same_seed() -> None:
    source, _ = make_source_target_samples(seed=6)
    model = make_empirical_model(source_samples=source)
    x0 = torch.zeros(4, 2)

    with isolated_seed("cpu", 123):
        xT, _, _ = model._simulate(x0, antithetic=False)
    with isolated_seed("cpu", 123):
        snaps = transport_with_snapshots(model, x0, snapshot_steps=[0, model.N], seed=123)

    assert torch.allclose(xT, snaps[model.N])


def test_make_time_grid_supports_multiple_schemes_and_rejects_unknown() -> None:
    uniform = make_time_grid(T=2.0, N=4, scheme="uniform")
    sqrt = make_time_grid(T=1.0, N=4, scheme="sqrt")

    assert torch.allclose(uniform, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert sqrt.shape == (5,)
    assert float(sqrt[0]) == 0.0
    assert float(sqrt[-1]) == 1.0

    with pytest.raises(ValueError, match="Unknown scheme"):
        make_time_grid(T=1.0, N=4, scheme="bad")


def test_znet_supports_full_and_diag_structures_and_time_encodings() -> None:
    x = torch.randn(3, 2)

    z_full = _tiny_znet(structure="full", time_encoding="sinusoidal")
    out_full = z_full(torch.tensor(0.25), x)
    assert out_full.shape == (3, 2, 2)

    z_diag = _tiny_znet(structure="diag", time_encoding="scalar")
    out_diag = z_diag(torch.tensor([0.5]), x)
    assert out_diag.shape == (3, 2)


def test_znet_validates_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="structure"):
        _ = ZNet(dim=2, structure="bad")
    with pytest.raises(ValueError, match="time_encoding"):
        _ = ZNet(dim=2, time_encoding="bad")
    with pytest.raises(ValueError, match="time_horizon"):
        _ = ZNet(dim=2, time_horizon=0.0)


def test_y0net_returns_expected_shape() -> None:
    net = Y0Net(dim=2, hidden_dim=8, num_layers=1)
    out = net(torch.randn(5, 2))
    assert out.shape == (5, 2)


def test_mean_field_control_validates_time_grid_and_y0_mode() -> None:
    model = DeepMKVGen(
        DeepMKVGenConfig(
            dim=2,
            T=1.0,
            N=4,
            sigma=0.3,
        ),
    )
    assert model.source_samples is None
    assert model.source_sampler is None

    with pytest.raises(ValueError, match="at most one"):
        DeepMKVGen(
            DeepMKVGenConfig(
                dim=2,
                T=1.0,
                N=4,
                sigma=0.3,
            ),
            source_samples=torch.zeros(3, 2),
            source_sampler=_zero_sampler,
        )

    with pytest.raises(ValueError, match="time_grid must have shape"):
        DeepMKVGen(
            DeepMKVGenConfig(
                dim=2,
                T=1.0,
                N=4,
                sigma=0.3,
                time_grid=torch.zeros(4),
            ),
            source_samples=torch.zeros(3, 2),
        )

    with pytest.raises(ValueError, match="y0_mode"):
        DeepMKVGen(
            DeepMKVGenConfig(
                dim=2,
                T=1.0,
                N=4,
                sigma=0.3,
                y0_mode="bad",
            ),
            source_samples=torch.zeros(3, 2),
        )


def test_sample_rho0_requires_source_configuration() -> None:
    model = DeepMKVGen(_tiny_cfg())
    with pytest.raises(RuntimeError, match="Source distribution is not configured"):
        model._sample_source(batch_size=2, device=torch.device("cpu"))


def test_sample_rho0_uses_internal_samples_when_provided() -> None:
    samples = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    model = DeepMKVGen(_tiny_cfg(), source_samples=samples)
    x0, idx = model._sample_source_with_indices(batch_size=3, device=torch.device("cpu"))

    assert x0.shape == (3, 2)
    assert idx is not None
    assert idx.shape == (3,)
    assert torch.all((0 <= idx) & (idx < samples.shape[0]))
    assert torch.allclose(x0, samples[idx])


def test_sample_rho0_with_indices_validates_sampler_outputs() -> None:
    def wrong_tuple_len(batch_size: int, device=None):
        del batch_size, device
        return (torch.zeros(1, 2),)

    model = DeepMKVGen(
        _tiny_cfg(),
        source_sampler=wrong_tuple_len,
    )
    with pytest.raises(ValueError, match=r"\(x0, idx\)"):
        model._sample_source_with_indices(batch_size=1, device=torch.device("cpu"))

    def non_tensor_x0(batch_size: int, device=None):
        del batch_size, device
        return "bad"

    model.source_sampler = non_tensor_x0
    with pytest.raises(TypeError, match="torch.Tensor"):
        model._sample_source_with_indices(batch_size=1, device=torch.device("cpu"))

    def wrong_shape(batch_size: int, device=None):
        del batch_size, device
        return torch.zeros(1, 3)

    model.source_sampler = wrong_shape
    with pytest.raises(ValueError, match="must return shape"):
        model._sample_source_with_indices(batch_size=1, device=torch.device("cpu"))

    def non_tensor_idx(batch_size: int, device=None):
        del device
        return torch.zeros(batch_size, 2), [0] * batch_size

    model.source_sampler = non_tensor_idx
    with pytest.raises(TypeError, match="idx must be a torch.Tensor"):
        model._sample_source_with_indices(batch_size=2, device=torch.device("cpu"))

    def wrong_idx_len(batch_size: int, device=None):
        del device
        return torch.zeros(batch_size, 2), torch.zeros(batch_size + 1)

    model.source_sampler = wrong_idx_len
    with pytest.raises(ValueError, match="idx must have"):
        model._sample_source_with_indices(batch_size=2, device=torch.device("cpu"))


def test_sample_rho0_with_indices_moves_outputs_to_requested_device() -> None:
    model = DeepMKVGen(
        _tiny_cfg(),
        source_sampler=lambda batch_size, device=None: (
            torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2),
            torch.arange(batch_size, dtype=torch.int32),
        ),
    )

    x0, idx = model._sample_source_with_indices(batch_size=3, device=torch.device("meta"))

    assert x0.device.type == "meta"
    assert idx is not None
    assert idx.device.type == "meta"
    assert idx.dtype == torch.long


def test_drift_clip_supports_tanh_and_rejects_unknown_type() -> None:
    source, _ = make_source_target_samples(seed=7)
    model = make_empirical_model(source_samples=source, drift_clip=0.1)
    model.drift_clip_type = "tanh"
    clipped = model._apply_drift_clip(torch.full((2, 2), 10.0))
    assert float(clipped.abs().max()) <= 0.1001

    model.drift_clip_type = "bad"
    with pytest.raises(ValueError, match="Unknown drift_clip_type"):
        model._apply_drift_clip(torch.zeros(2, 2))


def test_initial_y_uses_sample_dependent_network_and_validates_shape() -> None:
    class GoodY0(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    class BadY0(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1]

    x0 = torch.zeros(3, 2)
    model = DeepMKVGen(_tiny_cfg(y0_mode="network"), source_sampler=_zero_sampler)
    model.y0_net = GoodY0()
    assert torch.allclose(model._initial_y(x0), torch.ones(3, 2))

    bad_model = DeepMKVGen(_tiny_cfg(y0_mode="network"), source_sampler=_zero_sampler)
    bad_model.y0_net = BadY0()
    with pytest.raises(ValueError, match="y0_net must return shape"):
        bad_model._initial_y(x0)


def test_initial_y_validates_input_dimension() -> None:
    model = DeepMKVGen(_tiny_cfg(), source_sampler=_zero_sampler)
    with pytest.raises(ValueError, match="x0 has wrong dimension"):
        model._initial_y(torch.zeros(3, 3))


def test_update_y_supports_full_structure_and_rejects_bad_rank() -> None:
    source, _ = make_source_target_samples(seed=8)
    model = DeepMKVGen(
        _tiny_cfg(N=3, sigma=0.2, z_structure="full", z_time_encoding="sinusoidal"),
        source_sampler=lambda batch_size, device=None: torch.zeros(batch_size, 2, device=device),
    )
    x0 = torch.zeros(4, 2)
    xT, yT, _ = model._simulate(x0, antithetic=False)
    assert xT.shape == (4, 2)
    assert yT.shape == (4, 2)

    with pytest.raises(RuntimeError, match="unexpected shape"):
        model._update_y(torch.zeros(2, 2), torch.zeros(2, 2, 2, 2), torch.zeros(2, 2))


def test_rollout_validates_input_dimension() -> None:
    model = DeepMKVGen(_tiny_cfg(), source_sampler=_zero_sampler)
    with pytest.raises(ValueError, match="x0 has wrong dimension"):
        model._rollout(torch.zeros(2, 3), collect_metrics=False)


def test_brownian_increment_supports_successful_antithetic_pairing() -> None:
    model = DeepMKVGen(_tiny_cfg(), source_sampler=_zero_sampler)
    dW = model._brownian_increment(
        batch_size=6,
        dim=2,
        sqrt_dt=torch.tensor(0.5),
        device=torch.device("cpu"),
        antithetic=True,
    )

    assert dW.shape == (6, 2)
    assert torch.allclose(dW[:3], -dW[3:])
