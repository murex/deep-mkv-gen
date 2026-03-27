from __future__ import annotations

import pytest
import torch

from deep_mkv_gen.core import (
    CheckpointConfig,
    EvalConfig,
    KConfig,
    RunConfig,
    fit,
    sample,
    wasserstein2,
)

from tests.helpers import make_provider_spec_case, make_regression_model, make_source_target_samples, seed_all

try:  # pragma: no cover - environment dependent
    import geomloss  # noqa: F401

    HAS_GEOMLOSS = True
except Exception:  # pragma: no cover - environment dependent
    HAS_GEOMLOSS = False


GEOMLOSS_CASE = pytest.mark.skipif(not HAS_GEOMLOSS, reason="geomloss is not installed")

PROVIDER_CASES = [
    pytest.param("kl_ratio", id="kl-ratio"),
    pytest.param("kl_difference", id="kl-difference"),
    pytest.param("w2", marks=GEOMLOSS_CASE, id="w2"),
    pytest.param("hybrid_ratio", marks=GEOMLOSS_CASE, id="hybrid-ratio"),
    pytest.param("hybrid_difference", marks=GEOMLOSS_CASE, id="hybrid-difference"),
]


@pytest.mark.parametrize("mode", ["continuous", "refinement"])
@pytest.mark.parametrize("provider_case", PROVIDER_CASES)
def test_end_to_end_transport_improves_for_provider_case(
    tmp_path,
    provider_case: str,
    mode: str,
) -> None:
    source, target = make_source_target_samples(n_source=64, n_target=64, dim=2, seed=123)
    seed_all(999)

    model = make_regression_model()
    model._set_source(source_samples=source)
    provider_spec = make_provider_spec_case(provider_case)
    initial_samples = sample(model, num_samples=32, batch_size=32, device="cpu", seed=777)
    initial_w2 = wasserstein2(initial_samples, target[:32])

    cfg = RunConfig(
        device="cpu",
        tag=f"improve-{provider_case}-{mode}",
        train_seed=123,
        data_seed=456,
        total_steps=16,
        eval_every=2,
        eval=EvalConfig(
            eval_kl_source="batch",
            eval_kl_n=16,
            eval_n=32,
            sw2_proj=16,
            eval_w2_repeats=1,
        ),
        k_config=KConfig(k=1.0),
        mode=mode,
        model_lr=2e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        batch_size=16,
        checkpoint=CheckpointConfig(
            save_best_checkpoint=False,
            outdir=tmp_path / mode / provider_case,
        ),
    )

    result = fit(
        cfg=cfg,
        model=model,
        provider_spec=provider_spec,
        source_samples=source,
        target_samples=target,
    )

    assert result.summary["best_w2"] < initial_w2
    assert torch.isfinite(torch.tensor(result.records[-1].w2))
    assert torch.isfinite(torch.tensor(result.records[-1].sw2_sq))
    assert torch.isfinite(torch.tensor(result.records[-1].kl))
