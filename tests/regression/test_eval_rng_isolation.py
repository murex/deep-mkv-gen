from __future__ import annotations

import pytest
import torch

from deep_mkv_gen.core import (
    CheckpointConfig,
    EvalConfig,
    KConfig,
    RunConfig,
    fit,
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


def _run_with_eval_every(tmp_path, *, provider_case: str, eval_every: int):
    source, target = make_source_target_samples(n_source=64, n_target=64, dim=2, seed=123)
    seed_all(999)

    model = make_regression_model()
    provider_spec = make_provider_spec_case(provider_case)
    cfg = RunConfig(
        device="cpu",
        tag=f"rng-{provider_case}-{eval_every}",
        train_seed=123,
        data_seed=456,
        total_steps=16,
        eval_every=int(eval_every),
        eval=EvalConfig(
            eval_kl_source="batch",
            eval_kl_n=16,
            eval_n=32,
            sw2_proj=16,
            eval_w2_repeats=1,
        ),
        k_config=KConfig(k=1.0),
        mode="continuous",
        model_lr=2e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        batch_size=16,
        checkpoint=CheckpointConfig(
            save_best_checkpoint=False,
            outdir=tmp_path / f"{provider_case}-{eval_every}",
        ),
    )
    cfg.checkpoint.restore_best_at_end = False

    result = fit(
        cfg=cfg,
        model=model,
        provider_spec=provider_spec,
        source_samples=source,
        target_samples=target,
    )
    params = [p.detach().clone() for p in model.parameters()]
    return params, result


@pytest.mark.parametrize("provider_case", PROVIDER_CASES)
def test_intermediate_evaluations_do_not_change_training_for_provider_case(tmp_path, provider_case: str) -> None:
    params_with_eval, result_with_eval = _run_with_eval_every(
        tmp_path,
        provider_case=provider_case,
        eval_every=2,
    )
    params_final_only, result_final_only = _run_with_eval_every(
        tmp_path,
        provider_case=provider_case,
        eval_every=16,
    )

    for left, right in zip(params_with_eval, params_final_only):
        assert torch.allclose(left, right)

    final_a = result_with_eval.records[-1]
    final_b = result_final_only.records[-1]
    assert final_a.w2 == pytest.approx(final_b.w2, rel=0.0, abs=1e-8)
    assert final_a.sw2_sq == pytest.approx(final_b.sw2_sq, rel=0.0, abs=1e-8)
    assert final_a.kl == pytest.approx(final_b.kl, rel=0.0, abs=1e-8)
