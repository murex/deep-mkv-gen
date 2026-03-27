from __future__ import annotations

import torch

from deep_mkv_gen.core import TargetSampler
from deep_mkv_gen.core.run import StageEvaluator

from tests.helpers import ToyTrainModel, make_empirical_model, make_run_config, make_source_target_samples


def test_stage_evaluator_returns_finite_metrics_and_preserves_rng_and_mode(tmp_path) -> None:
    source, target = make_source_target_samples(seed=12)
    model = make_empirical_model(source_samples=source)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=TargetSampler(target),
        target_samples=target,
    )
    model.train(True)
    torch_before = torch.get_rng_state().clone()

    record = evaluator.evaluate_stage(
        stage=1,
        steps_done=1,
        train_logs=[{"classifier_acc": 0.5, "classifier_loss": 1.0, "grad_norm_mean": 1.0}],
    )

    assert torch.isfinite(torch.tensor(record.w2))
    assert torch.isfinite(torch.tensor(record.sw2_sq))
    assert torch.isfinite(torch.tensor(record.kl))
    assert record.provider_metrics["classifier_acc"] == 0.5
    assert record.provider_metrics["classifier_loss"] == 1.0
    assert model.training is True
    assert torch.equal(torch_before, torch.get_rng_state())


def test_target_eval_set_cache_is_stable(tmp_path) -> None:
    source, target = make_source_target_samples(seed=13)
    model = make_empirical_model(source_samples=source)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=TargetSampler(target),
        target_samples=target,
    )

    first = evaluator._target_eval_sets(cfg.eval, step_id=1)
    second = evaluator._target_eval_sets(cfg.eval, step_id=2)

    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert torch.allclose(left, right)


def test_target_eval_sets_split_fixed_samples_without_overlap_when_enough(tmp_path) -> None:
    source, _ = make_source_target_samples(seed=14)
    target = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    model = make_empirical_model(source_samples=source)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    cfg.eval.eval_n = 4
    cfg.eval.eval_w2_repeats = 3
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=TargetSampler(target),
        target_samples=target,
    )

    sets = evaluator._target_eval_sets(cfg.eval, step_id=1)
    flat_rows = [tuple(row.tolist()) for batch in sets for row in batch]

    assert len(sets) == 3
    assert all(batch.shape == (4, 2) for batch in sets)
    assert len(set(flat_rows)) == 12


def test_target_eval_sampler_draws_fresh_batches_per_stage(tmp_path) -> None:
    class CountingTargetSampler:
        def __init__(self) -> None:
            self.N = 100
            self.calls = 0

        def sample(self, batch_size: int) -> torch.Tensor:
            self.calls += 1
            return torch.full((batch_size, 2), float(self.calls))

    source, _ = make_source_target_samples(seed=15)
    sampler = CountingTargetSampler()
    model = make_empirical_model(source_samples=source)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    cfg.eval.eval_n = 3
    cfg.eval.eval_w2_repeats = 2
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=sampler,
    )

    first = evaluator._target_eval_sets(cfg.eval, step_id=1)
    second = evaluator._target_eval_sets(cfg.eval, step_id=2)

    assert sampler.calls == 4
    assert torch.all(first[0] == 1.0)
    assert torch.all(first[1] == 2.0)
    assert torch.all(second[0] == 3.0)
    assert torch.all(second[1] == 4.0)


def test_stage_evaluator_uses_validation_source_samples_for_generation(tmp_path) -> None:
    train_source = torch.zeros(8, 2)
    val_source = torch.ones(8, 2)
    target = torch.ones(8, 2)
    model = ToyTrainModel(initial_weight=1.0)
    model._set_source(source_samples=train_source)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    cfg.eval.eval_n = 8
    cfg.eval.eval_w2_repeats = 1
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=TargetSampler(target),
        target_samples=target,
        source_samples=val_source,
    )

    record = evaluator.evaluate_stage(stage=1, steps_done=1, train_logs=[])

    assert record.w2 == 0.0


def test_stage_evaluator_eval_kl_sample_uses_validation_source_override(tmp_path) -> None:
    class CountingSourceSampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, batch_size: int, device=None) -> torch.Tensor:
            self.calls += 1
            return torch.zeros(batch_size, 2, device=device)

    train_source_sampler = CountingSourceSampler()
    val_source = torch.ones(8, 2)
    target = torch.ones(8, 2)
    model = ToyTrainModel(initial_weight=1.0)
    model._set_source(source_sampler=train_source_sampler)
    cfg = make_run_config(outdir=tmp_path, total_steps=2, eval_every=1)
    cfg.eval.eval_n = 8
    cfg.eval.eval_kl_n = 4
    cfg.eval.eval_kl_source = "sample"
    evaluator = StageEvaluator(
        cfg=cfg,
        model=model,
        target_sampler=TargetSampler(target),
        target_samples=target,
        source_samples=val_source,
    )

    evaluator.evaluate_stage(stage=1, steps_done=1, train_logs=[])

    assert train_source_sampler.calls == 0
