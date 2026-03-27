from __future__ import annotations

import torch
import pytest

from deep_mkv_gen.core import RunHooks, fit, sample
from deep_mkv_gen.providers import KLProviderSpec, KLRatioScoreConfig

from tests.helpers import (
    ToyTrainModel,
    ZeroGradProvider,
    make_source_sampler,
    make_empirical_model,
    make_regression_model,
    make_run_config,
    make_source_target_samples,
    make_target_sampler,
    seed_all,
)


def test_fit_continuous_initializes_provider_once(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "continuous", mode="continuous", total_steps=6, eval_every=2)
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    source = torch.zeros(8, 2)
    target = torch.ones(8, 2)

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=source,
        target_samples=target,
    )

    assert provider.on_train_start_calls == 1
    assert len(result.records) == 3


def test_fit_accepts_provider_spec(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "provider-spec", mode="continuous", total_steps=4, eval_every=2)
    source, target = make_source_target_samples(seed=17)
    model = make_regression_model()
    provider_spec = KLProviderSpec(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=False,
            replay_batch_size=8,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )

    result = fit(
        cfg=cfg,
        model=model,
        provider_spec=provider_spec,
        source_samples=source,
        target_samples=target,
    )

    assert len(result.records) == 2


def test_fit_accepts_source_and_target_samplers_and_attaches_source(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "samplers", mode="continuous", total_steps=4, eval_every=2)
    source, target = make_source_target_samples(seed=21)
    model = make_regression_model()
    provider_spec = KLProviderSpec(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=False,
            replay_batch_size=8,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )

    result = fit(
        cfg=cfg,
        model=model,
        provider_spec=provider_spec,
        source_sampler=make_source_sampler(source),
        target_sampler=make_target_sampler(target),
    )
    generated = sample(model, num_samples=8, batch_size=4, device="cpu", seed=5)

    assert len(result.records) == 2
    assert generated.shape == (8, 2)
    assert result.summary["source_size"] == int(source.shape[0])
    assert result.summary["target_size"] == int(target.shape[0])


def test_fit_accepts_optional_validation_inputs(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "validation", mode="continuous", total_steps=4, eval_every=2)
    source_train, target_train = make_source_target_samples(seed=31)
    source_val, target_val = make_source_target_samples(seed=32)
    model = make_regression_model()
    provider_spec = KLProviderSpec(
        cfg=KLRatioScoreConfig(
            hidden_dim=8,
            num_layers=1,
            classifier_updates_per_step=1,
            classifier_lr=1e-3,
            use_replay=False,
            replay_batch_size=8,
            target_grad_norm_clip=1.0,
        ),
        dim=2,
    )

    result = fit(
        cfg=cfg,
        model=model,
        provider_spec=provider_spec,
        source_samples=source_train,
        target_samples=target_train,
        source_val_sampler=make_source_sampler(source_val),
        target_val_sampler=make_target_sampler(target_val),
    )

    assert len(result.records) == 2
    assert result.summary["validation_source_size"] == int(source_val.shape[0])
    assert result.summary["validation_target_size"] == int(target_val.shape[0])


def test_fit_requires_exactly_one_provider_input(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "provider-input", mode="continuous", total_steps=2, eval_every=1)
    model = ToyTrainModel()
    source = torch.zeros(8, 2)
    target = torch.ones(8, 2)

    with pytest.raises(ValueError, match="exactly one of provider_spec or provider"):
        fit(
            cfg=cfg,
            model=model,
            source_samples=source,
            target_samples=target,
        )

    with pytest.raises(ValueError, match="exactly one of provider_spec or provider"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            provider_spec=KLProviderSpec(cfg=KLRatioScoreConfig(use_replay=False), dim=2),
            source_samples=source,
            target_samples=target,
        )


def test_fit_requires_exactly_one_source_and_target_input(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "data-input", mode="continuous", total_steps=2, eval_every=1)
    model = ToyTrainModel()
    source = torch.zeros(8, 2)
    target = torch.ones(8, 2)

    with pytest.raises(ValueError, match="exactly one of source_samples or source_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            target_samples=target,
        )

    with pytest.raises(ValueError, match="exactly one of source_samples or source_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            source_samples=source,
            source_sampler=make_source_sampler(source),
            target_samples=target,
        )

    with pytest.raises(ValueError, match="exactly one of target_samples or target_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            source_samples=source,
        )

    with pytest.raises(ValueError, match="exactly one of target_samples or target_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            source_samples=source,
            target_samples=target,
            target_sampler=make_target_sampler(target),
        )

    with pytest.raises(ValueError, match="exactly one of source_val_samples or source_val_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            source_samples=source,
            target_samples=target,
            source_val_samples=source,
            source_val_sampler=make_source_sampler(source),
        )

    with pytest.raises(ValueError, match="exactly one of target_val_samples or target_val_sampler"):
        fit(
            cfg=cfg,
            model=model,
            provider=ZeroGradProvider(),
            source_samples=source,
            target_samples=target,
            target_val_samples=target,
            target_val_sampler=make_target_sampler(target),
        )


def test_fit_refinement_initializes_provider_per_stage(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path / "refinement", mode="refinement", total_steps=6, eval_every=2)
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    source = torch.zeros(8, 2)
    target = torch.ones(8, 2)

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=source,
        target_samples=target,
    )

    assert provider.on_train_start_calls == 3
    assert len(result.records) == 3


def test_run_calls_stage_hook_once_per_evaluation(tmp_path) -> None:
    captured = []
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=6, eval_every=2)
    cfg.hooks = RunHooks(on_stage_record=lambda record: captured.append(record.stage))
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    assert captured == [1, 2, 3]
    assert len(captured) == len(result.records)


def test_run_defaults_mean_and_std_to_identity(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=2, eval_every=1)
    model = ToyTrainModel()
    provider = ZeroGradProvider()

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    assert len(result.records) == 2


def test_run_hook_batch_size_change_takes_effect_after_eval(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=4, eval_every=2)
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    def on_stage_record(record) -> None:
        if record.stage == 1:
            cfg.batch_size = 6

    cfg.hooks = RunHooks(on_stage_record=on_stage_record)

    fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    train_batches = [b for b in model.batch_history if b != int(cfg.eval.eval_n)]
    assert train_batches == [4, 4, 6, 6]


def test_run_hook_lr_change_changes_training_trajectory(tmp_path) -> None:
    source = torch.zeros(8, 2)
    target = torch.ones(8, 2)

    baseline_cfg = make_run_config(outdir=tmp_path / "baseline", mode="continuous", total_steps=4, eval_every=2)
    baseline_cfg.checkpoint.restore_best_at_end = False
    baseline_model = ToyTrainModel(initial_weight=1.0)
    baseline_provider = ZeroGradProvider()
    fit(
        cfg=baseline_cfg,
        model=baseline_model,
        provider=baseline_provider,
        source_samples=source,
        target_samples=target,
    )

    changed_cfg = make_run_config(outdir=tmp_path / "changed", mode="continuous", total_steps=4, eval_every=2)
    changed_cfg.checkpoint.restore_best_at_end = False
    changed_model = ToyTrainModel(initial_weight=1.0)
    changed_provider = ZeroGradProvider()

    def on_stage_record(record) -> None:
        if record.stage == 1:
            changed_cfg.model_lr = 1e-4

    changed_cfg.hooks = RunHooks(on_stage_record=on_stage_record)
    fit(
        cfg=changed_cfg,
        model=changed_model,
        provider=changed_provider,
        source_samples=source,
        target_samples=target,
    )

    assert not torch.allclose(baseline_model.weight.detach(), changed_model.weight.detach())


def test_run_result_summary_and_checkpoint_metadata_are_coherent(tmp_path) -> None:
    cfg = make_run_config(
        outdir=tmp_path,
        mode="continuous",
        total_steps=4,
        eval_every=2,
        best_metric="w2",
        save_best_checkpoint=True,
    )
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.exists()
    assert len(result.summary["records"]) == len(result.records)
    assert result.summary["checkpoint_metric"] == "w2"
    assert result.summary["best_checkpoint_path"] == str(result.best_checkpoint_path)


def test_run_restores_best_in_memory_weights_by_default(tmp_path) -> None:
    cfg = make_run_config(
        outdir=tmp_path,
        mode="continuous",
        total_steps=4,
        eval_every=2,
        save_best_checkpoint=True,
    )
    model = ToyTrainModel(initial_weight=1.0)
    provider = ZeroGradProvider()

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    assert result.best_checkpoint_path is not None
    payload = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert torch.allclose(model.weight.detach().cpu(), payload["model_state_dict"]["weight"])
    assert cfg.checkpoint.keep_best_in_memory is True
    assert cfg.checkpoint.restore_best_at_end is True


def test_run_is_quiet_by_default(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=2, eval_every=1)
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_logs_when_verbose(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=2, eval_every=1)
    cfg.verbose = True
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    captured = capsys.readouterr()
    assert "run mode=continuous" in captured.out
    assert "stage=1" in captured.out


def test_run_uses_default_tag_when_tag_is_none(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_run_config(
        outdir=tmp_path,
        mode="continuous",
        total_steps=2,
        eval_every=1,
        save_best_checkpoint=True,
        tag=None,
    )
    cfg.verbose = True
    model = ToyTrainModel()
    provider = ZeroGradProvider()

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    captured = capsys.readouterr()
    assert "[run]" in captured.out
    assert result.summary["tag"] == "run"
    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.name.startswith("run_best_")


def test_run_logs_best_w2_with_best_record_k_stage_and_step(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = make_run_config(outdir=tmp_path, mode="continuous", total_steps=4, eval_every=2)
    cfg.verbose = True
    model = ToyTrainModel(initial_weight=1.0)
    provider = ZeroGradProvider()

    def on_stage_record(record) -> None:
        if record.stage == 1:
            cfg.k_config.k = 7.0

    cfg.hooks = RunHooks(on_stage_record=on_stage_record)

    result = fit(
        cfg=cfg,
        model=model,
        provider=provider,
        source_samples=torch.zeros(8, 2),
        target_samples=torch.ones(8, 2),
    )

    captured = capsys.readouterr()
    assert result.summary["best_stage_w2"] == 1
    assert result.summary["best_step_w2"] == 2
    assert result.summary["best_k_w2"] == pytest.approx(1.0)
    assert "best W2=" in captured.out
    assert "at k=1.0000, stage=1, step=2" in captured.out


def test_fit_reseeds_training_rng_for_reproducible_runs(tmp_path) -> None:
    source, target = make_source_target_samples(seed=99)

    seed_all(111)
    model1 = make_empirical_model(dim=2, N=4, sigma=0.2)
    initial_state = {k: v.detach().clone() for k, v in model1.state_dict().items()}

    seed_all(222)
    model2 = make_empirical_model(dim=2, N=4, sigma=0.2)
    model2.load_state_dict(initial_state)

    cfg1 = make_run_config(outdir=tmp_path / "run1", mode="continuous", total_steps=4, eval_every=2)
    cfg1.checkpoint.restore_best_at_end = False
    cfg2 = make_run_config(outdir=tmp_path / "run2", mode="continuous", total_steps=4, eval_every=2)
    cfg2.checkpoint.restore_best_at_end = False

    provider1 = ZeroGradProvider()
    provider2 = ZeroGradProvider()

    seed_all(333)
    result1 = fit(
        cfg=cfg1,
        model=model1,
        provider=provider1,
        source_samples=source,
        target_samples=target,
    )

    seed_all(444)
    result2 = fit(
        cfg=cfg2,
        model=model2,
        provider=provider2,
        source_samples=source,
        target_samples=target,
    )

    for left, right in zip(model1.state_dict().values(), model2.state_dict().values()):
        assert torch.allclose(left, right)

    assert len(result1.records) == len(result2.records)
    for rec1, rec2 in zip(result1.records, result2.records):
        assert rec1.steps_done == rec2.steps_done
        assert rec1.stage == rec2.stage
        assert rec1.w2 == pytest.approx(rec2.w2)
        assert rec1.sw2_sq == pytest.approx(rec2.sw2_sq)
        assert rec1.kl == pytest.approx(rec2.kl)
        assert rec1.mean_grad_norm == pytest.approx(rec2.mean_grad_norm)
