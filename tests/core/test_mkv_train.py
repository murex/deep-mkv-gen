from __future__ import annotations

from deep_mkv_gen.core import TrainHooks, TrainRuntime
from deep_mkv_gen.core.mkv_train import train

from tests.helpers import ToyTrainModel, ZeroGradProvider, make_small_train_cfg


def test_train_odd_batch_size_uses_even_effective_batch_size() -> None:
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    cfg = make_small_train_cfg(num_steps=1, batch_size=5)
    logs = train(model=model, cfg=cfg, grad_provider=provider, device="cpu")

    assert logs[0]["effective_batch_size"] == 4.0
    assert model.batch_history == [4]


def test_train_batch_size_one_duplicates_single_sample() -> None:
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    cfg = make_small_train_cfg(num_steps=1, batch_size=1)
    logs = train(model=model, cfg=cfg, grad_provider=provider, device="cpu")

    assert logs[0]["effective_batch_size"] == 2.0
    assert model.batch_history == [1]


def test_train_eval_callback_receives_dense_interval_logs() -> None:
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    cfg = make_small_train_cfg(num_steps=5, batch_size=4)
    cfg.log_every = 10
    interval_sizes: list[int] = []

    logs = train(
        model=model,
        cfg=cfg,
        grad_provider=provider,
        device="cpu",
        hooks=TrainHooks(on_eval=lambda event: interval_sizes.append(len(event.logs))),
        eval_every=2,
    )

    assert interval_sizes == [2, 2, 1]
    assert len(logs) == 1


def test_train_runtime_reuse_depends_on_reset_state() -> None:
    model = ToyTrainModel()
    provider = ZeroGradProvider()
    cfg = make_small_train_cfg(num_steps=2, batch_size=4)
    runtime = TrainRuntime()

    train(model=model, cfg=cfg, grad_provider=provider, device="cpu", runtime=runtime, reset_state=True)
    train(model=model, cfg=cfg, grad_provider=provider, device="cpu", runtime=runtime, reset_state=False)

    assert provider.on_train_start_calls == 1
    assert runtime.phase_step == 4

    model2 = ToyTrainModel()
    provider2 = ZeroGradProvider()
    runtime2 = TrainRuntime()
    train(model=model2, cfg=cfg, grad_provider=provider2, device="cpu", runtime=runtime2, reset_state=True)
    train(model=model2, cfg=cfg, grad_provider=provider2, device="cpu", runtime=runtime2, reset_state=True)

    assert provider2.on_train_start_calls == 2
    assert runtime2.phase_step == 2
