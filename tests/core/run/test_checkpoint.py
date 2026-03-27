from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deep_mkv_gen.core.run.checkpoint import maybe_restore_best_model, maybe_save_best_checkpoint
from deep_mkv_gen.core.run.state import RunState

from tests.helpers import make_run_config, make_stage_record


@pytest.mark.parametrize(
    ("metric_name", "record_a", "record_b"),
    [
        ("w2", make_stage_record(w2=2.0, sw2_sq=1.0, kl=1.0), make_stage_record(stage=2, steps_done=2, w2=1.0, sw2_sq=5.0, kl=5.0)),
        ("sw2", make_stage_record(w2=2.0, sw2_sq=2.0, kl=1.0), make_stage_record(stage=2, steps_done=2, w2=5.0, sw2_sq=1.0, kl=5.0)),
        ("kl", make_stage_record(w2=2.0, sw2_sq=2.0, kl=2.0), make_stage_record(stage=2, steps_done=2, w2=5.0, sw2_sq=5.0, kl=1.0)),
    ],
)
def test_maybe_save_best_checkpoint_tracks_selected_metric(tmp_path, metric_name, record_a, record_b) -> None:
    cfg = make_run_config(
        outdir=tmp_path,
        total_steps=2,
        eval_every=1,
        best_metric=metric_name,
        save_best_checkpoint=True,
    )
    state = RunState()
    model = nn.Linear(2, 2)

    assert maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=record_a) is True
    first_path = state.best_checkpoint_path
    assert first_path is not None and first_path.exists()
    assert state.best_model_state_dict is not None

    assert maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=record_b) is True
    payload = torch.load(first_path, map_location="cpu")
    assert payload["checkpoint_metric"] == metric_name
    assert payload["stage"] == 2

    worse_record = make_stage_record(stage=3, steps_done=3, w2=9.0, sw2_sq=9.0, kl=9.0)
    assert maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=worse_record) is False


def test_maybe_save_best_checkpoint_respects_fully_disabled_tracking(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path, save_best_checkpoint=False)
    cfg.checkpoint.keep_best_in_memory = False
    cfg.checkpoint.restore_best_at_end = False
    state = RunState()
    model = nn.Linear(2, 2)
    record = make_stage_record()

    assert maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=record) is False
    assert state.best_checkpoint_path is None
    assert state.best_model_state_dict is None


def test_maybe_save_best_checkpoint_keeps_best_in_memory_by_default(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path, save_best_checkpoint=False)
    state = RunState()
    model = nn.Linear(2, 2)
    record = make_stage_record()

    assert maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=record) is True
    assert state.best_checkpoint_path is None
    assert state.best_model_state_dict is not None


def test_maybe_restore_best_model_loads_in_memory_state(tmp_path) -> None:
    cfg = make_run_config(outdir=tmp_path, save_best_checkpoint=False)
    state = RunState()
    model = nn.Linear(2, 2)
    record = make_stage_record()

    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(2.0)
    maybe_save_best_checkpoint(cfg=cfg, state=state, model=model, record=record)

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    assert maybe_restore_best_model(cfg, state, model) is True
    assert torch.allclose(model.weight, torch.ones_like(model.weight))
    assert torch.allclose(model.bias, torch.full_like(model.bias, 2.0))
