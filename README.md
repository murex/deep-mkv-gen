# deep-mkv-gen

`deep-mkv-gen` is the official implementation of the deep MKV generative model.
It provides:

- the model and rollout runtime
- staged training and evaluation orchestration
- public provider-spec types for choosing KL, W2, or hybrid training
- sampling and evaluation helpers

This repo does not contain the notebook workflows or local experiment artifacts.
Those now live in the sibling repo `deep-mkv-gen-notebooks`.


## Install

Core package:

```bash
pip install -e .
```

With the GeomLoss-backed W2 provider:

```bash
pip install -e ".[geomloss]"
```

With test dependencies:

```bash
pip install -e ".[test]"
```

## Core Idea

The public training workflow is built around three pieces:

1. a `DeepMKVGen` model
2. a `RunConfig`
3. a provider spec such as `KLProviderSpec`, `W2ProviderSpec`, or `HybridProviderSpec`


## Quick Start

```python
import torch

from deep_mkv_gen import (
    DeepMKVGen,
    DeepMKVGenConfig,
    EvalConfig,
    GeomLossW2ProviderConfig,
    KConfig,
    RunConfig,
    W2ProviderSpec,
    fit,
    sample,
)

source = torch.randn(4096, 2)
target = torch.randn(4096, 2) + 2.0

model = DeepMKVGen(
    DeepMKVGenConfig(
        dim=2,
        T=1.0,
        N=16,
        sigma=0.3,
    ),
)

provider_spec = W2ProviderSpec(
    cfg=GeomLossW2ProviderConfig(
        blur=0.2,
        scaling=0.9,
    ),
    dim=2,
)

cfg = RunConfig(
    device="cpu",
    train_seed=123,
    data_seed=456,
    total_steps=200,
    eval_every=50,
    k_config=KConfig(k=1.0),
    eval=EvalConfig(
        eval_kl_source="batch",
        eval_kl_n=256,
        eval_n=512,
        sw2_proj=64,
        eval_w2_repeats=1,
    ),
)

result = fit(
    cfg=cfg,
    model=model,
    provider_spec=provider_spec,
    source_samples=source,
    target_samples=target,
)

generated = sample(model, num_samples=512, batch_size=256, device="cpu", seed=7)
print(result.summary["best_w2"])
print(generated.shape)
```

`fit(...)` accepts exactly one source input and exactly one target input:

- `source_samples` or `source_sampler`
- `target_samples` or `target_sampler`

If you pass sample tensors, `fit(...)` wraps them in empirical samplers and
attaches the resolved source sampler back onto the model so post-fit calls such
as `sample(model, ...)` still work.

`fit(...)` also accepts optional held-out validation inputs:

- `source_val_samples` or `source_val_sampler`
- `target_val_samples` or `target_val_sampler`

Training still uses the training source/target inputs. When validation inputs
are provided, stage metrics are computed on the held-out validation source and
target instead.

If you want explicit sampler objects on both sides, use `SourceSampler` and
`TargetSampler`:

```python
from deep_mkv_gen import SourceSampler, TargetSampler

result = fit(
    cfg=cfg,
    model=model,
    provider_spec=provider_spec,
    source_sampler=SourceSampler(source),
    target_sampler=TargetSampler(target),
)
```

`SourceSampler` and `TargetSampler` are convenience implementations, not closed
types. `fit(...)` accepts compatible external sampler objects as well.

For the source side, a custom sampler can subclass `SourceSampler` or just
provide the same interface:

- `sample(batch_size, device=None)` is sufficient
- `sample_with_indices(batch_size, device=None)` is optional and is used when
  source indices are requested by `sample_with_snapshots(...)`
- `N` is optional and is only used for summary metadata

For the target side, a custom sampler can subclass `TargetSampler` or provide:

- `sample(batch_size)`
- `N`

Validation evaluation uses slightly different semantics depending on whether
you pass fixed tensors or samplers:

- fixed `target_val_samples`: repeated W2 evaluation partitions the validation
  target set into disjoint chunks when `eval_n * eval_w2_repeats` fits inside
  the available samples; otherwise it falls back to resampling with replacement
- `target_val_sampler`: one fresh target batch is drawn per repeat at each
  evaluation stage
- fixed `source_val_samples`: a stable held-out source batch is cached for
  evaluation
- `source_val_sampler`: a fresh source batch is drawn at each evaluation stage

The user-facing action verbs are:

- `fit(...)` for training
- `sample(...)` and `sample_with_snapshots(...)` for unconditional generation
- `transport(...)` and `transport_with_snapshots(...)` for explicit-input transport

## Inference Modes

Use `sample(...)` when you want new draws from the fitted source distribution:

```python
generated = sample(model, num_samples=256, batch_size=128, device="cpu", seed=7)
```

Use `transport(...)` when you want to push an explicit input through the learned
transport:

```python
from deep_mkv_gen import transport, transport_with_snapshots

x0 = source[0]
xT = transport(model, x0, device="cpu", seed=11)
snaps = transport_with_snapshots(model, x0, snapshot_steps=[0, 4, 8, 16], device="cpu", seed=11)
```

## Choosing a Provider

Use `W2ProviderSpec` when you want a GeomLoss-based Wasserstein objective:

```python
from deep_mkv_gen import GeomLossW2ProviderConfig, W2ProviderSpec

provider_spec = W2ProviderSpec(
    cfg=GeomLossW2ProviderConfig(
        blur=0.2,
        scaling=0.9,
    ),
    dim=2,
)
```

Use `KLProviderSpec` with `KLRatioScoreConfig` when you want the
classifier-based KL ratio estimator:

```python
from deep_mkv_gen import KLProviderSpec, KLRatioScoreConfig

provider_spec = KLProviderSpec(
    cfg=KLRatioScoreConfig(
        hidden_dim=128,
        num_layers=3,
        classifier_updates_per_step=3,
        use_replay=False,
    ),
    dim=2,
)
```

Use `KLProviderSpec` with `KLScoreDifferenceConfig` when you want the
score-difference KL construction:

```python
from deep_mkv_gen import DSMScoreConfig, KLProviderSpec, KLScoreDifferenceConfig

provider_spec = KLProviderSpec(
    cfg=KLScoreDifferenceConfig(
        score_mu_updates_per_step=2,
        target_score_cfg=DSMScoreConfig(
            hidden_dim=64,
            num_layers=2,
            sigma_noise=0.2,
            calibration_epochs=2,
            batch_size=128,
        ),
        mu_score_cfg=DSMScoreConfig(
            hidden_dim=64,
            num_layers=2,
            sigma_noise=0.2,
            calibration_epochs=0,
            batch_size=128,
        ),
    ),
    dim=2,
)
```

Use `HybridProviderSpec` when you want to combine KL and W2 branches:

```python
from deep_mkv_gen import (
    GeomLossW2ProviderConfig,
    HybridProviderConfig,
    HybridProviderSpec,
    KLRatioScoreConfig,
)

provider_spec = HybridProviderSpec(
    cfg=HybridProviderConfig(
        kl_weight=1.0,
        w2_weight=1.0,
        kl_cfg=KLRatioScoreConfig(
            hidden_dim=128,
            num_layers=3,
            classifier_updates_per_step=3,
            use_replay=False,
        ),
        w2_cfg=GeomLossW2ProviderConfig(
            blur=0.2,
            scaling=0.9,
        ),
    ),
    dim=2,
)
```

## Main Public API

The root package is intended for the high-level workflow:

- `DeepMKVGen`, `DeepMKVGenConfig`
- `RunConfig`, `RunHooks`, `EvalConfig`, `CheckpointConfig`, `KConfig`
- `RunResult`, `StageRecord`
- `fit`
- `sample`, `sample_with_snapshots`
- `transport`, `transport_with_snapshots`
- `KLProviderSpec`, `W2ProviderSpec`, `HybridProviderSpec`
- provider config dataclasses such as `KLRatioScoreConfig`,
  `KLScoreDifferenceConfig`, `GeomLossW2ProviderConfig`, and
  `HybridProviderConfig`

The root package still exposes some additional helper types today, but the
intended stable workflow is the list above.

## Lower-Level Modules

If you need lower-level access, use subpackages directly:

- `deep_mkv_gen.core`
- `deep_mkv_gen.providers`
- `deep_mkv_gen.analysis`


## Repo Layout

```text
deep-mkv-gen/
├── pyproject.toml
├── README.md
├── src/
│   └── deep_mkv_gen/
│       ├── analysis/
│       ├── core/
│       └── providers/
└── tests/
```

## Development

Run the test suite from the repo root:

```bash
python -m pytest -q
```

The package metadata lives in `pyproject.toml`.
