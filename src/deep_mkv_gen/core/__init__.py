"""Core training and evaluation surface for deep_mkv_gen."""

from .mkv_eval import StageRecord, isolated_seed, transport_eval_stats
from .mkv_gen import DeepMKVGen, DeepMKVGenConfig, Y0Net, ZNet, make_time_grid
from .metrics import (
    ClassifierRunResult,
    RepeatedClassifierResult,
    estimate_kl_classifier_from_samples,
    estimate_kl_knn_from_samples,
    repeat_kl_classifier_from_samples,
    sliced_w2_sq,
    wasserstein2,
)
from .run import (
    CheckpointConfig,
    EvalConfig,
    fit,
    KConfig,
    RunConfig,
    RunHooks,
    RunResult,
    RunState,
    StageEvaluator,
)
from .protocols import (
    ScoreComputeOutput,
    ScoreProvider,
    TargetSamplerLike,
    TerminalGradOutput,
    TerminalGradProvider,
)
from .sampling import SourceSampler, TargetSampler, sample, sample_with_snapshots, transport, transport_with_snapshots
from .mkv_train import TrainConfig
from .mkv_train import TrainEvalEvent, TrainHooks, TrainRuntime

__all__ = [
    "CheckpointConfig",
    "ClassifierRunResult",
    "EvalConfig",
    "fit",
    "KConfig",
    "DeepMKVGen",
    "DeepMKVGenConfig",
    "RepeatedClassifierResult",
    "RunConfig",
    "RunHooks",
    "RunResult",
    "RunState",
    "StageEvaluator",
    "ScoreComputeOutput",
    "ScoreProvider",
    "SourceSampler",
    "StageRecord",
    "TargetSampler",
    "TargetSamplerLike",
    "TerminalGradOutput",
    "TerminalGradProvider",
    "TrainConfig",
    "TrainEvalEvent",
    "TrainHooks",
    "TrainRuntime",
    "Y0Net",
    "ZNet",
    "isolated_seed",
    "make_time_grid",
    "estimate_kl_classifier_from_samples",
    "estimate_kl_knn_from_samples",
    "repeat_kl_classifier_from_samples",
    "sample",
    "sample_with_snapshots",
    "transport",
    "transport_with_snapshots",
    "sliced_w2_sq",
    "transport_eval_stats",
    "wasserstein2",
]
