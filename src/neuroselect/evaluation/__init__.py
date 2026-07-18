"""Reproducible controlled experiments over the simulated vertical slice."""

from neuroselect.evaluation.artifacts import write_experiment_artifacts
from neuroselect.evaluation.conditions import condition_by_id, condition_catalog
from neuroselect.evaluation.metrics import calculate_metrics, expected_calibration_error
from neuroselect.evaluation.models import (
    ConditionAvailability,
    ConditionDefinition,
    ConditionFamily,
    ConditionMetrics,
    EvaluationCondition,
    EvaluationTiming,
    ExperimentResult,
    NeuralMode,
    RankingMode,
    RetrievalMode,
    SimulatedExperimentSpec,
    TrialRecord,
)
from neuroselect.evaluation.runner import (
    ExperimentConfigurationError,
    SimulatedExperimentRunner,
    load_experiment_spec,
)

__all__ = [
    "ConditionAvailability",
    "ConditionDefinition",
    "ConditionFamily",
    "ConditionMetrics",
    "EvaluationCondition",
    "EvaluationTiming",
    "ExperimentConfigurationError",
    "ExperimentResult",
    "NeuralMode",
    "RankingMode",
    "RetrievalMode",
    "SimulatedExperimentRunner",
    "SimulatedExperimentSpec",
    "TrialRecord",
    "calculate_metrics",
    "condition_by_id",
    "condition_catalog",
    "expected_calibration_error",
    "load_experiment_spec",
    "write_experiment_artifacts",
]
