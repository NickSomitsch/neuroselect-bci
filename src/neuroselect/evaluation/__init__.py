"""Reproducible controlled experiments over the simulated vertical slice."""

from neuroselect.evaluation.artifacts import capture_runtime_environment, write_experiment_artifacts
from neuroselect.evaluation.conditions import condition_by_id, condition_catalog
from neuroselect.evaluation.counterfactual import (
    CounterfactualConfigurationError,
    CounterfactualFusionRunner,
    flash_trials_from_decoder_evaluation,
    load_counterfactual_input,
    load_counterfactual_spec,
    shuffle_retrieval_across_candidates,
)
from neuroselect.evaluation.counterfactual_artifacts import (
    read_counterfactual_artifacts,
    write_counterfactual_artifacts,
)
from neuroselect.evaluation.counterfactual_models import (
    CounterfactualExperimentInput,
    CounterfactualFusionResult,
    CounterfactualFusionSpec,
    CounterfactualFusionTrial,
    CounterfactualTrialProvenance,
    PairedBootstrapInterval,
)
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
    "CounterfactualConfigurationError",
    "CounterfactualExperimentInput",
    "CounterfactualFusionResult",
    "CounterfactualFusionRunner",
    "CounterfactualFusionSpec",
    "CounterfactualFusionTrial",
    "CounterfactualTrialProvenance",
    "EvaluationCondition",
    "EvaluationTiming",
    "ExperimentConfigurationError",
    "ExperimentResult",
    "NeuralMode",
    "PairedBootstrapInterval",
    "RankingMode",
    "RetrievalMode",
    "SimulatedExperimentRunner",
    "SimulatedExperimentSpec",
    "TrialRecord",
    "calculate_metrics",
    "capture_runtime_environment",
    "condition_by_id",
    "condition_catalog",
    "expected_calibration_error",
    "flash_trials_from_decoder_evaluation",
    "load_counterfactual_input",
    "load_counterfactual_spec",
    "load_experiment_spec",
    "read_counterfactual_artifacts",
    "shuffle_retrieval_across_candidates",
    "write_counterfactual_artifacts",
    "write_experiment_artifacts",
]
