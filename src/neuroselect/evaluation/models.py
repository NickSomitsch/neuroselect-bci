"""Typed protocols, trial records, and summaries for controlled evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from neuroselect.bci import SimulationConfig
from neuroselect.ranking import RankingDisposition
from neuroselect.synthetic import BenchmarkSplit

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
UnitRate = Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluationCondition(StrEnum):
    """Planned baselines, the current vertical slice, and required stress ablations."""

    A_BCI_ONLY = "a_bci_only"
    B_GENERIC_LANGUAGE_ONLY = "b_generic_language_only"
    C_NEURAL_LANGUAGE = "c_neural_language"
    D_NEURAL_PERSONALIZED = "d_neural_personalized"
    E_NEURAL_PERSONALIZED_RAG = "e_neural_personalized_rag"
    F_COMPLETE_SYSTEM = "f_complete_system"
    CURRENT_NEURAL_LANGUAGE_RAG = "current_neural_language_rag"
    CURRENT_SAFE_FUSION = "current_safe_fusion"
    ABLATION_UNIFORM_NEURAL = "ablation_uniform_neural"
    ABLATION_SHUFFLED_NEURAL = "ablation_shuffled_neural"
    ABLATION_REMOVE_RAG = "ablation_remove_rag"
    ABLATION_SHUFFLED_RETRIEVAL = "ablation_shuffled_retrieval"
    ABLATION_IRRELEVANT_RETRIEVAL = "ablation_irrelevant_retrieval"
    ABLATION_REMOVE_CONTEXT = "ablation_remove_context"
    ABLATION_REMOVE_RETRIEVAL_CONTEXT = "ablation_remove_retrieval_context"


class ConditionFamily(StrEnum):
    BASELINE = "baseline"
    CURRENT_SYSTEM = "current_system"
    ABLATION = "ablation"


class ConditionAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class RankingMode(StrEnum):
    NEURAL_ONLY = "neural_only"
    LANGUAGE_ONLY = "language_only"
    WEIGHTED_BASELINE = "weighted_baseline"
    TRANSPARENT_SAFE_FUSION = "transparent_safe_fusion"


class NeuralMode(StrEnum):
    SIMULATED = "simulated"
    MISSING = "missing"
    UNIFORM = "uniform"
    SHUFFLED = "shuffled"


class RetrievalMode(StrEnum):
    NONE = "none"
    CURRENT = "current"
    SHUFFLED = "shuffled"
    IRRELEVANT = "irrelevant"


class ConditionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: EvaluationCondition
    label: str = Field(min_length=1, max_length=160)
    family: ConditionFamily
    availability: ConditionAvailability
    ranking_mode: RankingMode
    neural_mode: NeuralMode
    retrieval_mode: RetrievalMode
    personalization_enabled: bool = False
    safeguards_enabled: bool
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_availability_reason(self) -> ConditionDefinition:
        unavailable = self.availability is ConditionAvailability.UNAVAILABLE
        if unavailable != (self.unavailable_reason is not None):
            raise ValueError("only unavailable conditions require an unavailable reason")
        return self


class EvaluationTiming(BaseModel):
    """Deterministic interaction-time model; never confused with wall-clock runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_round_seconds: float = Field(default=2.0, gt=0.0, le=60.0)
    explicit_action_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    enhanced_confirmation_seconds: float = Field(default=1.0, gt=0.0, le=60.0)


class SimulatedExperimentSpec(BaseModel):
    """Versioned inputs for one deterministic held-out simulation experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    experiment_id: Identifier
    protocol_revision: Identifier = "controlled-fusion-evaluation-v2"
    seed: int = Field(ge=0)
    split: BenchmarkSplit = BenchmarkSplit.TEST
    profile_ids: tuple[Identifier, ...] = Field(min_length=1)
    message_limit_per_profile: int = Field(default=4, ge=1, le=1_000)
    candidate_count: Literal[4, 6, 8, 12] = 8
    maximum_phrase_tokens: int = Field(default=4, ge=1, le=8)
    conditions: tuple[EvaluationCondition, ...] = Field(min_length=1)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    language_conflict_every_n_trials: int = Field(default=4, ge=1, le=1_000)
    evaluation_time: datetime
    timing: EvaluationTiming = Field(default_factory=EvaluationTiming)
    simulator: SimulationConfig = Field(default_factory=SimulationConfig)

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> SimulatedExperimentSpec:
        if len(self.profile_ids) != len(set(self.profile_ids)):
            raise ValueError("profile IDs must be unique")
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("evaluation conditions must be unique")
        if self.evaluation_time.tzinfo is None or self.evaluation_time.utcoffset() is None:
            raise ValueError("evaluation_time must include a timezone")
        if self.simulator.seed != self.seed:
            raise ValueError("simulator seed must equal the experiment seed")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class TrialRecord(BaseModel):
    """One condition applied to one target span with explicit mapping provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: Identifier
    condition: EvaluationCondition
    profile_id: Identifier
    eeg_subject_id: Identifier | None = None
    eeg_session_id: Identifier | None = None
    message_id: Identifier
    span_index: int = Field(ge=0)
    message_span_count: int = Field(ge=1)
    confirmed_context: str = Field(max_length=4_000)
    retrieval_query_context_removed: bool = False
    target_text: str = Field(min_length=1, max_length=160)
    target_word_count: int = Field(ge=1, le=8)
    candidate_ids: tuple[Identifier, ...] = Field(min_length=4)
    ranked_candidate_ids: tuple[Identifier, ...] = Field(min_length=4)
    # Retained as the v1 compatibility name for the candidate that received the
    # recorded/simulated target signature. New consumers should use the explicit
    # intended_candidate_id and mapped_target_candidate_id fields.
    target_candidate_id: Identifier
    intended_candidate_id: Identifier | None = None
    mapped_target_candidate_id: Identifier | None = None
    fallback_candidate_id: Identifier | None = None
    target_available: bool = True
    target_rank: int | None = Field(default=None, ge=1)
    mapped_target_rank: int | None = Field(default=None, ge=1)
    top_candidate_id: Identifier
    neural_top_candidate_id: Identifier | None
    language_top_candidate_id: Identifier
    disposition: RankingDisposition
    reason_codes: tuple[str, ...] = ()
    language_conflict_context: bool
    neural_language_conflict: bool
    neural_target_probability: UnitRate | None = None
    prediction_confidence: UnitRate | None = None
    prediction_correct: bool | None = None
    neural_brier_score: UnitRate | None = None
    top_1_correct: bool
    top_3_correct: bool
    explicit_selection_completed: bool
    fallback_selected: bool = False
    fallback_selection_completed: bool = False
    enhanced_confirmation_required: bool
    correction_required: bool
    incorrect_display: bool = False
    candidate_generation_failed: bool = False
    explicit_action_count: int = Field(ge=0)
    baseline_selection_count: int = Field(default=1, ge=1)
    modeled_selection_count: int = Field(default=0, ge=0)
    selection_savings: int | None = None
    unintended_word: bool = False
    automatic_selection_permitted: Literal[False] = False
    retrieval_hit_count: int = Field(ge=0)
    source_flash_duration_seconds: float | None = Field(default=None, gt=0.0)
    modeled_duration_seconds: float = Field(gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def upgrade_v1_record(cls, value: Any) -> Any:
        """Infer additive v2 fields when reading a v1 trial record."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        mapped_id = payload.get("mapped_target_candidate_id") or payload.get("target_candidate_id")
        payload.setdefault("mapped_target_candidate_id", mapped_id)
        payload.setdefault("mapped_target_rank", payload.get("target_rank"))
        target_available = bool(payload.get("target_available", True))
        if target_available:
            payload.setdefault("intended_candidate_id", payload.get("target_candidate_id"))
        else:
            payload.setdefault("intended_candidate_id", None)
            payload.setdefault("target_rank", None)
            payload.setdefault("fallback_candidate_id", mapped_id)
        top_candidate_id = payload.get("top_candidate_id")
        disposition = payload.get("disposition")
        displayed = disposition in {
            RankingDisposition.DISPLAY,
            RankingDisposition.DISPLAY.value,
        }
        expected_id = (
            payload.get("intended_candidate_id")
            if target_available
            else payload.get("fallback_candidate_id")
        )
        payload.setdefault(
            "fallback_selected",
            not target_available and top_candidate_id == payload.get("fallback_candidate_id"),
        )
        payload.setdefault(
            "fallback_selection_completed",
            not target_available
            and displayed
            and top_candidate_id == payload.get("fallback_candidate_id"),
        )
        payload.setdefault(
            "incorrect_display",
            displayed and expected_id is not None and top_candidate_id != expected_id,
        )
        baseline_count = max(1, len(str(payload.get("target_text", "")).strip()))
        payload.setdefault("baseline_selection_count", baseline_count)
        modeled_count = int(payload.get("explicit_action_count", 0))
        payload.setdefault("modeled_selection_count", modeled_count)
        payload.setdefault(
            "selection_savings",
            baseline_count - modeled_count
            if bool(payload.get("explicit_selection_completed", False))
            else None,
        )
        return payload

    @model_validator(mode="after")
    def validate_candidate_alignment(self) -> TrialRecord:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("trial candidate IDs must be unique")
        if self.target_candidate_id not in self.candidate_ids:
            raise ValueError("the mapped target must remain in the visible candidate set")
        if self.mapped_target_candidate_id not in self.candidate_ids:
            raise ValueError("mapped target candidate must be visible")
        if self.mapped_target_candidate_id != self.target_candidate_id:
            raise ValueError("v1 target alias must equal mapped_target_candidate_id")
        if self.top_candidate_id not in self.candidate_ids:
            raise ValueError("the top candidate must be visible")
        if set(self.ranked_candidate_ids) != set(self.candidate_ids):
            raise ValueError(
                "ranked candidate IDs must contain every visible candidate exactly once"
            )
        if self.ranked_candidate_ids[0] != self.top_candidate_id:
            raise ValueError("top candidate must be first in ranked candidate IDs")
        if self.mapped_target_rank is None or (
            self.ranked_candidate_ids[self.mapped_target_rank - 1]
            != self.mapped_target_candidate_id
        ):
            raise ValueError("mapped target rank must identify the mapped target")
        if self.target_available:
            if self.intended_candidate_id is None or self.intended_candidate_id not in (
                self.candidate_ids
            ):
                raise ValueError("available intended candidate must be visible")
            if self.intended_candidate_id != self.target_candidate_id:
                raise ValueError("available intended candidate must receive the mapped signature")
            if self.target_rank is None or (
                self.ranked_candidate_ids[self.target_rank - 1] != self.intended_candidate_id
            ):
                raise ValueError("target rank must identify the intended candidate")
            if self.fallback_candidate_id is not None:
                raise ValueError("available targets cannot use a fallback candidate")
        else:
            if self.intended_candidate_id is not None or self.target_rank is not None:
                raise ValueError("unavailable targets cannot have an intended candidate or rank")
            if (
                self.fallback_candidate_id is None
                or self.fallback_candidate_id != self.mapped_target_candidate_id
            ):
                raise ValueError("unavailable targets must map to the explicit fallback")
            if self.top_1_correct or self.top_3_correct or self.explicit_selection_completed:
                raise ValueError("fallback selection cannot count as intended-target completion")
        if self.top_1_correct != (self.target_available and self.target_rank == 1):
            raise ValueError("top_1_correct must reflect intended-target rank")
        if self.top_3_correct != (
            self.target_available and self.target_rank is not None and self.target_rank <= 3
        ):
            raise ValueError("top_3_correct must reflect intended-target rank")
        if self.fallback_selected and (
            self.target_available or self.top_candidate_id != self.fallback_candidate_id
        ):
            raise ValueError("fallback_selected requires an unavailable target and top fallback")
        if self.fallback_selection_completed and (
            not self.fallback_selected or self.disposition is not RankingDisposition.DISPLAY
        ):
            raise ValueError("fallback completion requires a displayed top fallback")
        if self.explicit_selection_completed and (
            self.disposition is not RankingDisposition.DISPLAY or not self.top_1_correct
        ):
            raise ValueError("target completion requires a displayed correct intended target")
        if self.selection_savings is not None and not self.explicit_selection_completed:
            raise ValueError("selection savings are defined only for completed intended targets")
        if self.selection_savings is not None and self.selection_savings != (
            self.baseline_selection_count - self.modeled_selection_count
        ):
            raise ValueError("selection savings must match baseline minus modeled selections")
        calibration_values = (
            self.prediction_confidence,
            self.prediction_correct,
            self.neural_brier_score,
        )
        if any(value is None for value in calibration_values) and not all(
            value is None for value in calibration_values
        ):
            raise ValueError("neural calibration fields must be present or absent together")
        return self


class ConditionMetrics(BaseModel):
    """Aggregate metrics for either all profiles or one synthetic profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: EvaluationCondition
    profile_id: Identifier | None = None
    trial_count: int = Field(ge=1)
    message_count: int = Field(ge=1)
    completed_trial_count: int = Field(ge=0)
    completed_message_count: int = Field(ge=0)
    available_trial_count: int = Field(default=0, ge=0)
    fallback_trial_count: int = Field(default=0, ge=0)
    displayed_trial_count: int = Field(default=0, ge=0)
    candidate_generation_failure_count: int = Field(default=0, ge=0)
    target_availability_rate: UnitRate
    top_1_candidate_recall: UnitRate
    top_3_candidate_recall: UnitRate
    top_1_recall_given_available: UnitRate | None = None
    top_3_recall_given_available: UnitRate | None = None
    other_fallback_success_rate: UnitRate | None = None
    selection_completion_rate: UnitRate
    final_message_exact_accuracy: UnitRate
    correct_selections_per_minute: float = Field(ge=0.0)
    words_per_minute: float = Field(ge=0.0)
    selections_per_completed_message: float | None = Field(default=None, ge=0.0)
    mean_selection_savings: float | None = None
    unintended_word_rate: UnitRate
    incorrect_display_rate: UnitRate = 0.0
    candidate_generation_failure_rate: UnitRate = 0.0
    display_accuracy: UnitRate | None = None
    selective_risk: UnitRate | None = None
    correction_rate: UnitRate
    abstention_rate: UnitRate
    repeat_request_rate: UnitRate
    neural_expected_calibration_error: UnitRate | None = None
    neural_multiclass_brier_score: UnitRate | None = None
    mean_modeled_latency_seconds: float = Field(gt=0.0)
    conflict_trial_count: int = Field(ge=0)
    conflict_top_1_recall: UnitRate | None = None
    conflict_target_availability_rate: UnitRate | None = None
    automatic_selection_violation_count: Literal[0] = 0

    @model_validator(mode="before")
    @classmethod
    def upgrade_v1_metrics(cls, value: Any) -> Any:
        """Infer additive v2 summaries when loading legacy metric payloads."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        trial_count = int(payload.get("trial_count", 0))
        availability = float(payload.get("target_availability_rate", 1.0))
        available_count = round(trial_count * availability)
        fallback_count = trial_count - available_count
        payload.setdefault("available_trial_count", available_count)
        payload.setdefault("fallback_trial_count", fallback_count)
        if available_count:
            scale = trial_count / available_count
            payload.setdefault(
                "top_1_recall_given_available",
                min(1.0, float(payload.get("top_1_candidate_recall", 0.0)) * scale),
            )
            payload.setdefault(
                "top_3_recall_given_available",
                min(1.0, float(payload.get("top_3_candidate_recall", 0.0)) * scale),
            )
        else:
            payload.setdefault("top_1_recall_given_available", None)
            payload.setdefault("top_3_recall_given_available", None)
        payload.setdefault(
            "other_fallback_success_rate",
            0.0 if fallback_count else None,
        )
        abstention = float(payload.get("abstention_rate", 0.0))
        repeat = float(payload.get("repeat_request_rate", 0.0))
        displayed_count = max(
            int(payload.get("completed_trial_count", 0)),
            round(trial_count * max(0.0, 1.0 - abstention - repeat)),
        )
        payload.setdefault("displayed_trial_count", displayed_count)
        payload.setdefault("candidate_generation_failure_count", 0)
        payload.setdefault("candidate_generation_failure_rate", 0.0)
        payload.setdefault(
            "incorrect_display_rate",
            float(payload.get("correction_rate", 0.0)),
        )
        if displayed_count:
            display_accuracy = min(
                1.0,
                int(payload.get("completed_trial_count", 0)) / displayed_count,
            )
            payload.setdefault("display_accuracy", display_accuracy)
            payload.setdefault("selective_risk", 1.0 - display_accuracy)
        else:
            payload.setdefault("display_accuracy", None)
            payload.setdefault("selective_risk", None)
        payload.setdefault("mean_selection_savings", None)
        return payload

    @model_validator(mode="after")
    def validate_counts_and_conflict_slice(self) -> ConditionMetrics:
        if self.completed_trial_count > self.trial_count:
            raise ValueError("completed trial count cannot exceed trial count")
        if self.completed_message_count > self.message_count:
            raise ValueError("completed message count cannot exceed message count")
        if self.conflict_trial_count > self.trial_count:
            raise ValueError("conflict trial count cannot exceed trial count")
        for count_name in (
            "available_trial_count",
            "fallback_trial_count",
            "displayed_trial_count",
            "candidate_generation_failure_count",
        ):
            if getattr(self, count_name) > self.trial_count:
                raise ValueError(f"{count_name} cannot exceed trial count")
        if self.available_trial_count + self.fallback_trial_count != self.trial_count:
            raise ValueError("available and fallback trials must cover every trial")
        if (self.available_trial_count == 0) != (
            self.top_1_recall_given_available is None and self.top_3_recall_given_available is None
        ):
            raise ValueError("conditional recall is present exactly when targets are available")
        if (self.fallback_trial_count == 0) != (self.other_fallback_success_rate is None):
            raise ValueError("fallback success is present exactly when fallback trials exist")
        if self.displayed_trial_count == 0:
            if self.display_accuracy is not None or self.selective_risk is not None:
                raise ValueError("empty displayed slice cannot have selective-risk metrics")
        elif (
            self.display_accuracy is None
            or self.selective_risk is None
            or abs(self.selective_risk - (1.0 - self.display_accuracy)) > 1e-12
        ):
            raise ValueError("display accuracy and selective risk must be complementary")
        conflict_values = (
            self.conflict_top_1_recall,
            self.conflict_target_availability_rate,
        )
        if self.conflict_trial_count == 0 and any(value is not None for value in conflict_values):
            raise ValueError("an empty conflict slice cannot have conflict metrics")
        if self.conflict_trial_count > 0 and any(value is None for value in conflict_values):
            raise ValueError("a populated conflict slice requires conflict metrics")
        return self


class ExperimentResult(BaseModel):
    """Canonical in-memory result before deterministic artifact serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.1"] = "1.1"
    run_id: Identifier
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: SimulatedExperimentSpec
    condition_catalog: tuple[ConditionDefinition, ...]
    trial_records: tuple[TrialRecord, ...] = Field(min_length=1)
    metrics: tuple[ConditionMetrics, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance_and_condition_coverage(self) -> ExperimentResult:
        if self.config_sha256 != self.spec.digest():
            raise ValueError("config SHA-256 must match the embedded experiment specification")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        trial_conditions = {record.condition for record in self.trial_records}
        if trial_conditions != set(self.spec.conditions):
            raise ValueError("trial records must cover every requested condition")
        overall_conditions = {
            metric.condition for metric in self.metrics if metric.profile_id is None
        }
        if overall_conditions != set(self.spec.conditions):
            raise ValueError("overall metrics must cover every requested condition")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
