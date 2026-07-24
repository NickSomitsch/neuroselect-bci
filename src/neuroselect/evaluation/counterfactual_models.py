"""Typed inputs and outputs for offline counterfactual fusion experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from neuroselect.bci import FlashLayout, FlashProbabilityTrial, TileAggregationConfig
from neuroselect.core.models import CandidateKind, CandidateSet
from neuroselect.evaluation.models import (
    ConditionMetrics,
    EvaluationCondition,
    EvaluationTiming,
    TrialRecord,
)
from neuroselect.retrieval import CandidateRetrievalEvidence

COUNTERFACTUAL_CONDITIONS = frozenset(
    {
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
        EvaluationCondition.C_NEURAL_LANGUAGE,
        EvaluationCondition.D_NEURAL_PERSONALIZED,
        EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
        EvaluationCondition.F_COMPLETE_SYSTEM,
        EvaluationCondition.ABLATION_UNIFORM_NEURAL,
        EvaluationCondition.ABLATION_SHUFFLED_NEURAL,
        EvaluationCondition.ABLATION_REMOVE_RAG,
        EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL,
        EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL,
        EvaluationCondition.ABLATION_REMOVE_CONTEXT,
        EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
    }
)


def _normalize_unordered_values(value: object) -> object:
    """Convert sets to deterministically ordered JSON-compatible collections."""

    if isinstance(value, dict):
        return {key: _normalize_unordered_values(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        normalized = [_normalize_unordered_values(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                to_jsonable_python(item),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, list | tuple):
        return [_normalize_unordered_values(item) for item in value]
    return value


def canonical_counterfactual_json(value: object) -> str:
    """Serialize counterfactual values with stable mapping and set ordering."""

    normalized = _normalize_unordered_values(value)
    return json.dumps(
        to_jsonable_python(normalized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class CounterfactualFusionSpec(BaseModel):
    """Locked paired-condition protocol over recorded P300 event probabilities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "2.0"] = "2.0"
    experiment_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal[
        "offline-counterfactual-fusion-v1",
        "offline-counterfactual-fusion-v2",
    ] = "offline-counterfactual-fusion-v2"
    seed: int = Field(default=20260721, ge=0)
    conditions: tuple[EvaluationCondition, ...] = Field(min_length=1)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    bootstrap_resamples: int = Field(default=2_000, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    personalization_evidence_kind: Literal["held_out_adapter", "controlled_fixture"]
    aggregation: TileAggregationConfig = Field(default_factory=TileAggregationConfig)
    timing: EvaluationTiming = Field(default_factory=EvaluationTiming)

    @model_validator(mode="after")
    def validate_conditions(self) -> CounterfactualFusionSpec:
        expected_protocol = f"offline-counterfactual-fusion-v{self.schema_version[0]}"
        if self.protocol_revision != expected_protocol:
            raise ValueError("counterfactual schema version and protocol revision must agree")
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("counterfactual conditions must be unique")
        unsupported = set(self.conditions) - COUNTERFACTUAL_CONDITIONS
        if unsupported:
            raise ValueError(f"unsupported counterfactual conditions: {sorted(unsupported)}")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class CounterfactualFusionTrial(BaseModel):
    """One fixed candidate round plus recorded flashes and precomputed non-neural evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str = Field(min_length=1, max_length=160)
    synthetic_profile_id: str | None = Field(default=None, min_length=1, max_length=160)
    message_id: str | None = Field(default=None, min_length=1, max_length=160)
    span_index: int | None = Field(default=None, ge=0)
    message_span_count: int | None = Field(default=None, ge=1)
    candidate_set: CandidateSet
    flash_layout: FlashLayout
    flash_trial: FlashProbabilityTrial
    intended_text: str | None = Field(default=None, min_length=1, max_length=160)
    intended_candidate_id: str | None = None
    other_candidate_id: str = Field(min_length=1, max_length=128)
    candidate_generation_failed: bool = False
    confirmed_context: str = Field(default="", max_length=4_000)
    generic_language_support: dict[str, float]
    no_context_language_support: dict[str, float] | None = None
    personalization_lift: dict[str, float] = Field(default_factory=dict)
    personalization_adapter_id: str | None = Field(default=None, min_length=1, max_length=160)
    personalization_adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    retrieval_evidence: tuple[CandidateRetrievalEvidence, ...] = ()
    no_context_retrieval_evidence: tuple[CandidateRetrievalEvidence, ...] | None = None
    irrelevant_retrieval_evidence: tuple[CandidateRetrievalEvidence, ...] | None = None

    @model_validator(mode="after")
    def validate_alignment(self) -> CounterfactualFusionTrial:
        candidate_ids = tuple(item.candidate_id for item in self.candidate_set.candidates)
        if candidate_ids != self.flash_layout.candidate_ids:
            raise ValueError("candidate order must exactly match the fixed flash layout")
        if (
            self.intended_candidate_id is not None
            and self.intended_candidate_id not in candidate_ids
        ):
            raise ValueError("intended candidate must be visible when supplied")
        if self.other_candidate_id not in candidate_ids:
            raise ValueError("counterfactual fallback requires a visible Other candidate")
        other = next(
            item
            for item in self.candidate_set.candidates
            if item.candidate_id == self.other_candidate_id
        )
        normalized_other = other.text.casefold().strip().rstrip(".…").strip()
        if other.kind is not CandidateKind.CONTROL or normalized_other != "other":
            raise ValueError("other_candidate_id must identify the visible Other control")
        if self.intended_candidate_id is not None and self.intended_text is not None:
            intended_candidate = next(
                item
                for item in self.candidate_set.candidates
                if item.candidate_id == self.intended_candidate_id
            )
            if " ".join(intended_candidate.text.casefold().split()) != " ".join(
                self.intended_text.casefold().split()
            ):
                raise ValueError("intended text must match the visible intended candidate")
        message_values = (
            self.synthetic_profile_id,
            self.message_id,
            self.span_index,
            self.message_span_count,
        )
        if any(value is None for value in message_values) and not all(
            value is None for value in message_values
        ):
            raise ValueError("message and synthetic-profile provenance must be present together")
        if (
            self.span_index is not None
            and self.message_span_count is not None
            and self.span_index >= self.message_span_count
        ):
            raise ValueError("span_index must be smaller than message_span_count")
        language_ids = {
            item.candidate_id
            for item in self.candidate_set.candidates
            if item.kind is not CandidateKind.CONTROL
        }
        self._validate_language_support(self.generic_language_support, language_ids, "generic")
        if self.no_context_language_support is not None:
            self._validate_language_support(
                self.no_context_language_support, language_ids, "no-context"
            )
        if not set(self.personalization_lift).issubset(language_ids):
            raise ValueError("personalization lift may reference only language candidates")
        if any(not -1.0 <= value <= 1.0 for value in self.personalization_lift.values()):
            raise ValueError("personalization lifts must lie in [-1, 1]")
        if (self.personalization_adapter_id is None) != (
            self.personalization_adapter_sha256 is None
        ):
            raise ValueError("personalization adapter ID and checksum must be present together")
        for name, values in (
            ("retrieval", self.retrieval_evidence),
            ("no-context retrieval", self.no_context_retrieval_evidence or ()),
            ("irrelevant retrieval", self.irrelevant_retrieval_evidence or ()),
        ):
            ids = [item.candidate_id for item in values]
            if len(ids) != len(set(ids)) or not set(ids).issubset(language_ids):
                raise ValueError(f"{name} evidence must uniquely reference language candidates")
        return self

    @staticmethod
    def _validate_language_support(
        values: dict[str, float], expected_ids: set[str], name: str
    ) -> None:
        if set(values) != expected_ids:
            raise ValueError(f"{name} language support must cover exactly language candidates")
        if (
            any(not 0.0 <= value <= 1.0 for value in values.values())
            or abs(sum(values.values()) - 1.0) > 1e-9
        ):
            raise ValueError(f"{name} language support must be a probability distribution")

    @property
    def resolved_target_candidate_id(self) -> str:
        return self.intended_candidate_id or self.other_candidate_id

    @property
    def target_available(self) -> bool:
        return self.intended_candidate_id is not None

    @property
    def resolved_profile_id(self) -> str:
        return self.synthetic_profile_id or self.flash_trial.subject_id

    @property
    def resolved_message_id(self) -> str:
        return self.message_id or self.trial_id

    @property
    def resolved_intended_text(self) -> str:
        if self.intended_text is not None:
            return self.intended_text
        candidate = next(
            item
            for item in self.candidate_set.candidates
            if item.candidate_id == self.resolved_target_candidate_id
        )
        return candidate.text


class CounterfactualExperimentInput(BaseModel):
    """Portable, checksum-addressable input to a replay/fusion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "2.0"] = "2.0"
    prepared_at: datetime
    preparation_revision: str | None = Field(default=None, min_length=1, max_length=160)
    preparation_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_decoder_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_task_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_language_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_language_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_evidence_claim_eligible: bool = True
    preparation_limitations: tuple[str, ...] = ()
    spec: CounterfactualFusionSpec
    trials: tuple[CounterfactualFusionTrial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_input(self) -> CounterfactualExperimentInput:
        if self.schema_version != self.spec.schema_version:
            raise ValueError("counterfactual input and specification schema versions must agree")
        if self.prepared_at.tzinfo is None or self.prepared_at.utcoffset() is None:
            raise ValueError("counterfactual input preparation time must include a timezone")
        if (self.preparation_revision is None) != (self.preparation_config_sha256 is None):
            raise ValueError("preparation revision and checksum must be present together")
        if (self.source_language_manifest_sha256 is None) != (
            self.source_language_result_sha256 is None
        ):
            raise ValueError("language manifest and result checksums must be present together")
        identifiers = [trial.trial_id for trial in self.trials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("counterfactual input trial IDs must be unique")
        source_trials = [
            (
                trial.flash_trial.subject_id,
                trial.flash_trial.session_id,
                trial.flash_trial.selection_trial_id,
            )
            for trial in self.trials
        ]
        if len(source_trials) != len(set(source_trials)):
            raise ValueError("one recorded selection trial may be mapped only once per input")
        if len({len(trial.candidate_set.candidates) for trial in self.trials}) != 1:
            raise ValueError("paired counterfactual trials must use one fixed candidate count")
        if self.schema_version == "2.0":
            incomplete = [
                trial.trial_id
                for trial in self.trials
                if trial.synthetic_profile_id is None
                or trial.message_id is None
                or trial.span_index is None
                or trial.message_span_count is None
                or trial.intended_text is None
            ]
            if incomplete:
                raise ValueError(
                    "v2 counterfactual trials require target, message, and profile provenance: "
                    f"{incomplete[:3]}"
                )
            message_spans: dict[tuple[str, str], set[int]] = {}
            message_counts: dict[tuple[str, str], set[int]] = {}
            for trial in self.trials:
                assert trial.synthetic_profile_id is not None
                assert trial.message_id is not None
                assert trial.span_index is not None
                assert trial.message_span_count is not None
                key = trial.synthetic_profile_id, trial.message_id
                message_spans.setdefault(key, set()).add(trial.span_index)
                message_counts.setdefault(key, set()).add(trial.message_span_count)
            incomplete_messages = [
                key
                for key, spans in message_spans.items()
                if len(message_counts[key]) != 1
                or spans != set(range(next(iter(message_counts[key]))))
            ]
            if incomplete_messages:
                raise ValueError(
                    "v2 counterfactual inputs must contain every span for each message: "
                    f"{incomplete_messages[:3]}"
                )
        adapter_hashes: dict[str, set[str]] = {}
        for trial in self.trials:
            if (
                trial.personalization_adapter_id is not None
                and trial.personalization_adapter_sha256 is not None
            ):
                adapter_hashes.setdefault(trial.personalization_adapter_id, set()).add(
                    trial.personalization_adapter_sha256
                )
        if any(len(values) != 1 for values in adapter_hashes.values()):
            raise ValueError("one personalization adapter ID cannot reference multiple checksums")
        return self

    def canonical_json(self) -> str:
        """Return stable JSON, including deterministic ordering for provenance sets."""

        return canonical_counterfactual_json(self.model_dump(mode="python"))

    def digest(self) -> str:
        if self.schema_version == "2.0":
            return hashlib.sha256(self.canonical_json().encode()).hexdigest()
        payload = self.model_dump(mode="json")
        if self.schema_version == "1.0":
            for field in (
                "preparation_revision",
                "preparation_config_sha256",
                "source_language_manifest_sha256",
                "source_language_result_sha256",
                "source_evidence_claim_eligible",
                "preparation_limitations",
            ):
                payload.pop(field, None)
            v2_trial_fields = (
                "synthetic_profile_id",
                "message_id",
                "span_index",
                "message_span_count",
                "intended_text",
                "candidate_generation_failed",
            )
            for trial in payload["trials"]:
                for field in v2_trial_fields:
                    trial.pop(field, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class CounterfactualTrialProvenance(BaseModel):
    """Evidence that counterfactual mapping retained the original event stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_trial_id: str
    subject_id: str
    session_id: str
    synthetic_profile_id: str | None = None
    message_id: str | None = None
    span_index: int | None = Field(default=None, ge=0)
    message_span_count: int | None = Field(default=None, ge=1)
    intended_text: str | None = Field(default=None, min_length=1, max_length=160)
    intended_candidate_id: str | None = None
    event_ids: tuple[str, ...] = Field(min_length=2)
    event_onsets_seconds: tuple[float, ...] = Field(min_length=2)
    recorded_target_codes: tuple[int, ...] = Field(min_length=1)
    mapped_target_candidate_id: str
    intended_candidate_was_absent: bool
    fallback_candidate_id: str | None = None
    source_flash_duration_seconds: float | None = Field(default=None, gt=0.0)
    source_layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapped_layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    neural_evidence_id: str


class PairedBootstrapInterval(BaseModel):
    """Deterministic paired hierarchical-bootstrap comparison against condition F."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: EvaluationCondition
    reference_condition: Literal[EvaluationCondition.F_COMPLETE_SYSTEM] = (
        EvaluationCondition.F_COMPLETE_SYSTEM
    )
    metric: Literal["top_1_candidate_recall", "selection_completion_rate"]
    observed_delta: float = Field(ge=-1.0, le=1.0)
    lower_bound: float = Field(ge=-1.0, le=1.0)
    upper_bound: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    resamples: int = Field(ge=100)

    @model_validator(mode="after")
    def validate_interval(self) -> PairedBootstrapInterval:
        if self.lower_bound > self.upper_bound:
            raise ValueError("bootstrap interval bounds must be ordered")
        return self


class CounterfactualFusionResult(BaseModel):
    """Machine-readable counterfactual records kept separate from original-task EEG results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "2.0"] = "2.0"
    run_id: str = Field(min_length=1, max_length=160)
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_decoder_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_task_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_language_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_language_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    personalization_adapters: dict[str, str] = Field(default_factory=dict)
    spec: CounterfactualFusionSpec
    mapping_provenance: tuple[CounterfactualTrialProvenance, ...] = Field(min_length=1)
    trial_records: tuple[TrialRecord, ...] = Field(min_length=1)
    metrics: tuple[ConditionMetrics, ...] = Field(min_length=1)
    paired_intervals: tuple[PairedBootstrapInterval, ...] = ()
    claim_eligible: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> CounterfactualFusionResult:
        if self.schema_version != self.spec.schema_version:
            raise ValueError("counterfactual result and specification schema versions must agree")
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("counterfactual result time must include a timezone")
        if self.config_sha256 != self.spec.digest():
            raise ValueError("counterfactual result config hash must match the embedded spec")
        if (self.source_language_manifest_sha256 is None) != (
            self.source_language_result_sha256 is None
        ):
            raise ValueError("language manifest and result checksums must be present together")
        if self.spec.personalization_evidence_kind == "controlled_fixture" and self.claim_eligible:
            raise ValueError("controlled personalization fixtures cannot be claim-eligible")
        conditions = {record.condition for record in self.trial_records}
        if conditions != set(self.spec.conditions):
            raise ValueError("counterfactual records must cover every requested condition")
        overall = {metric.condition for metric in self.metrics if metric.profile_id is None}
        if overall != set(self.spec.conditions):
            raise ValueError("counterfactual metrics must cover every requested condition")
        return self
