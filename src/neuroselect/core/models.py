"""Domain contracts that keep neural, language, retrieval, and user evidence separate."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class KnowledgeKind(StrEnum):
    RELATIONSHIP = "relationship"
    ROUTINE = "routine"
    PREFERENCE = "preference"
    PHRASEBOOK = "phrasebook"
    CURRENT_EVENT = "current_event"


class RecordPermission(StrEnum):
    SUGGEST = "suggest"
    EXPLAIN = "explain"


class CandidateKind(StrEnum):
    WORD = "word"
    PHRASE = "phrase"
    CONTROL = "control"
    CHARACTER = "character"


class Candidate(BaseModel):
    """A visible, immutable selection option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: Identifier
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    kind: CandidateKind
    origins: frozenset[Identifier] = Field(default_factory=frozenset)
    risk_tags: frozenset[Identifier] = Field(default_factory=frozenset)
    retrieval_record_ids: tuple[Identifier, ...] = ()


class CandidateSet(BaseModel):
    """Candidate tiles whose order is fixed for a complete selection round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_set_id: Identifier
    context_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    candidates: tuple[Candidate, ...] = Field(min_length=1)
    generator_revision: Identifier
    prompt_revision: Identifier

    @model_validator(mode="after")
    def require_unique_candidates(self) -> CandidateSet:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique within a candidate set")
        return self


class EvidenceMode(StrEnum):
    MANUAL = "manual"
    SIMULATION = "simulation"
    REPLAY = "replay"
    LIVE = "live"


class NeuralSelectionEvidence(BaseModel):
    """Calibrated neural support over the currently visible tile IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: Identifier
    mode: EvidenceMode
    candidate_probabilities: dict[Identifier, float] = Field(default_factory=dict)
    calibration_id: Identifier | None = None
    entropy: float | None = Field(default=None, ge=0.0)
    top_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    subject_id: Identifier | None = None
    session_id: Identifier | None = None
    trial_id: Identifier | None = None
    missing_reason: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_probability_distribution(self) -> NeuralSelectionEvidence:
        probabilities = self.candidate_probabilities.values()
        if not self.candidate_probabilities:
            if self.missing_reason is None:
                raise ValueError("missing neural probabilities require a missing reason")
            return self
        if self.missing_reason is not None:
            raise ValueError("neural evidence cannot include probabilities and a missing reason")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("candidate probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("candidate probabilities must sum to one")
        return self


class EvidenceBreakdown(BaseModel):
    """Auditable, separately normalized inputs to a candidate score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: Identifier
    neural: float | None = Field(default=None, ge=0.0, le=1.0)
    generic_language: float = Field(ge=0.0, le=1.0)
    personal_lift: float = Field(ge=-1.0, le=1.0)
    retrieval: float = Field(ge=0.0, le=1.0)
    diversity_adjustment: float = Field(default=0.0, ge=-0.1, le=0.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    weighted_contributions: dict[Identifier, float]
    total_score: float
    dominance_flags: frozenset[Identifier] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def require_auditable_total(self) -> EvidenceBreakdown:
        if self.risk not in {0.0, 0.5, 1.0}:
            raise ValueError("risk must be one of 0.0, 0.5, or 1.0")
        expected = sum(self.weighted_contributions.values())
        if not math.isclose(expected, self.total_score, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("weighted contributions must sum to total_score")
        return self


class SessionState(StrEnum):
    DRAFT = "draft"
    GENERATING = "generating"
    CANDIDATES_READY = "candidates_ready"
    SELECTING = "selecting"
    AWAITING_SELECTION_CONFIRMATION = "awaiting_selection_confirmation"
    AWAITING_FINAL_CONFIRMATION = "awaiting_final_confirmation"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


class SelectionActionType(StrEnum):
    SELECT = "select"
    REJECT = "reject"
    REPEAT = "repeat"
    BACK = "back"
    CLEAR = "clear"
    CANCEL = "cancel"
    OTHER = "other"
    FINALIZE = "finalize"


class SelectionAction(BaseModel):
    """An explicit user action, never an inferred language-model action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: SelectionActionType
    input_mode: EvidenceMode
    candidate_id: Identifier | None = None
    evidence_id: Identifier | None = None
    explicit: Literal[True] = True
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def require_candidate_for_candidate_actions(self) -> SelectionAction:
        if (
            self.action in {SelectionActionType.SELECT, SelectionActionType.REJECT}
            and self.candidate_id is None
        ):
            raise ValueError("select and reject actions require a candidate ID")
        return self


class ConfirmedSpan(BaseModel):
    """Text added through an explicit selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    action_id: Identifier


class MessageSession(BaseModel):
    """Serializable session state without mutable model objects."""

    model_config = ConfigDict(extra="forbid")

    session_id: Identifier
    profile_id: Identifier
    state: SessionState = SessionState.DRAFT
    confirmed_spans: list[ConfirmedSpan] = Field(default_factory=list)
    active_candidate_set_id: Identifier | None = None
    provisional_candidate_id: Identifier | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def confirmed_text(self) -> str:
        return " ".join(span.text for span in self.confirmed_spans)


class FinalizationRequest(BaseModel):
    """One-time confirmation bound to the exact finalized text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: Identifier
    text_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    confirmation_nonce: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=16, max_length=256)
    ]
    explicit_confirmation: Literal[True]
    high_risk_acknowledged: bool = False
