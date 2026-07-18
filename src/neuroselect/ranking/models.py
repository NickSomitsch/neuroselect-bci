"""Typed inputs, policy, and auditable outputs for candidate fusion."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import (
    Candidate,
    CandidateKind,
    CandidateSet,
    EvidenceBreakdown,
    Identifier,
    NeuralSelectionEvidence,
)
from neuroselect.retrieval.models import CandidateRetrievalEvidence, RetrievalHit

UnitSupport = Annotated[float, Field(ge=0.0, le=1.0)]
PersonalLift = Annotated[float, Field(ge=-1.0, le=1.0)]


class RankingDisposition(StrEnum):
    DISPLAY = "display"
    REQUEST_REPEAT = "request_repeat"
    ABSTAIN = "abstain"


class ConfirmationLevel(StrEnum):
    STANDARD = "standard"
    ENHANCED = "enhanced"


class RankingReason(StrEnum):
    MISSING_NEURAL_EVIDENCE = "missing_neural_evidence"
    LOW_NEURAL_SUPPORT = "low_neural_support"
    LOW_NEURAL_MARGIN = "low_neural_margin"
    NEURAL_LANGUAGE_CONFLICT = "neural_language_conflict"
    LM_DOMINANCE_DETECTED = "lm_dominance_detected"
    LOW_FUSED_SCORE = "low_fused_score"
    LOW_FUSED_MARGIN = "low_fused_margin"
    SENSITIVE_CANDIDATE = "sensitive_candidate"


class DominanceFlag(StrEnum):
    LM_OVER_NEURAL = "lm-over-neural"
    MISSING_NEURAL = "missing-neural"


class RiskLevel(StrEnum):
    NONE = "none"
    ELEVATED = "elevated"
    HIGH = "high"

    @property
    def numeric(self) -> float:
        return {
            RiskLevel.NONE: 0.0,
            RiskLevel.ELEVATED: 0.5,
            RiskLevel.HIGH: 1.0,
        }[self]


class FusionWeights(BaseModel):
    """Versioned transparent weights; signal weights form a convex combination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    neural: float = Field(default=0.65, ge=0.5, le=1.0)
    generic_language: float = Field(default=0.15, ge=0.0, le=0.5)
    personalization: float = Field(default=0.08, ge=0.0, le=0.5)
    retrieval: float = Field(default=0.12, ge=0.0, le=0.5)
    risk_penalty: float = Field(default=0.35, ge=0.0, le=1.0)
    maximum_diversity_penalty: float = Field(default=0.08, ge=0.0, le=0.1)

    @model_validator(mode="after")
    def enforce_neural_floor_and_convex_signals(self) -> FusionWeights:
        signal_total = self.neural + self.generic_language + self.personalization + self.retrieval
        if not math.isclose(signal_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "neural, language, personalization, and retrieval weights must sum to one"
            )
        if self.generic_language + self.personalization + self.retrieval > 0.35 + 1e-9:
            raise ValueError("combined non-neural signal weight cannot exceed 0.35")
        return self


class RankingPolicy(BaseModel):
    """Safety thresholds and risk taxonomy for one ranking revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    policy_revision: Identifier
    weights: FusionWeights = Field(default_factory=FusionWeights)
    minimum_neural_top_support: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_neural_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_fused_top_score: float = Field(default=0.25, ge=-1.0, le=1.0)
    minimum_fused_margin: float = Field(default=0.03, ge=0.0, le=1.0)
    lm_dominance_ratio: float = Field(default=1.0, ge=1.0, le=10.0)
    elevated_risk_tags: frozenset[Identifier]
    high_risk_tags: frozenset[Identifier]

    @model_validator(mode="after")
    def require_disjoint_risk_tags(self) -> RankingPolicy:
        if self.elevated_risk_tags.intersection(self.high_risk_tags):
            raise ValueError("elevated and high risk tags must be disjoint")
        return self


class RankingInputs(BaseModel):
    """Separately normalized evidence supplied to the transparent ranker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_set: CandidateSet
    neural_evidence: NeuralSelectionEvidence
    generic_language_support: dict[Identifier, UnitSupport]
    personalization_lift: dict[Identifier, PersonalLift] = Field(default_factory=dict)
    retrieval_evidence: tuple[CandidateRetrievalEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_candidate_alignment(self) -> RankingInputs:
        all_ids = {candidate.candidate_id for candidate in self.candidate_set.candidates}
        language_ids = {
            candidate.candidate_id
            for candidate in self.candidate_set.candidates
            if candidate.kind is not CandidateKind.CONTROL
        }
        if set(self.generic_language_support) != language_ids:
            raise ValueError("generic language support must cover exactly the language candidates")
        if not math.isclose(
            sum(self.generic_language_support.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("generic language support must sum to one")
        neural_ids = set(self.neural_evidence.candidate_probabilities)
        if neural_ids and neural_ids != all_ids:
            raise ValueError("neural evidence must cover exactly the visible candidates")
        if not set(self.personalization_lift).issubset(language_ids):
            raise ValueError("personalization lift can reference only language candidates")
        retrieval_ids = [item.candidate_id for item in self.retrieval_evidence]
        if len(retrieval_ids) != len(set(retrieval_ids)):
            raise ValueError("retrieval evidence candidate IDs must be unique")
        if not set(retrieval_ids).issubset(language_ids):
            raise ValueError("retrieval evidence can reference only language candidates")
        return self


class RankedCandidate(BaseModel):
    """One candidate with every normalized and weighted contribution visible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    candidate: Candidate
    breakdown: EvidenceBreakdown
    risk_level: RiskLevel
    confirmation_level: ConfirmationLevel
    retrieval_hits: tuple[RetrievalHit, ...] = ()


class RankingResult(BaseModel):
    """Ranking plus a safety disposition; never an inferred selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_set_id: Identifier
    policy_revision: Identifier
    neural_evidence_id: Identifier
    disposition: RankingDisposition
    confirmation_level: ConfirmationLevel
    reason_codes: tuple[RankingReason, ...]
    ranked_candidates: tuple[RankedCandidate, ...] = Field(min_length=1)
    fused_top_candidate_id: Identifier
    neural_top_candidate_id: Identifier | None
    display_top_candidate_id: Identifier | None
    fused_margin: float = Field(ge=0.0)
    neural_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    requires_explicit_selection: Literal[True] = True
    automatic_selection_permitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_ranking_summary(self) -> RankingResult:
        ranks = tuple(item.rank for item in self.ranked_candidates)
        if ranks != tuple(range(1, len(self.ranked_candidates) + 1)):
            raise ValueError("ranked candidates must use contiguous one-based ranks")
        candidate_ids = tuple(item.candidate.candidate_id for item in self.ranked_candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("ranked candidate IDs must be unique")
        if self.fused_top_candidate_id != candidate_ids[0]:
            raise ValueError("fused top candidate must be the first ranked candidate")
        expected_display = (
            self.fused_top_candidate_id if self.disposition is RankingDisposition.DISPLAY else None
        )
        if self.display_top_candidate_id != expected_display:
            raise ValueError("only a display disposition may expose a display top candidate")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("ranking reason codes must be unique")
        return self
