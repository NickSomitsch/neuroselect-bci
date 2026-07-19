"""Typed contracts for language proposals and visible candidate sets."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from neuroselect.core.models import CandidateKind, CandidateSet

ShortIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]
CandidateText = Annotated[str, StringConstraints(max_length=500)]
LanguageSupport = Annotated[float, Field(ge=0.0, le=1.0)]


class ControlPath(StrEnum):
    """Application-owned paths that are never proposed by a language model."""

    OTHER = "other"
    BACK = "back"
    CANCEL = "cancel"


class ProposalRejectionReason(StrEnum):
    EMPTY = "empty"
    UNSAFE_TEXT = "unsafe_text"
    RESERVED_CONTROL = "reserved_control"
    TOO_LONG = "too_long"
    DUPLICATE = "duplicate"


class CandidateGenerationRequest(BaseModel):
    """Confirmed context and display limits; generation cannot mutate this object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmed_text: Annotated[str, StringConstraints(strip_whitespace=True, max_length=4_000)] = ""
    candidate_count: Literal[4, 6, 8, 12] = 8
    maximum_phrase_tokens: int = Field(default=4, ge=1, le=8)


class CandidateProposal(BaseModel):
    """One machine-readable language-backend proposal before policy filtering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: CandidateText
    support: float = Field(ge=0.0, allow_inf_nan=False)


class StructuredCandidateResponse(BaseModel):
    """Strict JSON shape expected from future local model adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[CandidateProposal, ...] = Field(min_length=1)


class BackendMetadata(BaseModel):
    """Immutable model, generator, and prompt provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend_id: ShortIdentifier
    model_id: ShortIdentifier
    model_revision: ShortIdentifier
    generator_revision: ShortIdentifier
    prompt_revision: ShortIdentifier
    deterministic: bool


class CandidateRiskRule(BaseModel):
    """Trusted application-owned patterns for one confirmation-policy tag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_tag: ShortIdentifier
    patterns: tuple[str, ...] = Field(min_length=1)

    @field_validator("patterns")
    @classmethod
    def require_valid_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            if not pattern:
                raise ValueError("candidate risk patterns cannot be empty")
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as error:
                raise ValueError(f"invalid candidate risk pattern: {pattern}") from error
        return patterns


class CandidateRiskPolicy(BaseModel):
    """Versioned conservative policy applied after untrusted model generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    policy_revision: ShortIdentifier
    rules: tuple[CandidateRiskRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_tags(self) -> CandidateRiskPolicy:
        risk_tags = [rule.risk_tag for rule in self.rules]
        if len(risk_tags) != len(set(risk_tags)):
            raise ValueError("candidate risk policy tags must be unique")
        return self


class FixtureRule(BaseModel):
    """Suffix-triggered proposals for deterministic development scenarios."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suffix: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    candidates: tuple[CandidateProposal, ...] = Field(min_length=1)


class FixtureBackendConfig(BaseModel):
    """Versioned, fully local fixture backend configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    backend_id: ShortIdentifier
    model_id: ShortIdentifier
    model_revision: ShortIdentifier
    generator_revision: ShortIdentifier
    prompt_revision: ShortIdentifier
    deterministic: Literal[True]
    default_candidates: tuple[CandidateProposal, ...] = Field(min_length=9)
    rules: tuple[FixtureRule, ...] = ()

    @model_validator(mode="after")
    def require_unique_rule_suffixes(self) -> FixtureBackendConfig:
        suffixes = [" ".join(rule.suffix.casefold().split()) for rule in self.rules]
        if len(suffixes) != len(set(suffixes)):
            raise ValueError("fixture rule suffixes must be unique")
        return self

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_id=self.backend_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            generator_revision=self.generator_revision,
            prompt_revision=self.prompt_revision,
            deterministic=self.deterministic,
        )


class GenerationDiagnostics(BaseModel):
    """Aggregate filtering diagnostics without retaining untrusted raw text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_proposal_count: int = Field(ge=0)
    selected_language_count: int = Field(ge=1)
    unused_valid_count: int = Field(ge=0)
    rejected_by_reason: dict[ProposalRejectionReason, int]


class CandidateGenerationResult(BaseModel):
    """Visible candidates with language support and explicit control semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_set: CandidateSet
    generic_language_support: dict[str, LanguageSupport]
    control_actions: dict[str, ControlPath]
    backend: BackendMetadata
    risk_policy_revision: ShortIdentifier
    diagnostics: GenerationDiagnostics

    @model_validator(mode="after")
    def validate_evidence_domains(self) -> CandidateGenerationResult:
        language_ids = {
            candidate.candidate_id
            for candidate in self.candidate_set.candidates
            if candidate.kind is not CandidateKind.CONTROL
        }
        control_ids = {
            candidate.candidate_id
            for candidate in self.candidate_set.candidates
            if candidate.kind is CandidateKind.CONTROL
        }
        if set(self.generic_language_support) != language_ids:
            raise ValueError("language support must cover exactly the language candidates")
        if not math.isclose(
            sum(self.generic_language_support.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("generic language support must sum to one")
        if set(self.control_actions) != control_ids:
            raise ValueError("control actions must cover exactly the control candidates")
        if set(self.control_actions.values()) != set(ControlPath):
            raise ValueError("candidate set must expose other, back, and cancel exactly once")
        return self
