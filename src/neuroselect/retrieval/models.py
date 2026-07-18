"""Typed records, policies, and explanations for personal retrieval."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from neuroselect.core.models import Identifier, KnowledgeKind, RecordPermission

RecordContent = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
]
SourceReference = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=512,
        pattern=r"^[a-z][a-z0-9+.-]*:\S+$",
    ),
]


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


class InjectionRisk(StrEnum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MARKER = "role_marker"
    PROMPT_EXFILTRATION = "prompt_exfiltration"


class KnowledgeRecordInput(BaseModel):
    """User-approved fact plus explicit provenance, permissions, and validity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: Identifier
    kind: KnowledgeKind
    content: RecordContent
    source: SourceReference
    permissions: frozenset[RecordPermission] = Field(min_length=1)
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_validity(self) -> KnowledgeRecordInput:
        if self.valid_from is not None:
            require_timezone(self.valid_from, "valid_from")
        if self.valid_until is not None:
            require_timezone(self.valid_until, "valid_until")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("valid_until must be later than valid_from")
        if self.kind is KnowledgeKind.CURRENT_EVENT and self.valid_until is None:
            raise ValueError("current-event records require valid_until")
        return self


class KnowledgeRecordPatch(BaseModel):
    """Explicit editable fields for optimistic record updates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: KnowledgeKind | None = None
    content: RecordContent | None = None
    source: SourceReference | None = None
    permissions: frozenset[RecordPermission] | None = Field(default=None, min_length=1)
    enabled: bool | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_empty_patch(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value:
            raise ValueError("knowledge record patch cannot be empty")
        return value


class StoredKnowledgeRecord(KnowledgeRecordInput):
    """Persisted record metadata and derived injection quarantine status."""

    profile_id: Identifier
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    injection_risk: bool
    risk_reasons: tuple[InjectionRisk, ...] = ()

    @model_validator(mode="after")
    def validate_storage_metadata(self) -> StoredKnowledgeRecord:
        require_timezone(self.created_at, "created_at")
        require_timezone(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.injection_risk != bool(self.risk_reasons):
            raise ValueError("injection risk must agree with its reasons")
        return self

    def is_active_at(self, at_time: datetime) -> bool:
        require_timezone(at_time, "at_time")
        if not self.enabled:
            return False
        if self.valid_from is not None and at_time < self.valid_from:
            return False
        return self.valid_until is None or at_time < self.valid_until


class RetrievalPolicy(BaseModel):
    """Versioned lexical retrieval and injection-detector policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    tokenizer_revision: Identifier
    injection_detector_revision: Identifier
    minimum_score: float = Field(default=0.08, ge=0.0, le=1.0)
    default_top_k: int = Field(default=3, ge=1, le=20)
    maximum_top_k: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_hit_limits(self) -> RetrievalPolicy:
        if self.default_top_k > self.maximum_top_k:
            raise ValueError("default_top_k cannot exceed maximum_top_k")
        return self


class RetrievalRequest(BaseModel):
    """A profile-scoped, permission-scoped retrieval request at a fixed time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: Identifier
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000)]
    permission: RecordPermission = RecordPermission.SUGGEST
    at_time: datetime
    top_k: int | None = Field(default=None, ge=1, le=20)
    kinds: frozenset[KnowledgeKind] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_aware_time(self) -> RetrievalRequest:
        require_timezone(self.at_time, "at_time")
        return self


class RetrievalHit(BaseModel):
    """One auditable retrieval result; content remains explicitly untrusted data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: StoredKnowledgeRecord
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1, max_length=1_000)


class CandidateRetrievalEvidence(BaseModel):
    """Visible retrieval influence for one language candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: Identifier
    retrieval_support: float = Field(ge=0.0, le=1.0)
    record_ids: tuple[Identifier, ...]
    hits: tuple[RetrievalHit, ...]

    @model_validator(mode="after")
    def require_consistent_hit_summary(self) -> CandidateRetrievalEvidence:
        expected_ids = tuple(hit.record.record_id for hit in self.hits)
        if self.record_ids != expected_ids:
            raise ValueError("record IDs must preserve retrieval-hit order")
        expected_support = self.hits[0].score if self.hits else 0.0
        if abs(self.retrieval_support - expected_support) > 1e-12:
            raise ValueError("retrieval support must equal the top hit score")
        return self
