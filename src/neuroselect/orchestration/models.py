"""API-safe session commands, views, metrics, and confirmation challenges."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from neuroselect.core.models import Identifier, MessageSession, SelectionActionType
from neuroselect.language.models import CandidateGenerationResult
from neuroselect.ranking.models import RankingResult


class SessionInputMode(StrEnum):
    MANUAL = "manual"
    SIMULATION = "simulation"


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: Identifier
    input_mode: SessionInputMode = SessionInputMode.SIMULATION


class ProfileSummary(BaseModel):
    """Public, synthetic profile metadata safe to expose in the local UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: Identifier
    display_name: str = Field(min_length=1, max_length=120)
    style_summary: str = Field(min_length=1, max_length=500)
    synthetic: Literal[True]


class RoundRequest(BaseModel):
    """Simulation ground truth is an explicit demo input, never an inferred thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulated_target_index: int = Field(default=0, ge=0, le=11)
    candidate_count: Literal[4, 6, 8, 12] | None = None
    maximum_phrase_tokens: int | None = Field(default=None, ge=1, le=8)


class SessionActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: SelectionActionType
    candidate_id: Identifier | None = None
    explicit: Literal[True] = True

    @model_validator(mode="after")
    def validate_action_shape(self) -> SessionActionRequest:
        allowed = {
            SelectionActionType.SELECT,
            SelectionActionType.REJECT,
            SelectionActionType.REPEAT,
            SelectionActionType.BACK,
            SelectionActionType.CLEAR,
            SelectionActionType.CANCEL,
            SelectionActionType.OTHER,
        }
        if self.action not in allowed:
            raise ValueError("finalization uses the dedicated confirmation endpoints")
        if self.action in {SelectionActionType.SELECT, SelectionActionType.REJECT}:
            if self.candidate_id is None:
                raise ValueError("select and reject actions require a candidate ID")
        elif self.candidate_id is not None:
            raise ValueError("candidate ID is valid only for select and reject actions")
        return self


class SelectionConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: Identifier
    accept: bool
    explicit_confirmation: Literal[True]


class ManualTextRequest(BaseModel):
    """Explicit keyboard text for debug mode; it bypasses language inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    explicit_confirmation: Literal[True]


class SessionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_count: int = Field(ge=0)
    selection_count: int = Field(ge=0)
    rejection_count: int = Field(ge=0)
    repeat_count: int = Field(ge=0)
    backtrack_count: int = Field(ge=0)
    clear_count: int = Field(ge=0)
    other_count: int = Field(ge=0)
    manual_text_count: int = Field(ge=0)


class FinalizationChallenge(BaseModel):
    """One-time nonce bound to the exact text hash awaiting explicit confirmation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: Identifier
    text: str = Field(min_length=1, max_length=4_000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_nonce: str = Field(min_length=16, max_length=256)
    high_risk_acknowledgement_required: bool
    expires_at: datetime


class SessionView(BaseModel):
    """Serializable state without simulator ground truth or confirmation secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: MessageSession
    input_mode: SessionInputMode
    active_generation: CandidateGenerationResult | None = None
    ranking: RankingResult | None = None
    rejected_candidate_ids: tuple[Identifier, ...] = ()
    pending_selection_candidate_id: Identifier | None = None
    finalization_pending: bool = False
    high_risk_acknowledgement_required: bool = False
    metrics: SessionMetrics
