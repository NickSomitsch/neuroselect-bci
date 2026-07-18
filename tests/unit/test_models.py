import math
from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from neuroselect.core.models import (
    Candidate,
    CandidateKind,
    CandidateSet,
    ConfirmedSpan,
    EvidenceBreakdown,
    EvidenceMode,
    FinalizationRequest,
    MessageSession,
    NeuralSelectionEvidence,
    SelectionAction,
    SelectionActionType,
)

SHA256 = "0" * 64


def candidate(candidate_id: str = "candidate-1") -> Candidate:
    return Candidate(candidate_id=candidate_id, text="hello", kind=CandidateKind.WORD)


def test_candidate_set_requires_unique_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        CandidateSet(
            candidate_set_id="set-1",
            context_sha256=SHA256,
            candidates=(candidate(), candidate()),
            generator_revision="fixture-v1",
            prompt_revision="prompt-v1",
        )


def test_candidate_set_preserves_fixed_order() -> None:
    candidate_set = CandidateSet(
        candidate_set_id="set-1",
        context_sha256=SHA256,
        candidates=(candidate("first"), candidate("second")),
        generator_revision="fixture-v1",
        prompt_revision="prompt-v1",
    )

    assert [item.candidate_id for item in candidate_set.candidates] == ["first", "second"]


def test_neural_evidence_accepts_probability_distribution() -> None:
    evidence = NeuralSelectionEvidence(
        evidence_id="evidence-1",
        mode=EvidenceMode.SIMULATION,
        candidate_probabilities={"first": 0.75, "second": 0.25},
        entropy=0.56,
        top_margin=0.5,
    )

    assert sum(evidence.candidate_probabilities.values()) == pytest.approx(1.0)


def test_missing_neural_evidence_is_explicit() -> None:
    evidence = NeuralSelectionEvidence(
        evidence_id="missing-1",
        mode=EvidenceMode.MANUAL,
        missing_reason="manual input has no neural evidence",
    )

    assert evidence.candidate_probabilities == {}


@given(
    first=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    second=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_neural_evidence_rejects_non_normalized_probabilities(first: float, second: float) -> None:
    if math.isclose(first + second, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        return
    with pytest.raises(ValidationError, match="sum to one"):
        NeuralSelectionEvidence(
            evidence_id="invalid-1",
            mode=EvidenceMode.SIMULATION,
            candidate_probabilities={"first": first, "second": second},
        )


def test_neural_evidence_rejects_conflicting_missing_reason() -> None:
    with pytest.raises(ValidationError, match="cannot include"):
        NeuralSelectionEvidence(
            evidence_id="invalid-2",
            mode=EvidenceMode.REPLAY,
            candidate_probabilities={"first": 1.0},
            missing_reason="not actually missing",
        )


def test_evidence_breakdown_total_is_auditable() -> None:
    breakdown = EvidenceBreakdown(
        candidate_id="candidate-1",
        neural=0.8,
        generic_language=0.4,
        personal_lift=0.1,
        retrieval=0.0,
        weighted_contributions={"neural": 0.52, "generic": 0.14},
        total_score=0.66,
    )
    assert breakdown.total_score == 0.66

    with pytest.raises(ValidationError, match="sum to total_score"):
        breakdown.model_copy(update={"total_score": 0.9}).__class__.model_validate(
            {**breakdown.model_dump(), "total_score": 0.9}
        )

    with pytest.raises(ValidationError, match="risk must be one of"):
        EvidenceBreakdown.model_validate({**breakdown.model_dump(), "risk": 0.25})


def test_selection_actions_require_explicit_candidate() -> None:
    with pytest.raises(ValidationError, match="require a candidate"):
        SelectionAction(action=SelectionActionType.SELECT, input_mode=EvidenceMode.MANUAL)

    action = SelectionAction(
        action=SelectionActionType.SELECT,
        input_mode=EvidenceMode.REPLAY,
        candidate_id="candidate-1",
        evidence_id="evidence-1",
        occurred_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    assert action.explicit is True


def test_session_exposes_only_confirmed_text() -> None:
    session = MessageSession(
        session_id="session-1",
        profile_id="synthetic-concise",
        confirmed_spans=[
            ConfirmedSpan(text="hello", action_id="action-1"),
            ConfirmedSpan(text="there", action_id="action-2"),
        ],
        provisional_candidate_id="candidate-3",
    )

    assert session.confirmed_text == "hello there"


def test_finalization_is_bound_to_explicit_confirmation() -> None:
    request = FinalizationRequest(
        session_id="session-1",
        text_sha256=SHA256,
        confirmation_nonce="nonce-that-is-long-enough",
        explicit_confirmation=True,
    )
    assert request.explicit_confirmation is True

    with pytest.raises(ValidationError):
        FinalizationRequest.model_validate(
            {
                "session_id": "session-1",
                "text_sha256": SHA256,
                "confirmation_nonce": "nonce-that-is-long-enough",
                "explicit_confirmation": False,
            }
        )
