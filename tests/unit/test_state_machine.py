import pytest

from neuroselect.core.models import SessionState
from neuroselect.core.state_machine import InvalidTransitionError, SessionEvent, transition


def test_high_confidence_selection_returns_to_draft() -> None:
    state = transition(SessionState.DRAFT, SessionEvent.REQUEST_CANDIDATES)
    state = transition(state, SessionEvent.CANDIDATES_GENERATED)
    state = transition(state, SessionEvent.START_SELECTION)
    state = transition(state, SessionEvent.ACCEPT_SELECTION)

    assert state is SessionState.DRAFT


def test_uncertain_selection_requires_confirmation() -> None:
    state = transition(SessionState.SELECTING, SessionEvent.REQUIRE_SELECTION_CONFIRMATION)
    assert state is SessionState.AWAITING_SELECTION_CONFIRMATION
    assert transition(state, SessionEvent.CONFIRM_SELECTION) is SessionState.DRAFT


def test_rejected_selection_preserves_candidate_set() -> None:
    state = transition(
        SessionState.AWAITING_SELECTION_CONFIRMATION,
        SessionEvent.REJECT_SELECTION,
    )
    assert state is SessionState.CANDIDATES_READY

    assert (
        transition(SessionState.SELECTING, SessionEvent.REJECT_SELECTION)
        is SessionState.CANDIDATES_READY
    )


@pytest.mark.parametrize(
    "state",
    [
        SessionState.DRAFT,
        SessionState.GENERATING,
        SessionState.CANDIDATES_READY,
        SessionState.SELECTING,
        SessionState.AWAITING_SELECTION_CONFIRMATION,
        SessionState.AWAITING_FINAL_CONFIRMATION,
    ],
)
def test_nonterminal_session_can_be_cancelled(state: SessionState) -> None:
    assert transition(state, SessionEvent.CANCEL_SESSION) is SessionState.CANCELLED


def test_finalization_cannot_skip_confirmation() -> None:
    with pytest.raises(InvalidTransitionError, match="confirm_finalization"):
        transition(SessionState.DRAFT, SessionEvent.CONFIRM_FINALIZATION)

    state = transition(SessionState.DRAFT, SessionEvent.REQUEST_FINALIZATION)
    assert state is SessionState.AWAITING_FINAL_CONFIRMATION
    assert transition(state, SessionEvent.CONFIRM_FINALIZATION) is SessionState.FINALIZED


def test_terminal_states_are_immutable() -> None:
    with pytest.raises(InvalidTransitionError):
        transition(SessionState.FINALIZED, SessionEvent.CANCEL_SESSION)
    with pytest.raises(InvalidTransitionError):
        transition(SessionState.CANCELLED, SessionEvent.REQUEST_CANDIDATES)


def test_repeat_returns_to_selection() -> None:
    assert (
        transition(
            SessionState.AWAITING_SELECTION_CONFIRMATION,
            SessionEvent.REPEAT_SELECTION,
        )
        is SessionState.SELECTING
    )


def test_finalization_can_be_rejected() -> None:
    assert (
        transition(
            SessionState.AWAITING_FINAL_CONFIRMATION,
            SessionEvent.REJECT_FINALIZATION,
        )
        is SessionState.DRAFT
    )


@pytest.mark.parametrize(
    "state",
    [
        SessionState.DRAFT,
        SessionState.CANDIDATES_READY,
        SessionState.SELECTING,
        SessionState.AWAITING_SELECTION_CONFIRMATION,
    ],
)
def test_explicit_back_clear_or_other_path_can_return_to_draft(state: SessionState) -> None:
    assert transition(state, SessionEvent.RETURN_TO_DRAFT) is SessionState.DRAFT
