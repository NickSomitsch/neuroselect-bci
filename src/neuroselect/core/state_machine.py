"""Pure session-state transitions used by every future input adapter."""

from __future__ import annotations

from enum import StrEnum

from neuroselect.core.models import SessionState


class SessionEvent(StrEnum):
    REQUEST_CANDIDATES = "request_candidates"
    CANDIDATES_GENERATED = "candidates_generated"
    START_SELECTION = "start_selection"
    ACCEPT_SELECTION = "accept_selection"
    REQUIRE_SELECTION_CONFIRMATION = "require_selection_confirmation"
    CONFIRM_SELECTION = "confirm_selection"
    REJECT_SELECTION = "reject_selection"
    REPEAT_SELECTION = "repeat_selection"
    REQUEST_FINALIZATION = "request_finalization"
    CONFIRM_FINALIZATION = "confirm_finalization"
    REJECT_FINALIZATION = "reject_finalization"
    CANCEL_SESSION = "cancel_session"


class InvalidTransitionError(ValueError):
    """Raised when an event is not valid for the current session state."""


_TRANSITIONS: dict[tuple[SessionState, SessionEvent], SessionState] = {
    (SessionState.DRAFT, SessionEvent.REQUEST_CANDIDATES): SessionState.GENERATING,
    (SessionState.GENERATING, SessionEvent.CANDIDATES_GENERATED): SessionState.CANDIDATES_READY,
    (SessionState.CANDIDATES_READY, SessionEvent.START_SELECTION): SessionState.SELECTING,
    (SessionState.SELECTING, SessionEvent.ACCEPT_SELECTION): SessionState.DRAFT,
    (
        SessionState.SELECTING,
        SessionEvent.REQUIRE_SELECTION_CONFIRMATION,
    ): SessionState.AWAITING_SELECTION_CONFIRMATION,
    (
        SessionState.AWAITING_SELECTION_CONFIRMATION,
        SessionEvent.CONFIRM_SELECTION,
    ): SessionState.DRAFT,
    (
        SessionState.AWAITING_SELECTION_CONFIRMATION,
        SessionEvent.REJECT_SELECTION,
    ): SessionState.CANDIDATES_READY,
    (SessionState.SELECTING, SessionEvent.REPEAT_SELECTION): SessionState.SELECTING,
    (
        SessionState.AWAITING_SELECTION_CONFIRMATION,
        SessionEvent.REPEAT_SELECTION,
    ): SessionState.SELECTING,
    (
        SessionState.DRAFT,
        SessionEvent.REQUEST_FINALIZATION,
    ): SessionState.AWAITING_FINAL_CONFIRMATION,
    (
        SessionState.AWAITING_FINAL_CONFIRMATION,
        SessionEvent.CONFIRM_FINALIZATION,
    ): SessionState.FINALIZED,
    (
        SessionState.AWAITING_FINAL_CONFIRMATION,
        SessionEvent.REJECT_FINALIZATION,
    ): SessionState.DRAFT,
}


def transition(state: SessionState, event: SessionEvent) -> SessionState:
    """Return the next state or reject an unsafe/undefined transition."""

    if event is SessionEvent.CANCEL_SESSION and state not in {
        SessionState.FINALIZED,
        SessionState.CANCELLED,
    }:
        return SessionState.CANCELLED
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as error:
        raise InvalidTransitionError(
            f"cannot apply {event.value!r} while session is {state.value!r}"
        ) from error
