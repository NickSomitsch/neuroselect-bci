from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from neuroselect.bci import SeededNeuralSimulator, SimulationConfig
from neuroselect.core.models import FinalizationRequest, SelectionActionType, SessionState
from neuroselect.language import (
    CandidateGenerator,
    FixtureCandidateBackend,
    load_fixture_backend_config,
)
from neuroselect.orchestration import (
    CreateSessionRequest,
    ManualTextRequest,
    RoundRequest,
    SelectionConfirmationRequest,
    SessionActionRequest,
    SessionConflictError,
    SessionInputMode,
    SessionNotFoundError,
    SessionOrchestrator,
    SessionValidationError,
)
from neuroselect.ranking import RankingDisposition
from neuroselect.retrieval import KnowledgeRecordInput, SQLiteKnowledgeStore
from neuroselect.synthetic import load_profiles

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
NONCE = "fixed-confirmation-nonce-12345"


class MutableClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


class RiskCandidateGenerator(CandidateGenerator):
    def __init__(self) -> None:
        config = load_fixture_backend_config()
        sensitive = config.default_candidates[0].model_copy(update={"text": "medical help"})
        configured = config.model_copy(
            update={"default_candidates": (sensitive, *config.default_candidates[1:])}
        )
        super().__init__(FixtureCandidateBackend(configured))


def make_service(
    clock: MutableClock,
    *,
    risk_candidates: bool = False,
) -> SessionOrchestrator:
    profiles = load_profiles()
    store = SQLiteKnowledgeStore(":memory:")
    for profile in profiles:
        for record in profile.knowledge:
            store.add(
                profile_id=profile.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=NOW,
            )
    identifiers = iter(f"session-test-{index}" for index in range(1, 20))
    return SessionOrchestrator(
        profiles=profiles,
        knowledge_store=store,
        candidate_generator=RiskCandidateGenerator() if risk_candidates else None,
        simulator=SeededNeuralSimulator(
            SimulationConfig(
                target_concentration=100.0,
                lapse_probability=0.0,
                ambiguous_probability=0.0,
            )
        ),
        clock=clock,
        session_id_factory=lambda: next(identifiers),
        nonce_factory=lambda: NONCE,
    )


@pytest.fixture
def service_bundle() -> Iterator[tuple[SessionOrchestrator, MutableClock]]:
    clock = MutableClock()
    service = make_service(clock)
    yield service, clock
    service.close()


def create(
    service: SessionOrchestrator,
    mode: SessionInputMode = SessionInputMode.SIMULATION,
) -> str:
    view = service.create_session(
        CreateSessionRequest(profile_id="synthetic-concise", input_mode=mode)
    )
    return view.session.session_id


def test_simulated_round_connects_all_components_without_selecting(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service)
    view = service.start_round(session_id, RoundRequest(simulated_target_index=0))

    assert view.session.state is SessionState.SELECTING
    assert view.session.confirmed_text == ""
    assert view.active_generation is not None
    assert view.ranking is not None
    assert view.ranking.disposition is RankingDisposition.DISPLAY
    assert view.ranking.requires_explicit_selection is True
    assert view.ranking.automatic_selection_permitted is False
    assert view.metrics.round_count == 1
    assert "simulation" not in view.model_dump()


def test_profiles_are_public_synthetic_summaries_and_round_size_is_configurable(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    profiles = service.list_profiles()

    assert [profile.profile_id for profile in profiles] == sorted(
        profile.profile_id for profile in profiles
    )
    assert all(profile.synthetic for profile in profiles)
    assert all(profile.display_name and profile.style_summary for profile in profiles)

    session_id = create(service)
    view = service.start_round(
        session_id,
        RoundRequest(candidate_count=6, maximum_phrase_tokens=2, simulated_target_index=2),
    )

    assert view.active_generation is not None
    assert len(view.active_generation.candidate_set.candidates) == 6
    language_candidates = view.active_generation.generic_language_support
    assert len(language_candidates) == 3
    assert all(
        len(candidate.text.split()) <= 2
        for candidate in view.active_generation.candidate_set.candidates
        if candidate.candidate_id in language_candidates
    )


def test_standard_explicit_selection_appends_only_the_selected_candidate(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service)
    round_view = service.start_round(session_id, RoundRequest())
    assert round_view.ranking is not None
    top_id = round_view.ranking.fused_top_candidate_id
    top_text = round_view.ranking.ranked_candidates[0].candidate.text

    selected = service.apply_action(
        session_id,
        SessionActionRequest(action=SelectionActionType.SELECT, candidate_id=top_id),
    )

    assert selected.session.state is SessionState.DRAFT
    assert selected.session.confirmed_text == top_text
    assert selected.active_generation is None
    assert selected.ranking is None
    assert selected.metrics.selection_count == 1


def test_non_top_selection_requires_second_confirmation_and_can_be_rejected(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service)
    round_view = service.start_round(session_id, RoundRequest())
    assert round_view.ranking is not None
    non_top = round_view.ranking.ranked_candidates[1].candidate

    provisional = service.apply_action(
        session_id,
        SessionActionRequest(
            action=SelectionActionType.SELECT,
            candidate_id=non_top.candidate_id,
        ),
    )
    assert provisional.session.state is SessionState.AWAITING_SELECTION_CONFIRMATION
    assert provisional.session.confirmed_text == ""
    assert provisional.pending_selection_candidate_id == non_top.candidate_id

    rejected = service.resolve_selection(
        session_id,
        SelectionConfirmationRequest(
            candidate_id=non_top.candidate_id,
            accept=False,
            explicit_confirmation=True,
        ),
    )
    assert rejected.session.state is SessionState.CANDIDATES_READY
    assert rejected.rejected_candidate_ids == (non_top.candidate_id,)
    assert rejected.metrics.rejection_count == 1

    with pytest.raises(SessionValidationError, match="explicitly rejected"):
        service.apply_action(
            session_id,
            SessionActionRequest(
                action=SelectionActionType.SELECT,
                candidate_id=non_top.candidate_id,
            ),
        )


def test_repeat_generates_new_neural_evidence_for_the_same_candidates(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, clock = service_bundle
    session_id = create(service)
    first = service.start_round(session_id, RoundRequest())
    assert first.ranking is not None
    first_evidence_id = first.ranking.neural_evidence_id
    first_candidate_set_id = first.session.active_candidate_set_id
    clock.advance(timedelta(seconds=1))

    repeated = service.apply_action(
        session_id,
        SessionActionRequest(action=SelectionActionType.REPEAT),
    )

    assert repeated.session.state is SessionState.SELECTING
    assert repeated.session.active_candidate_set_id == first_candidate_set_id
    assert repeated.ranking is not None
    assert repeated.ranking.neural_evidence_id != first_evidence_id
    assert repeated.metrics.round_count == 2
    assert repeated.metrics.repeat_count == 1


def test_manual_mode_can_debug_an_abstained_round_and_append_keyboard_text(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    appended = service.append_manual_text(
        session_id,
        ManualTextRequest(text="Hello", explicit_confirmation=True),
    )
    assert appended.session.confirmed_text == "Hello"
    assert appended.metrics.manual_text_count == 1

    round_view = service.start_round(session_id, RoundRequest())
    assert round_view.ranking is not None
    assert round_view.ranking.disposition is RankingDisposition.ABSTAIN
    top = round_view.ranking.ranked_candidates[0].candidate
    selected = service.apply_action(
        session_id,
        SessionActionRequest(
            action=SelectionActionType.SELECT,
            candidate_id=top.candidate_id,
        ),
    )
    assert selected.session.confirmed_text == f"Hello {top.text}"

    with pytest.raises(SessionConflictError, match="repeat requires"):
        service.start_round(session_id, RoundRequest())
        service.apply_action(
            session_id,
            SessionActionRequest(action=SelectionActionType.REPEAT),
        )


def test_back_clear_other_and_cancel_controls_are_explicit(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    service.append_manual_text(
        session_id, ManualTextRequest(text="one", explicit_confirmation=True)
    )
    backed = service.apply_action(session_id, SessionActionRequest(action=SelectionActionType.BACK))
    assert backed.session.confirmed_text == ""
    assert backed.metrics.backtrack_count == 1

    service.append_manual_text(
        session_id, ManualTextRequest(text="two", explicit_confirmation=True)
    )
    cleared = service.apply_action(
        session_id, SessionActionRequest(action=SelectionActionType.CLEAR)
    )
    assert cleared.session.confirmed_text == ""
    assert cleared.metrics.clear_count == 1

    round_view = service.start_round(session_id, RoundRequest())
    assert round_view.active_generation is not None
    other_id = next(
        candidate_id
        for candidate_id, action in round_view.active_generation.control_actions.items()
        if action.value == "other"
    )
    other = service.apply_action(
        session_id,
        SessionActionRequest(action=SelectionActionType.SELECT, candidate_id=other_id),
    )
    assert other.session.state is SessionState.DRAFT
    assert other.metrics.other_count == 1

    cancelled = service.apply_action(
        session_id, SessionActionRequest(action=SelectionActionType.CANCEL)
    )
    assert cancelled.session.state is SessionState.CANCELLED
    with pytest.raises(SessionConflictError):
        service.apply_action(session_id, SessionActionRequest(action=SelectionActionType.CANCEL))


def test_finalization_nonce_hash_expiry_rejection_and_one_time_confirmation(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, clock = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    service.append_manual_text(
        session_id,
        ManualTextRequest(text="Confirmed message", explicit_confirmation=True),
    )
    challenge = service.request_finalization(session_id)
    assert challenge.confirmation_nonce == NONCE
    assert challenge.text == "Confirmed message"

    with pytest.raises(SessionValidationError, match="nonce"):
        service.confirm_finalization(
            session_id,
            FinalizationRequest(
                session_id=session_id,
                text_sha256=challenge.text_sha256,
                confirmation_nonce="incorrect-confirmation-nonce",
                explicit_confirmation=True,
            ),
        )

    rejected = service.reject_finalization(session_id)
    assert rejected.session.state is SessionState.DRAFT
    challenge = service.request_finalization(session_id)
    clock.advance(timedelta(minutes=5))
    with pytest.raises(SessionConflictError, match="expired"):
        service.confirm_finalization(
            session_id,
            FinalizationRequest(
                session_id=session_id,
                text_sha256=challenge.text_sha256,
                confirmation_nonce=challenge.confirmation_nonce,
                explicit_confirmation=True,
            ),
        )
    service.reject_finalization(session_id)

    clock.value = NOW
    challenge = service.request_finalization(session_id)
    finalized = service.confirm_finalization(
        session_id,
        FinalizationRequest(
            session_id=session_id,
            text_sha256=challenge.text_sha256,
            confirmation_nonce=challenge.confirmation_nonce,
            explicit_confirmation=True,
        ),
    )
    assert finalized.session.state is SessionState.FINALIZED
    assert finalized.finalization_pending is False
    with pytest.raises(SessionConflictError):
        service.confirm_finalization(
            session_id,
            FinalizationRequest(
                session_id=session_id,
                text_sha256=challenge.text_sha256,
                confirmation_nonce=challenge.confirmation_nonce,
                explicit_confirmation=True,
            ),
        )


def test_unsafe_finalization_nonce_does_not_change_session_state(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    service.append_manual_text(
        session_id,
        ManualTextRequest(text="Keep this draft", explicit_confirmation=True),
    )
    before = service.get_session(session_id)
    service.nonce_factory = lambda: "too-short"

    with pytest.raises(SessionValidationError, match="unsafe nonce"):
        service.request_finalization(session_id)
    assert service.get_session(session_id) == before


def test_rejected_actions_cannot_mutate_pending_or_finalized_text(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    service.append_manual_text(
        session_id,
        ManualTextRequest(text="Exact approved text", explicit_confirmation=True),
    )
    challenge = service.request_finalization(session_id)
    pending = service.get_session(session_id)

    for action in (
        SelectionActionType.BACK,
        SelectionActionType.CLEAR,
        SelectionActionType.OTHER,
    ):
        with pytest.raises(SessionConflictError):
            service.apply_action(session_id, SessionActionRequest(action=action))
        assert service.get_session(session_id) == pending

    finalized = service.confirm_finalization(
        session_id,
        FinalizationRequest(
            session_id=session_id,
            text_sha256=challenge.text_sha256,
            confirmation_nonce=challenge.confirmation_nonce,
            explicit_confirmation=True,
        ),
    )
    for action in (
        SelectionActionType.BACK,
        SelectionActionType.CLEAR,
        SelectionActionType.CANCEL,
    ):
        with pytest.raises(SessionConflictError):
            service.apply_action(session_id, SessionActionRequest(action=action))
        assert service.get_session(session_id) == finalized


def test_finalization_recomputes_the_current_message_hash(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    session_id = create(service, SessionInputMode.MANUAL)
    service.append_manual_text(
        session_id,
        ManualTextRequest(text="Challenge-bound text", explicit_confirmation=True),
    )
    challenge = service.request_finalization(session_id)
    runtime = service._sessions[session_id]
    runtime.session.confirmed_spans[0] = runtime.session.confirmed_spans[0].model_copy(
        update={"text": "Unexpected replacement"}
    )

    with pytest.raises(SessionValidationError, match="text hash"):
        service.confirm_finalization(
            session_id,
            FinalizationRequest(
                session_id=session_id,
                text_sha256=challenge.text_sha256,
                confirmation_nonce=challenge.confirmation_nonce,
                explicit_confirmation=True,
            ),
        )
    assert service.get_session(session_id).session.state is SessionState.AWAITING_FINAL_CONFIRMATION


def test_high_risk_selection_requires_two_selection_steps_and_final_acknowledgement() -> None:
    clock = MutableClock()
    service = make_service(clock, risk_candidates=True)
    try:
        session_id = create(service)
        round_view = service.start_round(session_id, RoundRequest())
        assert round_view.ranking is not None
        risky = next(
            item for item in round_view.ranking.ranked_candidates if item.candidate.risk_tags
        )
        provisional = service.apply_action(
            session_id,
            SessionActionRequest(
                action=SelectionActionType.SELECT,
                candidate_id=risky.candidate.candidate_id,
            ),
        )
        assert provisional.session.state is SessionState.AWAITING_SELECTION_CONFIRMATION
        accepted = service.resolve_selection(
            session_id,
            SelectionConfirmationRequest(
                candidate_id=risky.candidate.candidate_id,
                accept=True,
                explicit_confirmation=True,
            ),
        )
        assert accepted.high_risk_acknowledgement_required is True
        challenge = service.request_finalization(session_id)
        assert challenge.high_risk_acknowledgement_required is True

        with pytest.raises(SessionValidationError, match="acknowledgement"):
            service.confirm_finalization(
                session_id,
                FinalizationRequest(
                    session_id=session_id,
                    text_sha256=challenge.text_sha256,
                    confirmation_nonce=challenge.confirmation_nonce,
                    explicit_confirmation=True,
                ),
            )
        finalized = service.confirm_finalization(
            session_id,
            FinalizationRequest(
                session_id=session_id,
                text_sha256=challenge.text_sha256,
                confirmation_nonce=challenge.confirmation_nonce,
                explicit_confirmation=True,
                high_risk_acknowledged=True,
            ),
        )
        assert finalized.session.state is SessionState.FINALIZED
    finally:
        service.close()


def test_service_rejects_unknown_resources_and_invalid_round_commands(
    service_bundle: tuple[SessionOrchestrator, MutableClock],
) -> None:
    service, _ = service_bundle
    with pytest.raises(SessionValidationError, match="unknown profile"):
        service.create_session(CreateSessionRequest(profile_id="unknown-profile"))
    with pytest.raises(SessionNotFoundError):
        service.get_session("missing-session")

    session_id = create(service)
    with pytest.raises(SessionValidationError, match="exceeds"):
        service.start_round(session_id, RoundRequest(simulated_target_index=11))
    assert service.get_session(session_id).session.state is SessionState.DRAFT
    service.start_round(session_id, RoundRequest())
    with pytest.raises(SessionConflictError):
        service.start_round(session_id, RoundRequest())
    with pytest.raises(SessionValidationError, match="not visible"):
        service.apply_action(
            session_id,
            SessionActionRequest(
                action=SelectionActionType.SELECT,
                candidate_id="not-visible",
            ),
        )
