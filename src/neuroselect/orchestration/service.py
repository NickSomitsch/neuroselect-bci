"""In-memory vertical-slice orchestration with explicit confirmation boundaries."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from neuroselect.bci import SeededNeuralSimulator, SimulatedRound
from neuroselect.core.config import SessionPolicyConfig
from neuroselect.core.models import (
    Candidate,
    ConfirmedSpan,
    EvidenceMode,
    FinalizationRequest,
    MessageSession,
    NeuralSelectionEvidence,
    SelectionAction,
    SelectionActionType,
    SessionState,
)
from neuroselect.core.state_machine import (
    InvalidTransitionError,
    SessionEvent,
    transition,
)
from neuroselect.language import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateGenerator,
    ControlPath,
    FixtureCandidateBackend,
)
from neuroselect.orchestration.models import (
    CreateSessionRequest,
    FinalizationChallenge,
    ManualTextRequest,
    RoundRequest,
    SelectionConfirmationRequest,
    SessionActionRequest,
    SessionInputMode,
    SessionMetrics,
    SessionView,
)
from neuroselect.ranking import (
    ConfirmationLevel,
    RankingDisposition,
    RankingInputs,
    RankingResult,
    RiskLevel,
    TransparentRanker,
)
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import SyntheticProfile, load_profiles


class SessionServiceError(RuntimeError):
    """Base class for expected session-service failures."""


class SessionNotFoundError(SessionServiceError):
    pass


class SessionConflictError(SessionServiceError):
    pass


class SessionValidationError(SessionServiceError):
    pass


@dataclass
class _Counters:
    round_count: int = 0
    selection_count: int = 0
    rejection_count: int = 0
    repeat_count: int = 0
    backtrack_count: int = 0
    clear_count: int = 0
    other_count: int = 0
    manual_text_count: int = 0

    def snapshot(self) -> SessionMetrics:
        return SessionMetrics(**vars(self))


@dataclass
class _SessionRuntime:
    session: MessageSession
    input_mode: SessionInputMode
    generation: CandidateGenerationResult | None = None
    ranking: RankingResult | None = None
    simulation: SimulatedRound | None = None
    simulated_target_id: str | None = None
    rejected_candidate_ids: set[str] = field(default_factory=set)
    provisional_candidate_id: str | None = None
    provisional_action_id: str | None = None
    high_risk_action_ids: set[str] = field(default_factory=set)
    finalization_nonce: str | None = None
    finalization_sha256: str | None = None
    finalization_expires_at: datetime | None = None
    counters: _Counters = field(default_factory=_Counters)
    actions: list[SelectionAction] = field(default_factory=list)
    action_sequence: int = 0


class SessionOrchestrator:
    """Connect generation, retrieval, neural input, ranking, and confirmation."""

    def __init__(
        self,
        *,
        profiles: tuple[SyntheticProfile, ...],
        knowledge_store: SQLiteKnowledgeStore,
        candidate_generator: CandidateGenerator | None = None,
        retriever: LexicalRetriever | None = None,
        simulator: SeededNeuralSimulator | None = None,
        ranker: TransparentRanker | None = None,
        session_policy: SessionPolicyConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        session_id_factory: Callable[[], str] | None = None,
        nonce_factory: Callable[[], str] | None = None,
    ) -> None:
        self.profiles = {profile.profile_id: profile for profile in profiles}
        self.knowledge_store = knowledge_store
        self.candidate_generator = candidate_generator or CandidateGenerator(
            FixtureCandidateBackend()
        )
        self.retriever = retriever or LexicalRetriever(knowledge_store)
        self.simulator = simulator or SeededNeuralSimulator()
        self.ranker = ranker or TransparentRanker()
        self.session_policy = session_policy or SessionPolicyConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.session_id_factory = session_id_factory or (lambda: f"session-{uuid.uuid4().hex[:20]}")
        self.nonce_factory = nonce_factory or (lambda: secrets.token_urlsafe(24))
        self._sessions: dict[str, _SessionRuntime] = {}

    def close(self) -> None:
        self.knowledge_store.close()

    def create_session(self, request: CreateSessionRequest) -> SessionView:
        if request.profile_id not in self.profiles:
            raise SessionValidationError(f"unknown profile: {request.profile_id}")
        session_id = self.session_id_factory()
        if session_id in self._sessions:
            raise SessionConflictError(f"session already exists: {session_id}")
        now = self._now()
        runtime = _SessionRuntime(
            session=MessageSession(
                session_id=session_id,
                profile_id=request.profile_id,
                created_at=now,
                updated_at=now,
            ),
            input_mode=request.input_mode,
        )
        self._sessions[session_id] = runtime
        return self._view(runtime)

    def get_session(self, session_id: str) -> SessionView:
        return self._view(self._runtime(session_id))

    def start_round(self, session_id: str, request: RoundRequest) -> SessionView:
        runtime = self._runtime(session_id)
        now = self._now()
        self._transition(runtime, SessionEvent.REQUEST_CANDIDATES, now)
        try:
            generation = self.candidate_generator.generate(
                CandidateGenerationRequest(
                    confirmed_text=runtime.session.confirmed_text,
                    candidate_count=self.session_policy.candidate_count,
                    maximum_phrase_tokens=self.session_policy.maximum_phrase_tokens,
                )
            )
            language_candidates = tuple(
                candidate
                for candidate in generation.candidate_set.candidates
                if candidate.candidate_id in generation.generic_language_support
            )
            if request.simulated_target_index >= len(language_candidates):
                raise SessionValidationError(
                    f"simulated target index {request.simulated_target_index} exceeds "
                    f"{len(language_candidates) - 1}"
                )
            target_id = language_candidates[request.simulated_target_index].candidate_id
            retrieval = self.retriever.retrieve_for_candidates(
                profile_id=runtime.session.profile_id,
                confirmed_text=runtime.session.confirmed_text,
                candidates=generation.candidate_set.candidates,
                at_time=now,
            )
            evidence, simulation = self._neural_evidence(
                runtime=runtime,
                generation=generation,
                target_id=target_id,
                now=now,
            )
            ranking = self.ranker.rank(
                RankingInputs(
                    candidate_set=generation.candidate_set,
                    neural_evidence=evidence,
                    generic_language_support=generation.generic_language_support,
                    retrieval_evidence=retrieval,
                )
            )
        except Exception:
            runtime.session.state = SessionState.DRAFT
            runtime.session.updated_at = now
            raise

        runtime.generation = generation
        runtime.ranking = ranking
        runtime.simulation = simulation
        runtime.simulated_target_id = target_id
        runtime.rejected_candidate_ids.clear()
        runtime.provisional_candidate_id = None
        runtime.provisional_action_id = None
        runtime.counters.round_count += 1
        runtime.session.active_candidate_set_id = generation.candidate_set.candidate_set_id
        runtime.session.provisional_candidate_id = None
        self._transition(runtime, SessionEvent.CANDIDATES_GENERATED, now)
        self._transition(runtime, SessionEvent.START_SELECTION, now)
        return self._view(runtime)

    def apply_action(self, session_id: str, request: SessionActionRequest) -> SessionView:
        runtime = self._runtime(session_id)
        now = self._now()
        if request.action is SelectionActionType.CANCEL:
            self._record_action(runtime, request.action, None, now)
            self._transition(runtime, SessionEvent.CANCEL_SESSION, now)
            self._clear_round(runtime)
        elif request.action is SelectionActionType.BACK:
            self._record_action(runtime, request.action, None, now)
            self._back(runtime, now)
        elif request.action is SelectionActionType.CLEAR:
            self._record_action(runtime, request.action, None, now)
            self._clear(runtime, now)
        elif request.action is SelectionActionType.OTHER:
            self._record_action(runtime, request.action, None, now)
            runtime.counters.other_count += 1
            self._return_to_draft(runtime, now)
        elif request.action is SelectionActionType.REPEAT:
            self._record_action(runtime, request.action, None, now)
            return self._repeat(runtime, now)
        elif request.action is SelectionActionType.REJECT:
            assert request.candidate_id is not None
            self._record_action(runtime, request.action, request.candidate_id, now)
            self._reject(runtime, request.candidate_id, now)
        elif request.action is SelectionActionType.SELECT:
            assert request.candidate_id is not None
            self._record_action(runtime, request.action, request.candidate_id, now)
            self._select(runtime, request.candidate_id, now)
        else:  # pragma: no cover - validated by SessionActionRequest
            raise SessionValidationError(f"unsupported action: {request.action.value}")
        return self._view(runtime)

    def resolve_selection(
        self, session_id: str, request: SelectionConfirmationRequest
    ) -> SessionView:
        runtime = self._runtime(session_id)
        now = self._now()
        if runtime.session.state is not SessionState.AWAITING_SELECTION_CONFIRMATION:
            raise SessionConflictError("session is not awaiting selection confirmation")
        if request.candidate_id != runtime.provisional_candidate_id:
            raise SessionValidationError(
                "confirmation candidate does not match the provisional choice"
            )
        if request.accept:
            self._append_provisional(runtime, now)
            self._transition(runtime, SessionEvent.CONFIRM_SELECTION, now)
        else:
            runtime.rejected_candidate_ids.add(request.candidate_id)
            runtime.counters.rejection_count += 1
            runtime.provisional_candidate_id = None
            runtime.provisional_action_id = None
            runtime.session.provisional_candidate_id = None
            self._transition(runtime, SessionEvent.REJECT_SELECTION, now)
        return self._view(runtime)

    def append_manual_text(self, session_id: str, request: ManualTextRequest) -> SessionView:
        runtime = self._runtime(session_id)
        if runtime.input_mode is not SessionInputMode.MANUAL:
            raise SessionConflictError("manual text is available only in manual input mode")
        if runtime.session.state is not SessionState.DRAFT:
            raise SessionConflictError("manual text can be appended only while drafting")
        now = self._now()
        runtime.session.confirmed_spans.append(
            ConfirmedSpan(text=request.text, action_id=self._next_action_id(runtime))
        )
        runtime.counters.manual_text_count += 1
        runtime.session.updated_at = now
        return self._view(runtime)

    def request_finalization(self, session_id: str) -> FinalizationChallenge:
        runtime = self._runtime(session_id)
        if not runtime.session.confirmed_text:
            raise SessionValidationError("cannot finalize an empty message")
        now = self._now()
        self._transition(runtime, SessionEvent.REQUEST_FINALIZATION, now)
        text = runtime.session.confirmed_text
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        nonce = self.nonce_factory()
        if len(nonce) < 16:
            runtime.session.state = SessionState.DRAFT
            raise SessionValidationError("confirmation nonce factory returned an unsafe nonce")
        expires_at = now + timedelta(
            seconds=self.session_policy.finalization_confirmation_ttl_seconds
        )
        runtime.finalization_nonce = nonce
        runtime.finalization_sha256 = text_sha256
        runtime.finalization_expires_at = expires_at
        return FinalizationChallenge(
            session_id=session_id,
            text=text,
            text_sha256=text_sha256,
            confirmation_nonce=nonce,
            high_risk_acknowledgement_required=bool(runtime.high_risk_action_ids),
            expires_at=expires_at,
        )

    def confirm_finalization(self, session_id: str, request: FinalizationRequest) -> SessionView:
        runtime = self._runtime(session_id)
        now = self._now()
        if runtime.session.state is not SessionState.AWAITING_FINAL_CONFIRMATION:
            raise SessionConflictError("session is not awaiting final confirmation")
        if request.session_id != session_id:
            raise SessionValidationError("confirmation session ID does not match the path")
        if runtime.finalization_expires_at is None or now > runtime.finalization_expires_at:
            raise SessionConflictError("finalization confirmation has expired")
        if runtime.finalization_nonce is None or not secrets.compare_digest(
            request.confirmation_nonce, runtime.finalization_nonce
        ):
            raise SessionValidationError("invalid finalization confirmation nonce")
        if request.text_sha256 != runtime.finalization_sha256:
            raise SessionValidationError("finalization text hash does not match")
        if runtime.high_risk_action_ids and not request.high_risk_acknowledged:
            raise SessionValidationError("high-risk content acknowledgement is required")
        self._transition(runtime, SessionEvent.CONFIRM_FINALIZATION, now)
        self._clear_finalization(runtime)
        return self._view(runtime)

    def reject_finalization(self, session_id: str) -> SessionView:
        runtime = self._runtime(session_id)
        now = self._now()
        self._transition(runtime, SessionEvent.REJECT_FINALIZATION, now)
        self._clear_finalization(runtime)
        return self._view(runtime)

    def _neural_evidence(
        self,
        *,
        runtime: _SessionRuntime,
        generation: CandidateGenerationResult,
        target_id: str,
        now: datetime,
    ) -> tuple[NeuralSelectionEvidence, SimulatedRound | None]:
        if runtime.input_mode is SessionInputMode.MANUAL:
            digest = hashlib.sha256(
                f"{runtime.session.session_id}:{runtime.counters.round_count}".encode()
            ).hexdigest()
            return (
                NeuralSelectionEvidence(
                    evidence_id=f"manual-{digest[:20]}",
                    mode=EvidenceMode.MANUAL,
                    missing_reason="manual debug mode has no neural evidence",
                    session_id=runtime.session.session_id,
                    trial_id=f"round-{runtime.counters.round_count:06d}",
                    recorded_at=now,
                ),
                None,
            )
        simulated = self.simulator.simulate(
            candidate_ids=tuple(
                candidate.candidate_id for candidate in generation.candidate_set.candidates
            ),
            intended_candidate_id=target_id,
            session_id=runtime.session.session_id,
            round_index=runtime.counters.round_count,
            subject_id=runtime.session.profile_id,
        )
        return simulated.evidence, simulated

    def _repeat(self, runtime: _SessionRuntime, now: datetime) -> SessionView:
        if runtime.input_mode is not SessionInputMode.SIMULATION:
            raise SessionConflictError("repeat requires a simulated or future neural input")
        if runtime.session.state is SessionState.CANDIDATES_READY:
            self._transition(runtime, SessionEvent.START_SELECTION, now)
        if runtime.session.state not in {
            SessionState.SELECTING,
            SessionState.AWAITING_SELECTION_CONFIRMATION,
        }:
            raise SessionConflictError("repeat is unavailable in the current session state")
        if runtime.generation is None or runtime.simulated_target_id is None:
            raise SessionConflictError("repeat requires an active candidate round")
        self._transition(runtime, SessionEvent.REPEAT_SELECTION, now)
        runtime.provisional_candidate_id = None
        runtime.provisional_action_id = None
        runtime.session.provisional_candidate_id = None
        simulated = self.simulator.simulate(
            candidate_ids=tuple(
                candidate.candidate_id for candidate in runtime.generation.candidate_set.candidates
            ),
            intended_candidate_id=runtime.simulated_target_id,
            session_id=runtime.session.session_id,
            round_index=runtime.counters.round_count,
            subject_id=runtime.session.profile_id,
        )
        retrieval = self.retriever.retrieve_for_candidates(
            profile_id=runtime.session.profile_id,
            confirmed_text=runtime.session.confirmed_text,
            candidates=runtime.generation.candidate_set.candidates,
            at_time=now,
        )
        runtime.ranking = self.ranker.rank(
            RankingInputs(
                candidate_set=runtime.generation.candidate_set,
                neural_evidence=simulated.evidence,
                generic_language_support=runtime.generation.generic_language_support,
                retrieval_evidence=retrieval,
            )
        )
        runtime.simulation = simulated
        runtime.counters.round_count += 1
        runtime.counters.repeat_count += 1
        runtime.session.updated_at = now
        return self._view(runtime)

    def _select(self, runtime: _SessionRuntime, candidate_id: str, now: datetime) -> None:
        if runtime.session.state is SessionState.CANDIDATES_READY:
            self._transition(runtime, SessionEvent.START_SELECTION, now)
        if runtime.session.state is not SessionState.SELECTING:
            raise SessionConflictError("selection is unavailable in the current session state")
        if runtime.generation is None or runtime.ranking is None:
            raise SessionConflictError("selection requires an active ranked candidate set")
        if candidate_id in runtime.rejected_candidate_ids:
            raise SessionValidationError("candidate was explicitly rejected in this round")
        candidate = self._candidate(runtime, candidate_id)
        control = runtime.generation.control_actions.get(candidate_id)
        if control is ControlPath.CANCEL:
            self._transition(runtime, SessionEvent.CANCEL_SESSION, now)
            self._clear_round(runtime)
            return
        if control is ControlPath.BACK:
            self._back(runtime, now)
            return
        if control is ControlPath.OTHER:
            runtime.counters.other_count += 1
            self._return_to_draft(runtime, now)
            return
        if (
            runtime.input_mode is SessionInputMode.SIMULATION
            and runtime.ranking.disposition is not RankingDisposition.DISPLAY
        ):
            raise SessionConflictError("ranking requires repeat or abstention before selection")

        ranked = next(
            item
            for item in runtime.ranking.ranked_candidates
            if item.candidate.candidate_id == candidate_id
        )
        action_id = self._next_action_id(runtime)
        enhanced = (
            ranked.confirmation_level is ConfirmationLevel.ENHANCED
            or candidate_id != runtime.ranking.fused_top_candidate_id
        )
        if enhanced:
            runtime.provisional_candidate_id = candidate_id
            runtime.provisional_action_id = action_id
            runtime.session.provisional_candidate_id = candidate_id
            self._transition(runtime, SessionEvent.REQUIRE_SELECTION_CONFIRMATION, now)
            return
        runtime.session.confirmed_spans.append(
            ConfirmedSpan(text=candidate.text, action_id=action_id)
        )
        runtime.counters.selection_count += 1
        self._transition(runtime, SessionEvent.ACCEPT_SELECTION, now)
        self._clear_round(runtime)

    def _append_provisional(self, runtime: _SessionRuntime, now: datetime) -> None:
        if runtime.provisional_candidate_id is None or runtime.provisional_action_id is None:
            raise SessionConflictError("no provisional selection is available")
        if runtime.ranking is None:
            raise SessionConflictError("no active ranking is available")
        candidate = self._candidate(runtime, runtime.provisional_candidate_id)
        ranked = next(
            item
            for item in runtime.ranking.ranked_candidates
            if item.candidate.candidate_id == candidate.candidate_id
        )
        runtime.session.confirmed_spans.append(
            ConfirmedSpan(text=candidate.text, action_id=runtime.provisional_action_id)
        )
        if ranked.risk_level is not RiskLevel.NONE:
            runtime.high_risk_action_ids.add(runtime.provisional_action_id)
        runtime.counters.selection_count += 1
        runtime.provisional_candidate_id = None
        runtime.provisional_action_id = None
        runtime.session.provisional_candidate_id = None
        runtime.session.updated_at = now
        self._clear_round(runtime)

    def _reject(self, runtime: _SessionRuntime, candidate_id: str, now: datetime) -> None:
        self._candidate(runtime, candidate_id)
        if runtime.session.state is SessionState.AWAITING_SELECTION_CONFIRMATION:
            if candidate_id != runtime.provisional_candidate_id:
                raise SessionValidationError(
                    "rejection candidate does not match provisional choice"
                )
            runtime.provisional_candidate_id = None
            runtime.provisional_action_id = None
            runtime.session.provisional_candidate_id = None
        elif runtime.session.state is not SessionState.SELECTING:
            raise SessionConflictError("rejection is unavailable in the current session state")
        runtime.rejected_candidate_ids.add(candidate_id)
        runtime.counters.rejection_count += 1
        self._transition(runtime, SessionEvent.REJECT_SELECTION, now)

    def _back(self, runtime: _SessionRuntime, now: datetime) -> None:
        if runtime.session.confirmed_spans:
            removed = runtime.session.confirmed_spans.pop()
            runtime.high_risk_action_ids.discard(removed.action_id)
        runtime.counters.backtrack_count += 1
        self._return_to_draft(runtime, now)

    def _clear(self, runtime: _SessionRuntime, now: datetime) -> None:
        runtime.session.confirmed_spans.clear()
        runtime.high_risk_action_ids.clear()
        runtime.counters.clear_count += 1
        self._return_to_draft(runtime, now)

    def _return_to_draft(self, runtime: _SessionRuntime, now: datetime) -> None:
        self._transition(runtime, SessionEvent.RETURN_TO_DRAFT, now)
        self._clear_round(runtime)

    def _record_action(
        self,
        runtime: _SessionRuntime,
        action: SelectionActionType,
        candidate_id: str | None,
        now: datetime,
    ) -> None:
        runtime.actions.append(
            SelectionAction(
                action=action,
                input_mode=(
                    EvidenceMode.MANUAL
                    if runtime.input_mode is SessionInputMode.MANUAL
                    else EvidenceMode.SIMULATION
                ),
                candidate_id=candidate_id,
                evidence_id=(
                    runtime.ranking.neural_evidence_id if runtime.ranking is not None else None
                ),
                occurred_at=now,
            )
        )

    def _candidate(self, runtime: _SessionRuntime, candidate_id: str) -> Candidate:
        if runtime.generation is None:
            raise SessionConflictError("no active candidate set")
        try:
            return next(
                candidate
                for candidate in runtime.generation.candidate_set.candidates
                if candidate.candidate_id == candidate_id
            )
        except StopIteration as error:
            raise SessionValidationError(
                f"candidate is not visible in the active round: {candidate_id}"
            ) from error

    def _next_action_id(self, runtime: _SessionRuntime) -> str:
        runtime.action_sequence += 1
        digest = hashlib.sha256(
            f"{runtime.session.session_id}:{runtime.action_sequence}".encode()
        ).hexdigest()
        return f"action-{digest[:20]}"

    def _transition(self, runtime: _SessionRuntime, event: SessionEvent, now: datetime) -> None:
        try:
            runtime.session.state = transition(runtime.session.state, event)
        except InvalidTransitionError as error:
            raise SessionConflictError(str(error)) from error
        runtime.session.updated_at = now

    @staticmethod
    def _clear_round(runtime: _SessionRuntime) -> None:
        runtime.generation = None
        runtime.ranking = None
        runtime.simulation = None
        runtime.simulated_target_id = None
        runtime.rejected_candidate_ids.clear()
        runtime.provisional_candidate_id = None
        runtime.provisional_action_id = None
        runtime.session.active_candidate_set_id = None
        runtime.session.provisional_candidate_id = None

    @staticmethod
    def _clear_finalization(runtime: _SessionRuntime) -> None:
        runtime.finalization_nonce = None
        runtime.finalization_sha256 = None
        runtime.finalization_expires_at = None

    def _runtime(self, session_id: str) -> _SessionRuntime:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(f"session not found: {session_id}") from error

    def _view(self, runtime: _SessionRuntime) -> SessionView:
        return SessionView(
            session=runtime.session.model_copy(deep=True),
            input_mode=runtime.input_mode,
            active_generation=runtime.generation,
            ranking=runtime.ranking,
            rejected_candidate_ids=tuple(sorted(runtime.rejected_candidate_ids)),
            pending_selection_candidate_id=runtime.provisional_candidate_id,
            finalization_pending=(
                runtime.session.state is SessionState.AWAITING_FINAL_CONFIRMATION
            ),
            high_risk_acknowledgement_required=bool(runtime.high_risk_action_ids),
            metrics=runtime.counters.snapshot(),
        )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise SessionValidationError("session clock must return a timezone-aware datetime")
        return value


def build_demo_orchestrator() -> SessionOrchestrator:
    """Build the local synthetic service without downloading data or model weights."""

    profiles = load_profiles()
    store = SQLiteKnowledgeStore(":memory:")
    imported_at = datetime.now(UTC)
    for profile in profiles:
        for record in profile.knowledge:
            store.add(
                profile_id=profile.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=imported_at,
            )
    return SessionOrchestrator(profiles=profiles, knowledge_store=store)
