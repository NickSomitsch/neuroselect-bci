"""Deterministic held-out simulation runner over the real vertical-slice components."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neuroselect.bci import SeededNeuralSimulator
from neuroselect.core.models import CandidateKind, NeuralSelectionEvidence, RecordPermission
from neuroselect.evaluation.conditions import condition_by_id, condition_catalog
from neuroselect.evaluation.metrics import calculate_metrics
from neuroselect.evaluation.models import (
    ConditionAvailability,
    EvaluationCondition,
    ExperimentResult,
    NeuralMode,
    RankingMode,
    RetrievalMode,
    SimulatedExperimentSpec,
    TrialRecord,
)
from neuroselect.language import (
    BackendMetadata,
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateGenerator,
    CandidateProposal,
)
from neuroselect.ranking import (
    ConfirmationLevel,
    RankingDisposition,
    RankingInputs,
    TransparentRanker,
)
from neuroselect.retrieval import (
    CandidateRetrievalEvidence,
    KnowledgeRecordInput,
    LexicalRetriever,
    RetrievalHit,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import BenchmarkMessage, GeneratedBenchmark, SyntheticProfile

FALLBACK_DISTRACTORS = (
    "Please",
    "Thank you",
    "Could you",
    "I need help",
    "Yes",
    "No",
    "One moment",
    "I am ready",
    "Maybe later",
    "Please wait",
    "That is right",
    "Try again",
)


class ExperimentConfigurationError(ValueError):
    """Raised when a requested experiment would make an unsupported claim."""


class _ControlledCandidateBackend:
    metadata = BackendMetadata(
        backend_id="controlled-evaluation",
        model_id="neuroselect/controlled-target-presence",
        model_revision="controlled-proposals-v1",
        generator_revision="deterministic-generator-v1",
        prompt_revision="controlled-target-presence-v1",
        deterministic=True,
    )

    def __init__(self, proposals: tuple[CandidateProposal, ...]) -> None:
        self.proposals = proposals

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]:
        del request
        return self.proposals


@dataclass(frozen=True)
class _PreparedTrial:
    ordinal: int
    message: BenchmarkMessage
    span_index: int
    confirmed_context: str
    target_text: str
    conflict: bool
    generation: CandidateGenerationResult
    target_candidate_id: str
    simulated_evidence: NeuralSelectionEvidence
    current_retrieval: tuple[CandidateRetrievalEvidence, ...]
    no_query_context_retrieval: tuple[CandidateRetrievalEvidence, ...]


def load_experiment_spec(path: str | Path) -> SimulatedExperimentSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must contain a YAML mapping")
    return SimulatedExperimentSpec.model_validate(payload)


class SimulatedExperimentRunner:
    """Evaluate paired conditions without training, network access, or private data."""

    def __init__(self, spec: SimulatedExperimentSpec) -> None:
        self.spec = spec
        self.simulator = SeededNeuralSimulator(spec.simulator)
        self.ranker = TransparentRanker()

    def run(
        self,
        *,
        benchmark: GeneratedBenchmark,
        profiles: tuple[SyntheticProfile, ...],
    ) -> ExperimentResult:
        profile_by_id = {profile.profile_id: profile for profile in profiles}
        self._validate_inputs(benchmark=benchmark, profile_by_id=profile_by_id)
        messages = self._select_messages(benchmark)
        distractor_pool = self._distractor_pool(messages)

        with SQLiteKnowledgeStore(":memory:") as store:
            for profile_id in self.spec.profile_ids:
                for record in profile_by_id[profile_id].knowledge:
                    store.add(
                        profile_id=profile_id,
                        record=KnowledgeRecordInput.model_validate(record.model_dump()),
                        at_time=self.spec.evaluation_time,
                    )
            retriever = LexicalRetriever(store)
            prepared = self._prepare_trials(
                messages=messages,
                distractor_pool=distractor_pool,
                retriever=retriever,
            )
            records = tuple(
                self._evaluate(prepared_trial=trial, condition=condition, store=store)
                for condition in self.spec.conditions
                for trial in prepared
            )

        metrics = calculate_metrics(
            records,
            self.spec.conditions,
            self.spec.calibration_bins,
        )
        config_sha256 = self.spec.digest()
        run_digest = hashlib.sha256(
            f"{config_sha256}:{benchmark.source_sha256}".encode()
        ).hexdigest()
        return ExperimentResult(
            run_id=f"sim-eval-{run_digest[:20]}",
            generated_at=self.spec.evaluation_time,
            config_sha256=config_sha256,
            benchmark_source_sha256=benchmark.source_sha256,
            spec=self.spec,
            condition_catalog=condition_catalog(),
            trial_records=records,
            metrics=metrics,
            limitations=(
                "This is controlled simulation, not original-task EEG or counterfactual replay.",
                "The known held-out target is injected into each visible set, so target "
                "availability measures the fusion protocol rather than generative recall.",
                "Correct-selection speed and latency use a versioned interaction-time model, "
                "not participant timing or wall-clock benchmark runtime.",
                "An ideal explicit confirmer rejects every incorrect top recommendation; "
                "therefore unintended-word rate measures the confirmation safety invariant.",
                "Personalized LoRA and complete calibrated-system conditions remain unavailable "
                "in this fixture recipe until local trained and decoder artifacts are supplied.",
                "Full conversation-context removal remains unavailable in this fixture recipe; "
                "the retrieval-context ablation removes context only from retrieval queries.",
            ),
        )

    def _validate_inputs(
        self,
        *,
        benchmark: GeneratedBenchmark,
        profile_by_id: dict[str, SyntheticProfile],
    ) -> None:
        missing_profiles = sorted(set(self.spec.profile_ids).difference(profile_by_id))
        if missing_profiles:
            raise ExperimentConfigurationError(
                f"unknown synthetic profiles: {', '.join(missing_profiles)}"
            )
        benchmark_profiles = set(benchmark.profile_ids)
        missing_benchmark_profiles = sorted(
            set(self.spec.profile_ids).difference(benchmark_profiles)
        )
        if missing_benchmark_profiles:
            raise ExperimentConfigurationError(
                f"profiles are absent from the benchmark: {', '.join(missing_benchmark_profiles)}"
            )
        unavailable = [
            condition_by_id(condition)
            for condition in self.spec.conditions
            if condition_by_id(condition).availability is ConditionAvailability.UNAVAILABLE
        ]
        if unavailable:
            details = "; ".join(
                f"{item.condition.value}: {item.unavailable_reason}" for item in unavailable
            )
            raise ExperimentConfigurationError(
                f"requested conditions have unmet dependencies: {details}"
            )

    def _select_messages(self, benchmark: GeneratedBenchmark) -> tuple[BenchmarkMessage, ...]:
        selected: list[BenchmarkMessage] = []
        split_messages = benchmark.messages[self.spec.split]
        for profile_id in self.spec.profile_ids:
            profile_messages = sorted(
                (message for message in split_messages if message.profile_id == profile_id),
                key=lambda message: message.message_id,
            )
            if len(profile_messages) < self.spec.message_limit_per_profile:
                raise ExperimentConfigurationError(
                    f"profile {profile_id} has {len(profile_messages)} messages in "
                    f"{self.spec.split.value}; {self.spec.message_limit_per_profile} requested"
                )
            selected.extend(profile_messages[: self.spec.message_limit_per_profile])
        return tuple(selected)

    def _distractor_pool(self, messages: tuple[BenchmarkMessage, ...]) -> tuple[str, ...]:
        values = (
            *(
                span
                for message in messages
                for span in message.target_spans
                if len(span.split()) <= self.spec.maximum_phrase_tokens
            ),
            *FALLBACK_DISTRACTORS,
        )
        unique: dict[str, str] = {}
        for value in values:
            if len(value.split()) <= self.spec.maximum_phrase_tokens:
                unique.setdefault(value.casefold(), value)
        return tuple(unique.values())

    def _prepare_trials(
        self,
        *,
        messages: tuple[BenchmarkMessage, ...],
        distractor_pool: tuple[str, ...],
        retriever: LexicalRetriever,
    ) -> tuple[_PreparedTrial, ...]:
        prepared: list[_PreparedTrial] = []
        ordinal = 0
        for message in messages:
            confirmed: list[str] = []
            for span_index, target_text in enumerate(message.target_spans):
                if len(target_text.split()) > self.spec.maximum_phrase_tokens:
                    raise ExperimentConfigurationError(
                        f"target span exceeds maximum_phrase_tokens: {target_text!r}"
                    )
                context = " ".join(confirmed)
                conflict = ordinal % self.spec.language_conflict_every_n_trials == 0
                generation = self._generation(
                    target_text=target_text,
                    confirmed_context=context,
                    distractor_pool=distractor_pool,
                    ordinal=ordinal,
                    conflict=conflict,
                )
                target_candidate = next(
                    candidate
                    for candidate in generation.candidate_set.candidates
                    if candidate.kind is not CandidateKind.CONTROL
                    and candidate.text.casefold() == target_text.casefold()
                )
                simulated = self.simulator.simulate(
                    candidate_ids=tuple(
                        candidate.candidate_id for candidate in generation.candidate_set.candidates
                    ),
                    intended_candidate_id=target_candidate.candidate_id,
                    session_id=f"evaluation:{message.profile_id}:{message.message_id}",
                    round_index=span_index,
                    subject_id=message.profile_id,
                )
                prepared.append(
                    _PreparedTrial(
                        ordinal=ordinal,
                        message=message,
                        span_index=span_index,
                        confirmed_context=context,
                        target_text=target_text,
                        conflict=conflict,
                        generation=generation,
                        target_candidate_id=target_candidate.candidate_id,
                        simulated_evidence=simulated.evidence,
                        current_retrieval=retriever.retrieve_for_candidates(
                            profile_id=message.profile_id,
                            confirmed_text=context,
                            candidates=generation.candidate_set.candidates,
                            at_time=self.spec.evaluation_time,
                        ),
                        no_query_context_retrieval=retriever.retrieve_for_candidates(
                            profile_id=message.profile_id,
                            confirmed_text="",
                            candidates=generation.candidate_set.candidates,
                            at_time=self.spec.evaluation_time,
                        ),
                    )
                )
                confirmed.append(target_text)
                ordinal += 1
        return tuple(prepared)

    def _generation(
        self,
        *,
        target_text: str,
        confirmed_context: str,
        distractor_pool: tuple[str, ...],
        ordinal: int,
        conflict: bool,
    ) -> CandidateGenerationResult:
        language_count = self.spec.candidate_count - 3
        eligible = [
            value for value in distractor_pool if value.casefold() != target_text.casefold()
        ]
        if len(eligible) < language_count - 1:
            raise ExperimentConfigurationError("not enough unique controlled distractors")
        offset = ordinal % len(eligible)
        ordered = (*eligible[offset:], *eligible[:offset])
        distractors = ordered[: language_count - 1]
        if conflict:
            supports = (0.02, *(0.9 / (index + 1) for index in range(len(distractors))))
        else:
            supports = (0.72, *(0.24 / (index + 1) for index in range(len(distractors))))
        proposals = tuple(
            CandidateProposal(text=text, support=support)
            for text, support in zip((target_text, *distractors), supports, strict=True)
        )
        return CandidateGenerator(_ControlledCandidateBackend(proposals)).generate(
            CandidateGenerationRequest(
                confirmed_text=confirmed_context,
                candidate_count=self.spec.candidate_count,
                maximum_phrase_tokens=self.spec.maximum_phrase_tokens,
            )
        )

    def _evaluate(
        self,
        *,
        prepared_trial: _PreparedTrial,
        condition: EvaluationCondition,
        store: SQLiteKnowledgeStore,
    ) -> TrialRecord:
        definition = condition_by_id(condition)
        evidence = self._neural_evidence(prepared_trial.simulated_evidence, definition.neural_mode)
        retrieval = self._retrieval_evidence(
            prepared_trial=prepared_trial,
            mode=definition.retrieval_mode,
            store=store,
            remove_query_context=(
                condition is EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT
            ),
        )
        generation = prepared_trial.generation
        candidate_ids = tuple(
            candidate.candidate_id for candidate in generation.candidate_set.candidates
        )
        language_top_id = min(
            generation.generic_language_support,
            key=lambda candidate_id: (
                -generation.generic_language_support[candidate_id],
                candidate_ids.index(candidate_id),
            ),
        )
        neural_top_id = (
            max(
                evidence.candidate_probabilities,
                key=evidence.candidate_probabilities.__getitem__,
            )
            if evidence is not None
            else None
        )

        if definition.ranking_mode is RankingMode.TRANSPARENT_SAFE_FUSION:
            assert evidence is not None
            ranking = self.ranker.rank(
                RankingInputs(
                    candidate_set=generation.candidate_set,
                    neural_evidence=evidence,
                    generic_language_support=generation.generic_language_support,
                    retrieval_evidence=retrieval,
                )
            )
            ranked_ids = tuple(item.candidate.candidate_id for item in ranking.ranked_candidates)
            disposition = ranking.disposition
            reason_codes = tuple(reason.value for reason in ranking.reason_codes)
            target_item = next(
                item
                for item in ranking.ranked_candidates
                if item.candidate.candidate_id == prepared_trial.target_candidate_id
            )
            enhanced = (
                target_item.confirmation_level is ConfirmationLevel.ENHANCED
                or prepared_trial.target_candidate_id != ranking.fused_top_candidate_id
            )
        else:
            ranked_ids = self._baseline_ranking(
                prepared_trial=prepared_trial,
                ranking_mode=definition.ranking_mode,
                evidence=evidence,
                retrieval=retrieval,
            )
            disposition = RankingDisposition.DISPLAY
            reason_codes = ("counterfactual_baseline_without_abstention",)
            enhanced = prepared_trial.target_candidate_id != ranked_ids[0]

        target_rank = ranked_ids.index(prepared_trial.target_candidate_id) + 1
        top_correct = target_rank == 1
        selection_completed = disposition is RankingDisposition.DISPLAY and top_correct
        correction_required = disposition is RankingDisposition.DISPLAY and not top_correct
        explicit_actions = int(
            disposition in {RankingDisposition.DISPLAY, RankingDisposition.REQUEST_REPEAT}
        )
        duration = (
            self.spec.timing.candidate_round_seconds
            + explicit_actions * self.spec.timing.explicit_action_seconds
            + int(selection_completed and enhanced) * self.spec.timing.enhanced_confirmation_seconds
        )
        calibration = self._calibration(evidence, prepared_trial.target_candidate_id)
        trial_material = (
            f"{condition.value}:{prepared_trial.message.message_id}:{prepared_trial.span_index}"
        )
        return TrialRecord(
            trial_id=f"trial-{hashlib.sha256(trial_material.encode()).hexdigest()[:20]}",
            condition=condition,
            profile_id=prepared_trial.message.profile_id,
            message_id=prepared_trial.message.message_id,
            span_index=prepared_trial.span_index,
            message_span_count=len(prepared_trial.message.target_spans),
            confirmed_context=prepared_trial.confirmed_context,
            retrieval_query_context_removed=(
                condition is EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT
            ),
            target_text=prepared_trial.target_text,
            target_word_count=len(prepared_trial.target_text.split()),
            candidate_ids=candidate_ids,
            ranked_candidate_ids=ranked_ids,
            target_candidate_id=prepared_trial.target_candidate_id,
            target_rank=target_rank,
            top_candidate_id=ranked_ids[0],
            neural_top_candidate_id=neural_top_id,
            language_top_candidate_id=language_top_id,
            disposition=disposition,
            reason_codes=reason_codes,
            language_conflict_context=prepared_trial.conflict,
            neural_language_conflict=(
                neural_top_id is not None and neural_top_id != language_top_id
            ),
            neural_target_probability=(
                evidence.candidate_probabilities[prepared_trial.target_candidate_id]
                if evidence is not None
                else None
            ),
            prediction_confidence=calibration[0],
            prediction_correct=calibration[1],
            neural_brier_score=calibration[2],
            top_1_correct=top_correct,
            top_3_correct=target_rank <= 3,
            explicit_selection_completed=selection_completed,
            enhanced_confirmation_required=enhanced,
            correction_required=correction_required,
            explicit_action_count=explicit_actions,
            retrieval_hit_count=sum(len(item.hits) for item in retrieval),
            modeled_duration_seconds=duration,
        )

    def _baseline_ranking(
        self,
        *,
        prepared_trial: _PreparedTrial,
        ranking_mode: RankingMode,
        evidence: NeuralSelectionEvidence | None,
        retrieval: tuple[CandidateRetrievalEvidence, ...],
    ) -> tuple[str, ...]:
        generation = prepared_trial.generation
        candidates = generation.candidate_set.candidates
        original_order = {
            candidate.candidate_id: index for index, candidate in enumerate(candidates)
        }
        retrieval_support = {item.candidate_id: item.retrieval_support for item in retrieval}
        scores: dict[str, float] = {}
        for candidate in candidates:
            neural = (
                evidence.candidate_probabilities.get(candidate.candidate_id, 0.0)
                if evidence is not None
                else 0.0
            )
            language = generation.generic_language_support.get(candidate.candidate_id, 0.0)
            if ranking_mode is RankingMode.NEURAL_ONLY:
                score = neural
            elif ranking_mode is RankingMode.LANGUAGE_ONLY:
                score = language
            else:
                score = 0.65 * neural + 0.15 * language
                if retrieval:
                    score += 0.12 * retrieval_support.get(candidate.candidate_id, 0.0)
            scores[candidate.candidate_id] = score
        return tuple(
            candidate_id
            for candidate_id, _ in sorted(
                scores.items(),
                key=lambda item: (-item[1], original_order[item[0]]),
            )
        )

    def _neural_evidence(
        self,
        evidence: NeuralSelectionEvidence,
        mode: NeuralMode,
    ) -> NeuralSelectionEvidence | None:
        if mode is NeuralMode.MISSING:
            return None
        if mode is NeuralMode.SIMULATED:
            return evidence
        candidate_ids = tuple(evidence.candidate_probabilities)
        if mode is NeuralMode.UNIFORM:
            probabilities = dict.fromkeys(candidate_ids, 1.0 / len(candidate_ids))
        else:
            probabilities = {
                candidate_id: evidence.candidate_probabilities[
                    candidate_ids[(index + 1) % len(candidate_ids)]
                ]
                for index, candidate_id in enumerate(candidate_ids)
            }
        ordered = sorted(probabilities.values(), reverse=True)
        entropy = -sum(value * math.log(value) for value in probabilities.values())
        return NeuralSelectionEvidence(
            evidence_id=f"{evidence.evidence_id}-{mode.value}",
            mode=evidence.mode,
            candidate_probabilities=probabilities,
            calibration_id=evidence.calibration_id,
            entropy=entropy,
            top_margin=ordered[0] - ordered[1],
            subject_id=evidence.subject_id,
            session_id=evidence.session_id,
            trial_id=evidence.trial_id,
            recorded_at=evidence.recorded_at,
        )

    def _retrieval_evidence(
        self,
        *,
        prepared_trial: _PreparedTrial,
        mode: RetrievalMode,
        store: SQLiteKnowledgeStore,
        remove_query_context: bool,
    ) -> tuple[CandidateRetrievalEvidence, ...]:
        if mode is RetrievalMode.NONE:
            return ()
        base = (
            prepared_trial.no_query_context_retrieval
            if remove_query_context
            else prepared_trial.current_retrieval
        )
        if mode is RetrievalMode.CURRENT:
            return base
        if mode is RetrievalMode.SHUFFLED:
            if not base:
                return ()
            return tuple(
                CandidateRetrievalEvidence(
                    candidate_id=item.candidate_id,
                    retrieval_support=base[(index + 1) % len(base)].retrieval_support,
                    record_ids=base[(index + 1) % len(base)].record_ids,
                    hits=base[(index + 1) % len(base)].hits,
                )
                for index, item in enumerate(base)
            )

        records = tuple(
            record
            for record in store.list_records(profile_id=prepared_trial.message.profile_id)
            if not record.injection_risk
            and record.is_active_at(self.spec.evaluation_time)
            and RecordPermission.SUGGEST in record.permissions
        )
        if not records:
            raise ExperimentConfigurationError(
                "irrelevant-retrieval ablation needs an active record"
            )
        language_ids = tuple(prepared_trial.generation.generic_language_support)
        language_top_id = max(
            language_ids,
            key=prepared_trial.generation.generic_language_support.__getitem__,
        )
        hit = RetrievalHit(
            record=records[0],
            score=1.0,
            matched_terms=("controlled-irrelevant",),
            explanation="Deliberately irrelevant record injected by the controlled ablation.",
        )
        return tuple(
            CandidateRetrievalEvidence(
                candidate_id=candidate_id,
                retrieval_support=1.0 if candidate_id == language_top_id else 0.0,
                record_ids=(hit.record.record_id,) if candidate_id == language_top_id else (),
                hits=(hit,) if candidate_id == language_top_id else (),
            )
            for candidate_id in language_ids
        )

    @staticmethod
    def _calibration(
        evidence: NeuralSelectionEvidence | None,
        target_candidate_id: str,
    ) -> tuple[float | None, bool | None, float | None]:
        if evidence is None:
            return None, None, None
        probabilities = evidence.candidate_probabilities
        top_id = max(probabilities, key=probabilities.__getitem__)
        confidence = probabilities[top_id]
        brier = sum(
            (probability - float(candidate_id == target_candidate_id)) ** 2
            for candidate_id, probability in probabilities.items()
        ) / len(probabilities)
        return confidence, top_id == target_candidate_id, brier
