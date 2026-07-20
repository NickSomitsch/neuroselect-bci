"""Paired offline counterfactual replay and neural-language fusion experiments."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from neuroselect.bci import (
    FlashProbability,
    FlashProbabilityTrial,
    aggregate_flash_probabilities,
    remap_recorded_target,
)
from neuroselect.core.models import NeuralSelectionEvidence
from neuroselect.decoding.models import DecoderEvaluation, EpochPrediction
from neuroselect.eeg import P300Label
from neuroselect.evaluation.counterfactual_models import (
    CounterfactualExperimentInput,
    CounterfactualFusionResult,
    CounterfactualFusionSpec,
    CounterfactualFusionTrial,
    CounterfactualTrialProvenance,
    PairedBootstrapInterval,
)
from neuroselect.evaluation.metrics import calculate_metrics
from neuroselect.evaluation.models import EvaluationCondition, TrialRecord
from neuroselect.ranking import (
    ConfirmationLevel,
    RankingDisposition,
    RankingInputs,
    TransparentRanker,
)
from neuroselect.retrieval import CandidateRetrievalEvidence


class CounterfactualConfigurationError(ValueError):
    """Raised when counterfactual inputs cannot support the requested claims or ablations."""


def load_counterfactual_spec(path: str | Path) -> CounterfactualFusionSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("counterfactual configuration must contain a YAML mapping")
    return CounterfactualFusionSpec.model_validate(payload)


def load_counterfactual_input(path: str | Path) -> CounterfactualExperimentInput:
    return CounterfactualExperimentInput.model_validate_json(Path(path).read_text(encoding="utf-8"))


def shuffle_retrieval_across_candidates(
    trial: CounterfactualFusionTrial,
) -> tuple[CandidateRetrievalEvidence, ...]:
    """Rotate each retrieved evidence bundle to a different language candidate."""

    language_ids = tuple(
        candidate.candidate_id
        for candidate in trial.candidate_set.candidates
        if candidate.candidate_id in trial.generic_language_support
    )
    if len(language_ids) < 2:
        raise CounterfactualConfigurationError(
            "shuffled retrieval requires at least two language candidates"
        )
    next_candidate = {
        candidate_id: language_ids[(index + 1) % len(language_ids)]
        for index, candidate_id in enumerate(language_ids)
    }
    return tuple(
        evidence.model_copy(update={"candidate_id": next_candidate[evidence.candidate_id]})
        for evidence in trial.retrieval_evidence
    )


def flash_trials_from_decoder_evaluation(
    evaluation: DecoderEvaluation,
) -> tuple[FlashProbabilityTrial, ...]:
    """Extract labeled, timed original-task trials without changing labels or event order."""

    grouped: dict[str, list[EpochPrediction]] = defaultdict(list)
    for prediction in evaluation.predictions:
        grouped[prediction.selection_trial_id].append(prediction)
    output: list[FlashProbabilityTrial] = []
    for selection_trial_id, raw_predictions in sorted(grouped.items()):
        if any(
            prediction.true_label is P300Label.UNKNOWN
            or prediction.stimulus_code is None
            or prediction.onset_seconds is None
            for prediction in raw_predictions
        ):
            continue

        def source_onset(prediction: EpochPrediction) -> float:
            assert prediction.onset_seconds is not None
            return prediction.onset_seconds

        predictions = sorted(raw_predictions, key=source_onset)
        subject_ids = {prediction.subject_id for prediction in predictions}
        session_ids = {prediction.session_id for prediction in predictions}
        if len(subject_ids) != 1 or len(session_ids) != 1:
            raise ValueError("one decoder selection trial cannot mix subjects or sessions")
        target_codes = tuple(
            sorted(
                {
                    prediction.stimulus_code
                    for prediction in predictions
                    if prediction.true_label is P300Label.TARGET
                    and prediction.stimulus_code is not None
                }
            )
        )
        if not target_codes:
            continue
        events: list[FlashProbability] = []
        for index, prediction in enumerate(predictions):
            assert prediction.stimulus_code is not None
            assert prediction.onset_seconds is not None
            events.append(
                FlashProbability(
                    sequence_index=index,
                    event_id=prediction.event_id,
                    stimulus_code=prediction.stimulus_code,
                    target_probability=prediction.target_probability,
                    onset_seconds=prediction.onset_seconds,
                )
            )
        output.append(
            FlashProbabilityTrial(
                selection_trial_id=selection_trial_id,
                subject_id=next(iter(subject_ids)),
                session_id=next(iter(session_ids)),
                events=tuple(events),
                recorded_target_codes=target_codes,
            )
        )
    if not output:
        raise ValueError("decoder evaluation contains no labeled timed trials for replay")
    return tuple(output)


@dataclass(frozen=True)
class _PreparedReplay:
    trial: CounterfactualFusionTrial
    evidence: NeuralSelectionEvidence
    provenance: CounterfactualTrialProvenance


PERSONALIZED_CONDITIONS = frozenset(
    {
        EvaluationCondition.D_NEURAL_PERSONALIZED,
        EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
        EvaluationCondition.F_COMPLETE_SYSTEM,
        EvaluationCondition.ABLATION_UNIFORM_NEURAL,
        EvaluationCondition.ABLATION_SHUFFLED_NEURAL,
        EvaluationCondition.ABLATION_REMOVE_RAG,
        EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL,
        EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL,
        EvaluationCondition.ABLATION_REMOVE_CONTEXT,
        EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
    }
)

SAFE_FUSION_CONDITIONS = frozenset(
    {
        EvaluationCondition.F_COMPLETE_SYSTEM,
        EvaluationCondition.ABLATION_UNIFORM_NEURAL,
        EvaluationCondition.ABLATION_SHUFFLED_NEURAL,
        EvaluationCondition.ABLATION_REMOVE_RAG,
        EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL,
        EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL,
        EvaluationCondition.ABLATION_REMOVE_CONTEXT,
        EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
    }
)


class CounterfactualFusionRunner:
    """Run every condition on paired candidate grids and recorded flash streams."""

    def __init__(self, experiment_input: CounterfactualExperimentInput) -> None:
        self.input = experiment_input
        self.spec = experiment_input.spec
        self.ranker = TransparentRanker()

    def run(self) -> CounterfactualFusionResult:
        self._validate_dependencies()
        prepared = tuple(self._prepare(trial) for trial in self.input.trials)
        records = tuple(
            self._evaluate(prepared, index, condition)
            for index in range(len(prepared))
            for condition in self.spec.conditions
        )
        metrics = calculate_metrics(
            records,
            self.spec.conditions,
            self.spec.calibration_bins,
        )
        intervals = self._paired_intervals(records)
        input_sha = self.input.digest()
        run_material = f"{self.spec.digest()}:{input_sha}"
        controlled = self.spec.personalization_evidence_kind == "controlled_fixture"
        limitations = [
            "Offline counterfactual replay remaps a recorded target position to a visible tile; "
            "it is not an original participant word selection.",
            "Original-task decoder metrics, counterfactual fusion metrics, and controlled "
            "simulation metrics must remain in separate result tables.",
            "Intervals are descriptive hierarchical bootstrap summaries and do not by themselves "
            "establish non-inferiority or clinical utility.",
        ]
        if controlled:
            limitations.append(
                "Personalization lift came from a controlled fixture rather than a held-out LoRA; "
                "conditions D-F are executable mechanics checks but are not claim-eligible."
            )
        adapters = {
            trial.personalization_adapter_id: trial.personalization_adapter_sha256
            for trial in self.input.trials
            if trial.personalization_adapter_id is not None
            and trial.personalization_adapter_sha256 is not None
        }
        return CounterfactualFusionResult(
            run_id=f"counterfactual-fusion-{hashlib.sha256(run_material.encode()).hexdigest()[:20]}",
            generated_at=self.input.prepared_at,
            config_sha256=self.spec.digest(),
            input_sha256=input_sha,
            source_decoder_manifest_sha256=self.input.source_decoder_manifest_sha256,
            original_task_evaluation_sha256=self.input.original_task_evaluation_sha256,
            personalization_adapters=adapters,
            spec=self.spec,
            mapping_provenance=tuple(item.provenance for item in prepared),
            trial_records=records,
            metrics=metrics,
            paired_intervals=intervals,
            claim_eligible=not controlled,
            limitations=tuple(limitations),
        )

    def _validate_dependencies(self) -> None:
        conditions = set(self.spec.conditions)
        if conditions & PERSONALIZED_CONDITIONS:
            missing = [
                trial.trial_id
                for trial in self.input.trials
                if trial.personalization_adapter_id is None
                or trial.personalization_adapter_sha256 is None
                or not trial.personalization_lift
            ]
            if missing:
                raise CounterfactualConfigurationError(
                    f"personalized conditions require adapter provenance and lift: {missing[:3]}"
                )
        if EvaluationCondition.ABLATION_REMOVE_CONTEXT in conditions:
            missing = [
                trial.trial_id
                for trial in self.input.trials
                if trial.no_context_language_support is None
                or trial.no_context_retrieval_evidence is None
            ]
            if missing:
                raise CounterfactualConfigurationError(
                    "context ablation requires no-context language and retrieval evidence: "
                    f"{missing[:3]}"
                )
        if EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT in conditions:
            missing = [
                trial.trial_id
                for trial in self.input.trials
                if trial.no_context_retrieval_evidence is None
            ]
            if missing:
                raise CounterfactualConfigurationError(
                    "retrieval-context ablation requires a no-context retrieval snapshot: "
                    f"{missing[:3]}"
                )
        if EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL in conditions:
            missing = [
                trial.trial_id
                for trial in self.input.trials
                if trial.irrelevant_retrieval_evidence is None
            ]
            if missing:
                raise CounterfactualConfigurationError(
                    "irrelevant-RAG ablation requires an explicit irrelevant snapshot: "
                    f"{missing[:3]}"
                )

    def _prepare(self, trial: CounterfactualFusionTrial) -> _PreparedReplay:
        mapped_layout = remap_recorded_target(
            trial.flash_layout,
            target_candidate_id=trial.resolved_target_candidate_id,
            recorded_target_codes=trial.flash_trial.recorded_target_codes,
        )
        evidence = aggregate_flash_probabilities(
            trial.flash_trial,
            mapped_layout,
            calibration_id="counterfactual-event-calibration-v1",
            recorded_at=self.input.prepared_at,
            config=self.spec.aggregation,
        )
        return _PreparedReplay(
            trial=trial,
            evidence=evidence,
            provenance=CounterfactualTrialProvenance(
                source_trial_id=trial.flash_trial.selection_trial_id,
                subject_id=trial.flash_trial.subject_id,
                session_id=trial.flash_trial.session_id,
                event_ids=tuple(item.event_id for item in trial.flash_trial.events),
                event_onsets_seconds=tuple(item.onset_seconds for item in trial.flash_trial.events),
                recorded_target_codes=trial.flash_trial.recorded_target_codes,
                mapped_target_candidate_id=trial.resolved_target_candidate_id,
                intended_candidate_was_absent=trial.intended_candidate_id is None,
                source_layout_sha256=trial.flash_layout.digest(),
                mapped_layout_sha256=mapped_layout.digest(),
                neural_evidence_id=evidence.evidence_id,
            ),
        )

    def _evaluate(
        self,
        prepared: tuple[_PreparedReplay, ...],
        index: int,
        condition: EvaluationCondition,
    ) -> TrialRecord:
        item = prepared[index]
        trial = item.trial
        evidence = self._condition_evidence(prepared, index, condition)
        language = (
            trial.no_context_language_support
            if condition is EvaluationCondition.ABLATION_REMOVE_CONTEXT
            else trial.generic_language_support
        )
        assert language is not None
        personalization = trial.personalization_lift if condition in PERSONALIZED_CONDITIONS else {}
        retrieval = self._condition_retrieval(trial, condition)
        candidate_ids = tuple(item.candidate_id for item in trial.candidate_set.candidates)
        language_top_id = min(
            language,
            key=lambda candidate_id: (-language[candidate_id], candidate_ids.index(candidate_id)),
        )
        neural_top_id = (
            max(evidence.candidate_probabilities, key=evidence.candidate_probabilities.__getitem__)
            if evidence is not None
            else None
        )
        if condition in SAFE_FUSION_CONDITIONS:
            assert evidence is not None
            ranking = self.ranker.rank(
                RankingInputs(
                    candidate_set=trial.candidate_set,
                    neural_evidence=evidence,
                    generic_language_support=language,
                    personalization_lift=personalization,
                    retrieval_evidence=retrieval,
                )
            )
            ranked_ids = tuple(value.candidate.candidate_id for value in ranking.ranked_candidates)
            disposition = ranking.disposition
            reason_codes = tuple(reason.value for reason in ranking.reason_codes)
            target_item = next(
                value
                for value in ranking.ranked_candidates
                if value.candidate.candidate_id == trial.resolved_target_candidate_id
            )
            enhanced = (
                target_item.confirmation_level is ConfirmationLevel.ENHANCED
                or ranking.fused_top_candidate_id != trial.resolved_target_candidate_id
            )
        else:
            ranked_ids = self._baseline_ranking(
                trial=trial,
                condition=condition,
                evidence=evidence,
                language=language,
                personalization=personalization,
                retrieval=retrieval,
            )
            disposition = RankingDisposition.DISPLAY
            reason_codes = ("counterfactual_baseline_without_abstention",)
            enhanced = ranked_ids[0] != trial.resolved_target_candidate_id
        target_rank = ranked_ids.index(trial.resolved_target_candidate_id) + 1
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
        calibration = self._calibration(evidence, trial.resolved_target_candidate_id)
        target_candidate = next(
            candidate
            for candidate in trial.candidate_set.candidates
            if candidate.candidate_id == trial.resolved_target_candidate_id
        )
        record_material = f"{condition.value}:{trial.trial_id}"
        return TrialRecord(
            trial_id=f"replay-{hashlib.sha256(record_material.encode()).hexdigest()[:20]}",
            condition=condition,
            profile_id=trial.flash_trial.subject_id,
            message_id=trial.trial_id,
            span_index=0,
            message_span_count=1,
            confirmed_context=trial.confirmed_context,
            retrieval_query_context_removed=condition
            in {
                EvaluationCondition.ABLATION_REMOVE_CONTEXT,
                EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
            },
            target_text=target_candidate.text,
            target_word_count=max(1, len(target_candidate.text.split())),
            candidate_ids=candidate_ids,
            ranked_candidate_ids=ranked_ids,
            target_candidate_id=trial.resolved_target_candidate_id,
            target_rank=target_rank,
            top_candidate_id=ranked_ids[0],
            neural_top_candidate_id=neural_top_id,
            language_top_candidate_id=language_top_id,
            disposition=disposition,
            reason_codes=reason_codes,
            language_conflict_context=(
                neural_top_id is not None and neural_top_id != language_top_id
            ),
            neural_language_conflict=(
                neural_top_id is not None and neural_top_id != language_top_id
            ),
            neural_target_probability=(
                evidence.candidate_probabilities[trial.resolved_target_candidate_id]
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
            retrieval_hit_count=sum(len(value.hits) for value in retrieval),
            modeled_duration_seconds=duration,
        )

    def _condition_evidence(
        self,
        prepared: tuple[_PreparedReplay, ...],
        index: int,
        condition: EvaluationCondition,
    ) -> NeuralSelectionEvidence | None:
        if condition is EvaluationCondition.B_GENERIC_LANGUAGE_ONLY:
            return None
        evidence = prepared[index].evidence
        candidate_ids = tuple(evidence.candidate_probabilities)
        if condition is EvaluationCondition.ABLATION_UNIFORM_NEURAL:
            probabilities = dict.fromkeys(candidate_ids, 1.0 / len(candidate_ids))
            suffix = "uniform"
        elif condition is EvaluationCondition.ABLATION_SHUFFLED_NEURAL:
            donor = prepared[(index + 1) % len(prepared)].evidence
            donor_values = tuple(donor.candidate_probabilities.values())
            if len(donor_values) != len(candidate_ids):
                raise CounterfactualConfigurationError(
                    "shuffled EEG ablation requires equal candidate counts"
                )
            probabilities = dict(zip(candidate_ids, donor_values, strict=True))
            suffix = f"shuffled-{donor.evidence_id}"
        else:
            return evidence
        ordered = sorted(probabilities.values(), reverse=True)
        entropy = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
        return evidence.model_copy(
            update={
                "evidence_id": f"{evidence.evidence_id}-{suffix}",
                "candidate_probabilities": probabilities,
                "entropy": entropy,
                "top_margin": ordered[0] - ordered[1],
            }
        )

    def _condition_retrieval(
        self,
        trial: CounterfactualFusionTrial,
        condition: EvaluationCondition,
    ) -> tuple[CandidateRetrievalEvidence, ...]:
        if condition in {
            EvaluationCondition.A_BCI_ONLY,
            EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
            EvaluationCondition.C_NEURAL_LANGUAGE,
            EvaluationCondition.D_NEURAL_PERSONALIZED,
            EvaluationCondition.ABLATION_REMOVE_RAG,
        }:
            return ()
        if condition is EvaluationCondition.ABLATION_REMOVE_CONTEXT:
            assert trial.no_context_retrieval_evidence is not None
            return trial.no_context_retrieval_evidence
        if condition is EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT:
            assert trial.no_context_retrieval_evidence is not None
            return trial.no_context_retrieval_evidence
        if condition is EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL:
            assert trial.irrelevant_retrieval_evidence is not None
            return trial.irrelevant_retrieval_evidence
        if condition is EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL:
            return shuffle_retrieval_across_candidates(trial)
        return trial.retrieval_evidence

    def _baseline_ranking(
        self,
        *,
        trial: CounterfactualFusionTrial,
        condition: EvaluationCondition,
        evidence: NeuralSelectionEvidence | None,
        language: dict[str, float],
        personalization: dict[str, float],
        retrieval: tuple[CandidateRetrievalEvidence, ...],
    ) -> tuple[str, ...]:
        original_order = {
            candidate.candidate_id: index
            for index, candidate in enumerate(trial.candidate_set.candidates)
        }
        retrieval_support = {value.candidate_id: value.retrieval_support for value in retrieval}
        weights = self.ranker.policy.weights
        scores: dict[str, float] = {}
        for candidate in trial.candidate_set.candidates:
            candidate_id = candidate.candidate_id
            neural = (
                evidence.candidate_probabilities.get(candidate_id, 0.0)
                if evidence is not None
                else 0.0
            )
            generic = language.get(candidate_id, 0.0)
            if condition is EvaluationCondition.A_BCI_ONLY:
                score = neural
            elif condition is EvaluationCondition.B_GENERIC_LANGUAGE_ONLY:
                score = generic
            else:
                score = weights.neural * neural + weights.generic_language * generic
                if condition in {
                    EvaluationCondition.D_NEURAL_PERSONALIZED,
                    EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
                }:
                    score += weights.personalization * personalization.get(candidate_id, 0.0)
                if condition is EvaluationCondition.E_NEURAL_PERSONALIZED_RAG:
                    score += weights.retrieval * retrieval_support.get(candidate_id, 0.0)
            scores[candidate_id] = score
        return tuple(
            candidate_id
            for candidate_id, _ in sorted(
                scores.items(), key=lambda value: (-value[1], original_order[value[0]])
            )
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

    def _paired_intervals(
        self,
        records: tuple[TrialRecord, ...],
    ) -> tuple[PairedBootstrapInterval, ...]:
        if EvaluationCondition.F_COMPLETE_SYSTEM not in self.spec.conditions:
            return ()
        output: list[PairedBootstrapInterval] = []
        for condition in self.spec.conditions:
            if condition is EvaluationCondition.F_COMPLETE_SYSTEM:
                continue
            for metric in ("top_1_candidate_recall", "selection_completion_rate"):
                output.append(self._bootstrap(records, condition, metric))
        return tuple(output)

    def _bootstrap(
        self,
        records: tuple[TrialRecord, ...],
        condition: EvaluationCondition,
        metric: Literal["top_1_candidate_recall", "selection_completion_rate"],
    ) -> PairedBootstrapInterval:
        by_key = {(record.condition, record.message_id): record for record in records}
        reference_records = tuple(
            record
            for record in records
            if record.condition is EvaluationCondition.F_COMPLETE_SYSTEM
        )
        subject_trials: dict[str, list[str]] = {}
        for record in reference_records:
            if (condition, record.message_id) not in by_key:
                raise CounterfactualConfigurationError("paired condition is missing a replay trial")
            subject_trials.setdefault(record.profile_id, []).append(record.message_id)
        subjects = tuple(sorted(subject_trials))
        material = f"{self.spec.seed}:{condition.value}:{metric}"
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
        generator = np.random.default_rng(seed)

        def value(record: TrialRecord) -> float:
            if metric == "top_1_candidate_recall":
                return float(record.top_1_correct)
            return float(record.explicit_selection_completed)

        observed_pairs = [
            value(by_key[(condition, record.message_id)]) - value(record)
            for record in reference_records
        ]
        samples = np.empty(self.spec.bootstrap_resamples, dtype=np.float64)
        for iteration in range(self.spec.bootstrap_resamples):
            deltas: list[float] = []
            for subject_index in generator.integers(0, len(subjects), size=len(subjects)):
                subject = subjects[int(subject_index)]
                trial_ids = subject_trials[subject]
                for trial_index in generator.integers(0, len(trial_ids), size=len(trial_ids)):
                    trial_id = trial_ids[int(trial_index)]
                    deltas.append(
                        value(by_key[(condition, trial_id)])
                        - value(by_key[(EvaluationCondition.F_COMPLETE_SYSTEM, trial_id)])
                    )
            samples[iteration] = float(np.mean(deltas))
        alpha = (1.0 - self.spec.confidence_level) / 2.0
        return PairedBootstrapInterval(
            condition=condition,
            metric=metric,
            observed_delta=float(np.mean(observed_pairs)),
            lower_bound=float(np.quantile(samples, alpha)),
            upper_bound=float(np.quantile(samples, 1.0 - alpha)),
            confidence_level=self.spec.confidence_level,
            resamples=self.spec.bootstrap_resamples,
        )
