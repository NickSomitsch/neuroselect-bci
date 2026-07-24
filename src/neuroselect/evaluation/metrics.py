"""Deterministic aggregate metrics for simulated evaluation records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from neuroselect.evaluation.models import ConditionMetrics, EvaluationCondition, TrialRecord
from neuroselect.ranking import RankingDisposition


def expected_calibration_error(records: tuple[TrialRecord, ...], bins: int) -> float | None:
    """Compute top-label ECE over records that contain neural probabilities."""

    calibrated = tuple(record for record in records if record.prediction_confidence is not None)
    if not calibrated:
        return None
    grouped: dict[int, list[TrialRecord]] = defaultdict(list)
    for record in calibrated:
        assert record.prediction_confidence is not None
        index = min(int(record.prediction_confidence * bins), bins - 1)
        grouped[index].append(record)

    total = len(calibrated)
    error = 0.0
    for values in grouped.values():
        confidence = sum(record.prediction_confidence or 0.0 for record in values) / len(values)
        accuracy = sum(bool(record.prediction_correct) for record in values) / len(values)
        error += (len(values) / total) * abs(accuracy - confidence)
    return error


def _rate(values: Iterable[bool]) -> float:
    items = tuple(values)
    return sum(items) / len(items)


def _one_slice(
    *,
    condition: EvaluationCondition,
    records: tuple[TrialRecord, ...],
    calibration_bins: int,
    profile_id: str | None,
) -> ConditionMetrics:
    message_records: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        message_records[(record.profile_id, record.message_id)].append(record)
    for message_key, values in message_records.items():
        expected_counts = {record.message_span_count for record in values}
        indices = {record.span_index for record in values}
        if len(expected_counts) != 1:
            raise ValueError(f"message has inconsistent span counts: {message_key}")
        expected_count = next(iter(expected_counts))
        if len(values) != expected_count or indices != set(range(expected_count)):
            raise ValueError(f"message does not contain every expected span: {message_key}")
    completed_message_ids = frozenset(
        message_key
        for message_key, values in message_records.items()
        if all(record.explicit_selection_completed for record in values)
    )
    completed_records = tuple(record for record in records if record.explicit_selection_completed)
    available_records = tuple(record for record in records if record.target_available)
    fallback_records = tuple(record for record in records if not record.target_available)
    displayed_records = tuple(
        record for record in records if record.disposition is RankingDisposition.DISPLAY
    )
    selection_savings = tuple(
        record.selection_savings for record in records if record.selection_savings is not None
    )
    total_duration = sum(record.modeled_duration_seconds for record in records)
    completed_actions = sum(
        record.modeled_selection_count
        for record in records
        if (record.profile_id, record.message_id) in completed_message_ids
    )
    conflict_records = tuple(record for record in records if record.language_conflict_context)
    brier_values = tuple(
        record.neural_brier_score for record in records if record.neural_brier_score is not None
    )
    correct_displays = sum(
        record.top_1_correct if record.target_available else record.fallback_selected
        for record in displayed_records
    )
    display_accuracy = correct_displays / len(displayed_records) if displayed_records else None

    return ConditionMetrics(
        condition=condition,
        profile_id=profile_id,
        trial_count=len(records),
        message_count=len(message_records),
        completed_trial_count=len(completed_records),
        completed_message_count=len(completed_message_ids),
        available_trial_count=len(available_records),
        fallback_trial_count=len(fallback_records),
        displayed_trial_count=len(displayed_records),
        candidate_generation_failure_count=sum(
            record.candidate_generation_failed for record in records
        ),
        target_availability_rate=_rate(record.target_available for record in records),
        top_1_candidate_recall=_rate(record.top_1_correct for record in records),
        top_3_candidate_recall=_rate(record.top_3_correct for record in records),
        top_1_recall_given_available=(
            _rate(record.top_1_correct for record in available_records)
            if available_records
            else None
        ),
        top_3_recall_given_available=(
            _rate(record.top_3_correct for record in available_records)
            if available_records
            else None
        ),
        other_fallback_success_rate=(
            _rate(record.fallback_selection_completed for record in fallback_records)
            if fallback_records
            else None
        ),
        selection_completion_rate=len(completed_records) / len(records),
        final_message_exact_accuracy=len(completed_message_ids) / len(message_records),
        correct_selections_per_minute=60.0 * len(completed_records) / total_duration,
        words_per_minute=(
            60.0 * sum(record.target_word_count for record in completed_records) / total_duration
        ),
        selections_per_completed_message=(
            completed_actions / len(completed_message_ids) if completed_message_ids else None
        ),
        mean_selection_savings=(
            sum(selection_savings) / len(selection_savings) if selection_savings else None
        ),
        unintended_word_rate=_rate(record.unintended_word for record in records),
        incorrect_display_rate=_rate(record.incorrect_display for record in records),
        candidate_generation_failure_rate=_rate(
            record.candidate_generation_failed for record in records
        ),
        display_accuracy=display_accuracy,
        selective_risk=(1.0 - display_accuracy if display_accuracy is not None else None),
        correction_rate=_rate(record.correction_required for record in records),
        abstention_rate=_rate(
            record.disposition is RankingDisposition.ABSTAIN for record in records
        ),
        repeat_request_rate=_rate(
            record.disposition is RankingDisposition.REQUEST_REPEAT for record in records
        ),
        neural_expected_calibration_error=expected_calibration_error(records, calibration_bins),
        neural_multiclass_brier_score=(
            sum(brier_values) / len(brier_values) if brier_values else None
        ),
        mean_modeled_latency_seconds=total_duration / len(records),
        conflict_trial_count=len(conflict_records),
        conflict_top_1_recall=(
            _rate(record.top_1_correct for record in conflict_records) if conflict_records else None
        ),
        conflict_target_availability_rate=(
            _rate(record.target_available for record in conflict_records)
            if conflict_records
            else None
        ),
    )


def calculate_metrics(
    records: tuple[TrialRecord, ...],
    conditions: tuple[EvaluationCondition, ...],
    calibration_bins: int,
) -> tuple[ConditionMetrics, ...]:
    """Calculate an overall and per-profile slice in stable condition order."""

    output: list[ConditionMetrics] = []
    for condition in conditions:
        condition_records = tuple(record for record in records if record.condition is condition)
        if not condition_records:
            raise ValueError(f"condition has no trial records: {condition.value}")
        output.append(
            _one_slice(
                condition=condition,
                records=condition_records,
                calibration_bins=calibration_bins,
                profile_id=None,
            )
        )
        for profile_id in sorted({record.profile_id for record in condition_records}):
            output.append(
                _one_slice(
                    condition=condition,
                    records=tuple(
                        record for record in condition_records if record.profile_id == profile_id
                    ),
                    calibration_bins=calibration_bins,
                    profile_id=profile_id,
                )
            )
    return tuple(output)
