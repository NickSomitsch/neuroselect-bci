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
    message_records: dict[str, list[TrialRecord]] = defaultdict(list)
    for record in records:
        message_records[record.message_id].append(record)
    completed_message_ids = {
        message_id
        for message_id, values in message_records.items()
        if all(record.explicit_selection_completed for record in values)
    }
    completed_records = tuple(record for record in records if record.explicit_selection_completed)
    total_duration = sum(record.modeled_duration_seconds for record in records)
    completed_actions = sum(
        record.explicit_action_count
        for record in records
        if record.message_id in completed_message_ids
    )
    conflict_records = tuple(record for record in records if record.language_conflict_context)
    brier_values = tuple(
        record.neural_brier_score for record in records if record.neural_brier_score is not None
    )

    return ConditionMetrics(
        condition=condition,
        profile_id=profile_id,
        trial_count=len(records),
        message_count=len(message_records),
        completed_trial_count=len(completed_records),
        completed_message_count=len(completed_message_ids),
        target_availability_rate=_rate(record.target_available for record in records),
        top_1_candidate_recall=_rate(record.top_1_correct for record in records),
        top_3_candidate_recall=_rate(record.top_3_correct for record in records),
        selection_completion_rate=len(completed_records) / len(records),
        final_message_exact_accuracy=len(completed_message_ids) / len(message_records),
        correct_selections_per_minute=60.0 * len(completed_records) / total_duration,
        words_per_minute=(
            60.0 * sum(record.target_word_count for record in completed_records) / total_duration
        ),
        selections_per_completed_message=(
            completed_actions / len(completed_message_ids) if completed_message_ids else None
        ),
        unintended_word_rate=_rate(record.unintended_word for record in records),
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
