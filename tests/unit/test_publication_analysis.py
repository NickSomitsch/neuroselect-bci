from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from neuroselect.decoding import BinaryDecoderMetrics, DecoderEvaluation, EpochPrediction
from neuroselect.eeg import P300Label
from neuroselect.evaluation.models import EvaluationCondition
from neuroselect.publication.analysis import (
    PublicationAnalysisResult,
    PublicationAnalysisSpec,
    PublicationEstimate,
    PublicationInterval,
    _condition_mean,
    _counterfactual_analysis,
    _counterfactual_sample,
    _counterfactual_value,
    _language_analysis,
    _language_metrics,
    _p300_analysis,
    _selection_metric_values,
    load_publication_analysis_spec,
    read_publication_analysis,
    write_publication_analysis,
)
from neuroselect.ranking import RankingDisposition

NOW = datetime(2026, 7, 27, 15, 0, tzinfo=UTC)


def analysis_spec(**updates: object) -> PublicationAnalysisSpec:
    values: dict[str, object] = {
        "analysis_id": "test-analysis",
        "analysis_revision": "offline-primary-analysis-v1",
        "analyzed_at": NOW,
        "publication_protocol": Path("protocol.yaml"),
        "eegnet_artifacts": Path("eegnet"),
        "eegnet_required": False,
        "confidence_level": 0.95,
        "bootstrap_resamples": 2_000,
        "bootstrap_seed": 7,
    }
    values.update(updates)
    return PublicationAnalysisSpec.model_validate(values)


def test_publication_analysis_config_is_strict_and_tracked() -> None:
    tracked = load_publication_analysis_spec()
    assert tracked.bootstrap_resamples == 10_000
    assert tracked.eegnet_required is True
    assert tracked.expected_eegnet_manifest_sha256 is not None
    with pytest.raises(ValidationError, match="pin its manifest"):
        analysis_spec(eegnet_required=True)
    with pytest.raises(ValidationError, match="include a timezone"):
        analysis_spec(analyzed_at=datetime(2026, 7, 27))


def test_publication_analysis_rejects_invalid_boundaries_and_inputs(tmp_path: Path) -> None:
    invalid_config = tmp_path / "analysis.yaml"
    invalid_config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_publication_analysis_spec(invalid_config)
    with pytest.raises(ValidationError, match="bounds must be ordered"):
        PublicationInterval(
            component="language",
            scope="overall",
            contrast="rate",
            metric="availability",
            estimate=0.5,
            lower_bound=0.6,
            upper_bound=0.4,
            confidence_level=0.95,
            resamples=2_000,
            sampling_unit="message",
        )
    with pytest.raises(ValueError, match="require trials"):
        _language_metrics([])

    unavailable = _language_metrics(
        [_language_trial("synthetic-a", "a1", 0, generic_rank=None, personalized_rank=None)]
    )
    assert unavailable == {
        "target_availability_rate": 0.0,
        "message_target_availability_rate": 0.0,
        "generic_top1_unconditional": 0.0,
        "personalized_top1_unconditional": 0.0,
        "generic_top3_unconditional": 0.0,
        "personalized_top3_unconditional": 0.0,
    }
    with pytest.raises(ValueError, match="unsupported counterfactual"):
        _counterfactual_value(cast(Any, SimpleNamespace()), "not-a-metric")
    missing_subject = cast(
        Any,
        [
            SimpleNamespace(
                eeg_subject_id=None,
                profile_id="synthetic-a",
                message_id="message-a",
            )
        ],
    )
    with pytest.raises(ValueError, match="EEG subject provenance"):
        _counterfactual_sample(missing_subject, np.random.default_rng(1))
    with pytest.raises(ValueError, match="condition is absent"):
        _condition_mean([], EvaluationCondition.A_BCI_ONLY, "top_1_candidate_recall")
    with pytest.raises(ValueError, match="scorable trials"):
        _selection_metric_values([])


def _language_trial(
    profile: str,
    message: str,
    span: int,
    *,
    generic_rank: int | None,
    personalized_rank: int | None,
) -> Any:
    return SimpleNamespace(
        profile_id=profile,
        message_id=message,
        span_index=span,
        intended_text="two words",
        target_available=generic_rank is not None,
        generic_rank=generic_rank,
        personalized_rank=personalized_rank,
    )


def test_language_analysis_clusters_messages_and_reports_span_zero() -> None:
    trials = cast(
        Any,
        [
            _language_trial("synthetic-a", "a1", 0, generic_rank=None, personalized_rank=None),
            _language_trial("synthetic-a", "a1", 1, generic_rank=2, personalized_rank=1),
            _language_trial("synthetic-a", "a2", 0, generic_rank=None, personalized_rank=None),
            _language_trial("synthetic-a", "a2", 1, generic_rank=3, personalized_rank=2),
            _language_trial("synthetic-b", "b1", 0, generic_rank=None, personalized_rank=None),
            _language_trial("synthetic-b", "b1", 1, generic_rank=1, personalized_rank=1),
            _language_trial("synthetic-b", "b2", 0, generic_rank=None, personalized_rank=None),
            _language_trial("synthetic-b", "b2", 1, generic_rank=2, personalized_rank=1),
        ],
    )

    estimates, intervals = _language_analysis(trials, analysis_spec())

    span_zero = next(
        item
        for item in estimates
        if item.scope == "span-0" and item.metric == "target_availability_rate"
    )
    mrr = next(
        item
        for item in intervals
        if item.scope == "overall" and item.metric == "mrr_delta_given_available"
    )
    assert span_zero.estimate == 0.0
    assert mrr.estimate > 0.0
    assert mrr.sampling_unit == "messages_within_fixed_profile_strata"


def _evaluation(offset: float) -> DecoderEvaluation:
    predictions: list[EpochPrediction] = []
    for subject_index, subject in enumerate(("P_02", "P_11", "P_13")):
        for trial_index in range(2):
            for event_index in range(4):
                target = event_index == 0
                probability = (0.8 if target else 0.2) + offset
                predictions.append(
                    EpochPrediction(
                        epoch_id=f"{subject}-{trial_index}-{event_index}",
                        event_id=f"event-{subject}-{trial_index}-{event_index}",
                        selection_trial_id=f"{subject}-trial-{trial_index}",
                        recording_id=f"{subject}:SE001:fixture",
                        subject_id=subject,
                        session_id="SE001",
                        true_label=(P300Label.TARGET if target else P300Label.NON_TARGET),
                        target_probability=probability,
                        predicted_target=probability >= 0.5,
                        onset_seconds=float(subject_index + trial_index + event_index),
                        stimulus_code=event_index,
                    )
                )
    return DecoderEvaluation(
        dataset_sha256="d" * 64,
        predictions=tuple(predictions),
        labeled_epoch_count=len(predictions),
        unknown_epoch_count=0,
        metrics=BinaryDecoderMetrics(
            auroc=1.0,
            balanced_accuracy=1.0,
            brier_score=0.04,
            negative_log_likelihood=0.1,
            expected_calibration_error=0.2,
        ),
        selection_trial_count=6,
        selection_code_set_accuracy=1.0,
    )


def test_p300_analysis_adds_target_event_metrics_and_paired_comparison() -> None:
    estimates, intervals = _p300_analysis(
        {"xdawn": _evaluation(0.0), "eegnet": _evaluation(-0.05)},
        analysis_spec(),
    )

    assert any(item.metric == "target_event_average_precision" for item in estimates)
    paired = [item for item in intervals if item.contrast == "eegnet-minus-xdawn"]
    assert {item.metric for item in paired} == {
        "brier_score",
        "exact_target_event_set_accuracy",
        "target_event_recall_at_k",
        "target_event_average_precision",
        "top_event_hit_rate",
    }
    assert all(
        item.sampling_unit == "paired_held_out_subject_then_selection_trial" for item in paired
    )


def _counterfactual_records() -> Any:
    records = []
    conditions = (
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
        EvaluationCondition.C_NEURAL_LANGUAGE,
        EvaluationCondition.D_NEURAL_PERSONALIZED,
        EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
        EvaluationCondition.F_COMPLETE_SYSTEM,
    )
    for subject in ("P_02", "P_11", "P_13"):
        for span in range(4):
            for condition in conditions:
                top1 = condition in {
                    EvaluationCondition.A_BCI_ONLY,
                    EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
                    EvaluationCondition.F_COMPLETE_SYSTEM,
                }
                records.append(
                    SimpleNamespace(
                        eeg_subject_id=subject,
                        profile_id="synthetic-a",
                        message_id=f"{subject}-message",
                        span_index=span,
                        condition=condition,
                        top_1_correct=top1,
                        explicit_selection_completed=top1,
                        target_available=True,
                        disposition=RankingDisposition.DISPLAY,
                    )
                )
    return cast(Any, records)


def test_counterfactual_analysis_resamples_complete_messages() -> None:
    _, intervals = _counterfactual_analysis(_counterfactual_records(), analysis_spec())

    complete_vs_bci = next(
        item
        for item in intervals
        if item.contrast == "f_complete_system-minus-a_bci_only"
        and item.metric == "selection_completion_rate"
    )
    assert complete_vs_bci.estimate == 0.0
    assert complete_vs_bci.sampling_unit == "held_out_subject_then_complete_message"


def test_publication_analysis_artifact_round_trip_and_tampering(tmp_path: Path) -> None:
    spec = analysis_spec()
    result = PublicationAnalysisResult(
        analysis_id=spec.analysis_id,
        analyzed_at=spec.analyzed_at,
        config_sha256=spec.digest(),
        protocol_sha256="b" * 64,
        source_manifest_sha256={"source": "c" * 64},
        estimates=(
            PublicationEstimate(
                component="language",
                scope="overall",
                variant="observed",
                metric="availability",
                sample_count=10,
                estimate=0.5,
            ),
        ),
        intervals=(
            PublicationInterval(
                component="language",
                scope="overall",
                contrast="rate",
                metric="availability",
                estimate=0.5,
                lower_bound=0.4,
                upper_bound=0.6,
                confidence_level=0.95,
                resamples=2_000,
                sampling_unit="message",
            ),
        ),
        eegnet_included=False,
        limitations=("fixture",),
    )
    manifest = write_publication_analysis(
        result,
        spec,
        tmp_path,
        git_sha="b239179",
        source_tree_sha256=None,
    )

    restored, restored_manifest = read_publication_analysis(tmp_path)

    assert restored == result
    assert restored_manifest == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_publication_analysis(
            result,
            spec,
            tmp_path,
            git_sha="b239179",
            source_tree_sha256=None,
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest.model_copy(update={"config_sha256": "f" * 64}).canonical_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not agree"):
        read_publication_analysis(tmp_path)
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    (tmp_path / "intervals.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_publication_analysis(tmp_path)
