from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.bci import (
    FlashLayout,
    FlashProbability,
    FlashProbabilityTrial,
    TileAggregationConfig,
    aggregate_flash_probabilities,
    remap_recorded_target,
)
from neuroselect.core.models import Candidate, CandidateKind, CandidateSet, KnowledgeKind
from neuroselect.decoding import (
    BinaryDecoderMetrics,
    DecoderEvaluation,
    EpochPrediction,
)
from neuroselect.eeg import P300Label
from neuroselect.evaluation import (
    CounterfactualConfigurationError,
    CounterfactualExperimentInput,
    CounterfactualFusionRunner,
    CounterfactualFusionSpec,
    CounterfactualFusionTrial,
    EvaluationCondition,
    flash_trials_from_decoder_evaluation,
    load_counterfactual_spec,
    read_counterfactual_artifacts,
    shuffle_retrieval_across_candidates,
    write_counterfactual_artifacts,
)
from neuroselect.ranking import RankingDisposition
from neuroselect.retrieval import (
    CandidateRetrievalEvidence,
    RecordPermission,
    RetrievalHit,
    StoredKnowledgeRecord,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
ALL_CONDITIONS = (
    EvaluationCondition.A_BCI_ONLY,
    EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
    EvaluationCondition.C_NEURAL_LANGUAGE,
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
)


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(candidate_id="intended", text="open window", kind=CandidateKind.PHRASE),
        Candidate(candidate_id="likely", text="thank you", kind=CandidateKind.PHRASE),
        Candidate(candidate_id="alternative", text="some water", kind=CandidateKind.PHRASE),
        Candidate(candidate_id="other", text="Other", kind=CandidateKind.CONTROL),
        Candidate(candidate_id="back", text="Back", kind=CandidateKind.CONTROL),
        Candidate(candidate_id="cancel", text="Cancel", kind=CandidateKind.CONTROL),
    )


def candidate_set(ordinal: int) -> CandidateSet:
    return CandidateSet(
        candidate_set_id=f"candidate-set-{ordinal}",
        context_sha256=f"{ordinal + 1:064x}",
        candidates=candidates(),
        generator_revision="controlled-fixture-v1",
        prompt_revision="controlled-fixture-v1",
    )


def layout() -> FlashLayout:
    return FlashLayout(
        layout_id="two-by-three-source-grid",
        candidate_ids=tuple(item.candidate_id for item in candidates()),
        stimulus_codes=(1, 2, 3, 4, 5),
        candidate_code_sets={
            "intended": (1, 3),
            "likely": (1, 4),
            "alternative": (1, 5),
            "other": (2, 3),
            "back": (2, 4),
            "cancel": (2, 5),
        },
    )


def flash_trial(ordinal: int, subject_id: str) -> FlashProbabilityTrial:
    events: list[FlashProbability] = []
    index = 0
    for _ in range(2):
        for code in layout().stimulus_codes:
            events.append(
                FlashProbability(
                    sequence_index=index,
                    event_id=f"event-{ordinal}-{index}",
                    stimulus_code=code,
                    target_probability=0.9 if code in {1, 3} else 0.1,
                    onset_seconds=ordinal * 100 + index * 0.25,
                )
            )
            index += 1
    return FlashProbabilityTrial(
        selection_trial_id=f"source-trial-{ordinal}",
        subject_id=subject_id,
        session_id="SE002",
        events=tuple(events),
        recorded_target_codes=(1, 3),
    )


def stored_record(record_id: str, content: str) -> StoredKnowledgeRecord:
    return StoredKnowledgeRecord(
        record_id=record_id,
        kind=KnowledgeKind.PREFERENCE,
        content=content,
        source="fixture:counterfactual-test",
        permissions=frozenset({RecordPermission.SUGGEST, RecordPermission.EXPLAIN}),
        profile_id="P_03",
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        injection_risk=False,
    )


def retrieval(candidate_id: str, *, irrelevant: bool = False) -> CandidateRetrievalEvidence:
    record = stored_record(
        "irrelevant-record" if irrelevant else "window-preference",
        "The garden path is blue." if irrelevant else "Please open the window in the morning.",
    )
    hit = RetrievalHit(
        record=record,
        score=0.8,
        matched_terms=("controlled",),
        explanation="Explicit controlled retrieval fixture.",
    )
    return CandidateRetrievalEvidence(
        candidate_id=candidate_id,
        retrieval_support=0.8,
        record_ids=(record.record_id,),
        hits=(hit,),
    )


def fusion_trial(
    ordinal: int,
    subject_id: str,
    intended_candidate_id: str | None,
) -> CounterfactualFusionTrial:
    return CounterfactualFusionTrial(
        trial_id=f"prepared-{ordinal}",
        candidate_set=candidate_set(ordinal),
        flash_layout=layout(),
        flash_trial=flash_trial(ordinal, subject_id),
        intended_candidate_id=intended_candidate_id,
        other_candidate_id="other",
        confirmed_context="Could you",
        generic_language_support={"intended": 0.1, "likely": 0.8, "alternative": 0.1},
        no_context_language_support={
            "intended": 0.2,
            "likely": 0.6,
            "alternative": 0.2,
        },
        personalization_lift={"intended": 0.5, "likely": -0.2},
        personalization_adapter_id="controlled-profile-adapter-v1",
        personalization_adapter_sha256="d" * 64,
        retrieval_evidence=(retrieval("intended"),),
        no_context_retrieval_evidence=(),
        irrelevant_retrieval_evidence=(retrieval("likely", irrelevant=True),),
    )


def experiment_input(
    *,
    conditions: tuple[EvaluationCondition, ...] = ALL_CONDITIONS,
) -> CounterfactualExperimentInput:
    spec = CounterfactualFusionSpec(
        experiment_id="controlled-counterfactual-test",
        conditions=conditions,
        bootstrap_resamples=100,
        personalization_evidence_kind="controlled_fixture",
    )
    return CounterfactualExperimentInput(
        prepared_at=NOW,
        source_decoder_manifest_sha256="a" * 64,
        original_task_evaluation_sha256="b" * 64,
        spec=spec,
        trials=(
            fusion_trial(0, "P_03", "intended"),
            fusion_trial(1, "P_03", "likely"),
            fusion_trial(2, "P_04", "alternative"),
            fusion_trial(3, "P_04", None),
        ),
    )


def test_flash_aggregation_and_counterfactual_mapping_preserve_source_stream() -> None:
    source = flash_trial(0, "P_03")
    source_layout = layout()
    original = aggregate_flash_probabilities(
        source,
        source_layout,
        calibration_id="decoder-calibration-v1",
        recorded_at=NOW,
    )
    mapped_layout = remap_recorded_target(
        source_layout,
        target_candidate_id="likely",
        recorded_target_codes=source.recorded_target_codes,
    )
    mapped = aggregate_flash_probabilities(
        source,
        mapped_layout,
        calibration_id="decoder-calibration-v1",
        recorded_at=NOW,
    )

    assert (
        max(
            original.candidate_probabilities,
            key=original.candidate_probabilities.__getitem__,
        )
        == "intended"
    )
    assert (
        max(
            mapped.candidate_probabilities,
            key=mapped.candidate_probabilities.__getitem__,
        )
        == "likely"
    )
    assert tuple(item.event_id for item in source.events) == tuple(
        f"event-0-{index}" for index in range(10)
    )
    assert mapped.mode.value == "replay"
    assert mapped.top_margin is not None and mapped.top_margin > 0.9


def test_flash_aggregation_rejects_incomplete_or_invalid_streams() -> None:
    source = flash_trial(0, "P_03")
    incomplete = source.model_copy(update={"events": source.events[:-1]})
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_flash_probabilities(
            incomplete,
            layout(),
            calibration_id="decoder-calibration-v1",
            recorded_at=NOW,
            config=TileAggregationConfig(minimum_code_repetitions=2),
        )
    with pytest.raises(ValueError, match="absent"):
        remap_recorded_target(layout(), target_candidate_id="likely", recorded_target_codes=(1, 2))
    invalid_payload = source.model_dump(mode="json")
    invalid_payload["events"] = list(reversed(invalid_payload["events"]))
    with pytest.raises(ValidationError, match="chronological"):
        FlashProbabilityTrial.model_validate(invalid_payload)
    duplicate_payload = source.model_dump(mode="json")
    duplicate_payload["events"][1]["event_id"] = duplicate_payload["events"][0]["event_id"]
    with pytest.raises(ValidationError, match="unique"):
        FlashProbabilityTrial.model_validate(duplicate_payload)
    with pytest.raises(ValueError, match="timezone"):
        aggregate_flash_probabilities(
            source,
            layout(),
            calibration_id="decoder-calibration-v1",
            recorded_at=datetime(2026, 7, 19),
        )


def test_counterfactual_runner_executes_paired_matrix_and_marks_fixture_claims() -> None:
    result = CounterfactualFusionRunner(experiment_input()).run()

    assert len(result.mapping_provenance) == 4
    assert result.mapping_provenance[0].source_trial_id == "source-trial-0"
    assert len(result.trial_records) == 4 * len(ALL_CONDITIONS)
    assert len(result.paired_intervals) == (len(ALL_CONDITIONS) - 1) * 2
    assert result.claim_eligible is False
    assert result.personalization_adapters == {"controlled-profile-adapter-v1": "d" * 64}
    assert result.mapping_provenance[-1].mapped_target_candidate_id == "other"
    assert result.mapping_provenance[-1].intended_candidate_was_absent is True
    uniform = [
        record
        for record in result.trial_records
        if record.condition is EvaluationCondition.ABLATION_UNIFORM_NEURAL
    ]
    assert all(record.disposition is RankingDisposition.REQUEST_REPEAT for record in uniform)
    assert all(record.automatic_selection_permitted is False for record in result.trial_records)
    overall_conditions = {
        metric.condition for metric in result.metrics if metric.profile_id is None
    }
    assert overall_conditions == set(ALL_CONDITIONS)
    invalid_result = result.model_dump(mode="json")
    invalid_result["claim_eligible"] = True
    with pytest.raises(ValidationError, match="cannot be claim-eligible"):
        type(result).model_validate(invalid_result)


def test_shuffled_retrieval_moves_evidence_to_a_different_candidate() -> None:
    trial = fusion_trial(0, "P_03", "intended")

    shuffled = shuffle_retrieval_across_candidates(trial)

    assert len(shuffled) == 1
    assert shuffled[0].candidate_id == "likely"
    assert shuffled[0].hits == trial.retrieval_evidence[0].hits


def test_counterfactual_dependencies_are_explicitly_gated() -> None:
    missing_adapter = fusion_trial(0, "P_03", "intended").model_copy(
        update={
            "personalization_adapter_id": None,
            "personalization_adapter_sha256": None,
            "personalization_lift": {},
        }
    )
    source = experiment_input(conditions=(EvaluationCondition.F_COMPLETE_SYSTEM,))
    invalid = source.model_copy(update={"trials": (missing_adapter,)})
    with pytest.raises(CounterfactualConfigurationError, match="adapter provenance"):
        CounterfactualFusionRunner(invalid).run()

    missing_context = fusion_trial(0, "P_03", "intended").model_copy(
        update={"no_context_language_support": None}
    )
    context_source = experiment_input(
        conditions=(EvaluationCondition.ABLATION_REMOVE_CONTEXT,)
    ).model_copy(update={"trials": (missing_context,)})
    with pytest.raises(CounterfactualConfigurationError, match="context ablation"):
        CounterfactualFusionRunner(context_source).run()

    duplicate_source_payload = experiment_input().model_dump(mode="json")
    duplicate_source_payload["trials"][1]["flash_trial"]["selection_trial_id"] = (
        duplicate_source_payload["trials"][0]["flash_trial"]["selection_trial_id"]
    )
    with pytest.raises(ValidationError, match="mapped only once"):
        CounterfactualExperimentInput.model_validate(duplicate_source_payload)


def test_counterfactual_artifacts_round_trip_and_detect_tampering(tmp_path: Path) -> None:
    result = CounterfactualFusionRunner(experiment_input()).run()
    manifest = write_counterfactual_artifacts(
        result,
        tmp_path,
        git_sha="b4cfd22",
    )

    restored, restored_manifest = read_counterfactual_artifacts(tmp_path)
    assert restored == result
    assert restored_manifest == manifest
    assert manifest.run_kind.value == "counterfactual_replay"
    assert (
        (tmp_path / "condition-metrics.csv")
        .read_text(encoding="utf-8")
        .startswith("condition,trial_count")
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_counterfactual_artifacts(result, tmp_path, git_sha="b4cfd22")
    invalid_manifest = manifest.model_copy(
        update={
            "models": (
                manifest.models[0].model_copy(update={"sha256": "e" * 64}),
                *manifest.models[1:],
            )
        }
    )
    (tmp_path / "manifest.json").write_text(
        invalid_manifest.canonical_json() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not agree"):
        read_counterfactual_artifacts(tmp_path)
    (tmp_path / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    (tmp_path / "trials.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_counterfactual_artifacts(tmp_path)


def test_decoder_evaluation_extraction_keeps_only_labeled_timed_trials() -> None:
    predictions: list[EpochPrediction] = []
    for trial_index, label in (
        (0, P300Label.TARGET),
        (1, P300Label.UNKNOWN),
        (2, P300Label.TARGET),
    ):
        for index, code in enumerate((1, 2, 3, 4, 5)):
            true_label = (
                label
                if code == 1
                else (P300Label.UNKNOWN if label is P300Label.UNKNOWN else P300Label.NON_TARGET)
            )
            predictions.append(
                EpochPrediction(
                    epoch_id=f"epoch-{trial_index}-{index}",
                    event_id=f"event-{trial_index}-{index}",
                    selection_trial_id=f"selection-{trial_index}",
                    recording_id="P_03:SE002:recording",
                    subject_id="P_03",
                    session_id="SE002",
                    true_label=true_label,
                    target_probability=0.9 if code == 1 else 0.1,
                    predicted_target=code == 1,
                    onset_seconds=(
                        None if trial_index == 2 and index == 0 else trial_index * 10 + index * 0.25
                    ),
                    stimulus_code=code,
                )
            )
    evaluation = DecoderEvaluation(
        dataset_sha256="c" * 64,
        predictions=tuple(predictions),
        labeled_epoch_count=10,
        unknown_epoch_count=5,
        metrics=BinaryDecoderMetrics(
            auroc=1.0,
            balanced_accuracy=1.0,
            brier_score=0.01,
            negative_log_likelihood=0.02,
            expected_calibration_error=0.01,
        ),
        selection_trial_count=1,
        selection_code_set_accuracy=1.0,
    )

    extracted = flash_trials_from_decoder_evaluation(evaluation)

    assert len(extracted) == 1
    assert extracted[0].selection_trial_id == "selection-0"
    assert extracted[0].recorded_target_codes == (1,)
    assert tuple(item.event_id for item in extracted[0].events) == tuple(
        f"event-0-{index}" for index in range(5)
    )


def test_tracked_counterfactual_config_is_strict(tmp_path: Path) -> None:
    spec = load_counterfactual_spec("configs/experiments/counterfactual_fusion.yaml")
    assert spec.conditions == ALL_CONDITIONS
    assert len(spec.digest()) == 64
    with pytest.raises(ValidationError):
        CounterfactualFusionSpec(
            experiment_id="invalid",
            conditions=(EvaluationCondition.CURRENT_SAFE_FUSION,),
            personalization_evidence_kind="controlled_fixture",
        )
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- invalid\n- config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_counterfactual_spec(invalid)
