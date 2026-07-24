from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from neuroselect.core.models import Candidate, CandidateKind, CandidateSet
from neuroselect.decoding import BinaryDecoderMetrics, DecoderEvaluation, EpochPrediction
from neuroselect.eeg import P300Label
from neuroselect.evaluation import (
    CounterfactualExperimentInput,
    CounterfactualFusionRunner,
    CounterfactualFusionSpec,
    EvaluationCondition,
    load_counterfactual_spec,
)
from neuroselect.evaluation.counterfactual_preparation import (
    CounterfactualInputBuilder,
    CounterfactualPreparationSpec,
    load_counterfactual_preparation_spec,
    read_counterfactual_input_artifacts,
    write_counterfactual_input_artifacts,
)
from neuroselect.evaluation.language_benchmark import (
    HeldOutLanguageResult,
    LanguageBenchmarkTrial,
)
from neuroselect.retrieval import CandidateRetrievalEvidence

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def language_candidate_set(ordinal: int, fixture_id: str = "default") -> CandidateSet:
    return CandidateSet(
        candidate_set_id=f"language-candidates-{fixture_id}-{ordinal}",
        context_sha256=f"{ordinal + 1:064x}",
        candidates=(
            Candidate(
                candidate_id="language-0",
                text="open window",
                kind=CandidateKind.PHRASE,
                origins=frozenset({"z-source", "a-source"}),
            ),
            Candidate(
                candidate_id="language-1",
                text="some water",
                kind=CandidateKind.PHRASE,
            ),
            Candidate(
                candidate_id="language-2",
                text="thank you",
                kind=CandidateKind.PHRASE,
            ),
            Candidate(
                candidate_id="control-other",
                text="Other…",
                kind=CandidateKind.CONTROL,
            ),
            Candidate(
                candidate_id="control-back",
                text="Back",
                kind=CandidateKind.CONTROL,
            ),
            Candidate(
                candidate_id="control-cancel",
                text="Cancel",
                kind=CandidateKind.CONTROL,
            ),
        ),
        generator_revision="natural-candidate-fixture-v1",
        prompt_revision="natural-candidate-fixture-v1",
    )


def language_trial(
    ordinal: int,
    *,
    profile_id: str = "synthetic-concise",
    message_id: str = "message-with-two-spans",
    message_span_count: int = 2,
) -> LanguageBenchmarkTrial:
    generic = {"language-0": 0.6, "language-1": 0.3, "language-2": 0.1}
    personalized = {"language-0": 0.7, "language-1": 0.2, "language-2": 0.1}
    intended_id = "language-0" if ordinal == 0 else None
    fixture_id = f"{profile_id}-{message_id}"
    return LanguageBenchmarkTrial(
        trial_id=f"language-trial-{fixture_id}-{ordinal}",
        profile_id=profile_id,
        message_id=message_id,
        span_index=ordinal,
        message_span_count=message_span_count,
        confirmed_context="" if ordinal == 0 else "open window",
        intended_text="open window" if ordinal == 0 else "missing phrase",
        candidate_set=language_candidate_set(ordinal, fixture_id),
        intended_candidate_id=intended_id,
        other_candidate_id="control-other",
        generic_language_support=generic,
        personalization_support=personalized,
        personalization_lift={
            candidate_id: personalized[candidate_id] - generic[candidate_id]
            for candidate_id in generic
        },
        retrieval_evidence=tuple(
            CandidateRetrievalEvidence(
                candidate_id=candidate_id,
                retrieval_support=0.0,
                record_ids=(),
                hits=(),
            )
            for candidate_id in generic
        ),
        generic_rank=1 if intended_id is not None else None,
        personalized_rank=1 if intended_id is not None else None,
        adapter_id="synthetic-concise-dev-v1",
        adapter_sha256="d" * 64,
    )


def language_result() -> HeldOutLanguageResult:
    return HeldOutLanguageResult.model_construct(
        trials=(language_trial(0), language_trial(1)),
        claim_eligible=False,
    )


def decoder_evaluation(trial_count: int = 3) -> DecoderEvaluation:
    predictions: list[EpochPrediction] = []
    for trial_index in range(trial_count):
        for code_index, code in enumerate(range(10, 17)):
            is_target = code == 10
            predictions.append(
                EpochPrediction(
                    epoch_id=f"epoch-{trial_index}-{code}",
                    event_id=f"event-{trial_index}-{code}",
                    selection_trial_id=f"selection-{trial_index}",
                    recording_id="P_02:SE001:fixture",
                    subject_id="P_02",
                    session_id="SE001",
                    true_label=(P300Label.TARGET if is_target else P300Label.NON_TARGET),
                    target_probability=0.9 if is_target else 0.1,
                    predicted_target=is_target,
                    onset_seconds=trial_index * 10 + code_index * 0.2,
                    stimulus_code=code,
                )
            )
    return DecoderEvaluation(
        dataset_sha256="c" * 64,
        predictions=tuple(predictions),
        labeled_epoch_count=len(predictions),
        unknown_epoch_count=0,
        metrics=BinaryDecoderMetrics(
            auroc=1.0,
            balanced_accuracy=1.0,
            brier_score=0.01,
            negative_log_likelihood=0.02,
            expected_calibration_error=0.01,
        ),
        selection_trial_count=trial_count,
        selection_code_set_accuracy=1.0,
    )


def multi_subject_decoder_evaluation(
    subject_ids: tuple[str, ...],
    *,
    trials_per_subject: int,
) -> DecoderEvaluation:
    predictions: list[EpochPrediction] = []
    for subject_index, subject_id in enumerate(subject_ids):
        for trial_index in range(trials_per_subject):
            selection_index = subject_index * trials_per_subject + trial_index
            for code_index, code in enumerate(range(10, 17)):
                is_target = code == 10
                predictions.append(
                    EpochPrediction(
                        epoch_id=f"epoch-{subject_id}-{trial_index}-{code}",
                        event_id=f"event-{subject_id}-{trial_index}-{code}",
                        selection_trial_id=f"{subject_id}-selection-{trial_index}",
                        recording_id=f"{subject_id}:SE001:fixture",
                        subject_id=subject_id,
                        session_id="SE001",
                        true_label=(P300Label.TARGET if is_target else P300Label.NON_TARGET),
                        target_probability=0.9 if is_target else 0.1,
                        predicted_target=is_target,
                        onset_seconds=selection_index * 10 + code_index * 0.2,
                        stimulus_code=code,
                    )
                )
    return DecoderEvaluation(
        dataset_sha256="c" * 64,
        predictions=tuple(predictions),
        labeled_epoch_count=len(predictions),
        unknown_epoch_count=0,
        metrics=BinaryDecoderMetrics(
            auroc=1.0,
            balanced_accuracy=1.0,
            brier_score=0.01,
            negative_log_likelihood=0.02,
            expected_calibration_error=0.01,
        ),
        selection_trial_count=len(subject_ids) * trials_per_subject,
        selection_code_set_accuracy=1.0,
    )


def preparation_spec() -> CounterfactualPreparationSpec:
    return CounterfactualPreparationSpec(
        experiment_id="counterfactual-input-test",
        maximum_messages=1,
    )


def fusion_spec() -> CounterfactualFusionSpec:
    return CounterfactualFusionSpec(
        experiment_id="counterfactual-fusion-test",
        conditions=(
            EvaluationCondition.A_BCI_ONLY,
            EvaluationCondition.F_COMPLETE_SYSTEM,
        ),
        bootstrap_resamples=100,
        personalization_evidence_kind="held_out_adapter",
    )


def prepared_input() -> tuple[CounterfactualPreparationSpec, CounterfactualExperimentInput]:
    spec = preparation_spec()
    experiment_input = CounterfactualInputBuilder(spec, fusion_spec()).build(
        language_result=language_result(),
        decoder_evaluation=decoder_evaluation(),
        source_decoder_manifest_sha256="a" * 64,
        original_task_evaluation_sha256="b" * 64,
        source_language_manifest_sha256="e" * 64,
        source_language_result_sha256="f" * 64,
        prepared_at=NOW,
    )
    return spec, experiment_input


def test_builder_pairs_complete_messages_and_preserves_source_trials() -> None:
    _, experiment_input = prepared_input()

    assert len(experiment_input.trials) == 2
    assert {trial.flash_trial.selection_trial_id for trial in experiment_input.trials} == {
        "selection-0",
        "selection-1",
    }
    assert {trial.message_id for trial in experiment_input.trials} == {"message-with-two-spans"}
    assert experiment_input.trials[0].flash_layout.candidate_code_sets["language-0"] == (10,)
    assert experiment_input.trials[1].resolved_target_candidate_id == "control-other"
    assert experiment_input.source_evidence_claim_eligible is False
    assert experiment_input.source_language_result_sha256 == "f" * 64
    assert '"origins":["a-source","z-source"]' in experiment_input.canonical_json()

    result = CounterfactualFusionRunner(experiment_input).run()

    assert result.claim_eligible is False
    assert result.source_language_manifest_sha256 == "e" * 64
    assert result.mapping_provenance[1].intended_candidate_was_absent
    assert result.mapping_provenance[1].mapped_target_candidate_id == "control-other"


def test_builder_requires_enough_eeg_trials_for_one_complete_message() -> None:
    with pytest.raises(ValueError, match="shortest complete language message requires 2"):
        CounterfactualInputBuilder(preparation_spec(), fusion_spec()).build(
            language_result=language_result(),
            decoder_evaluation=decoder_evaluation(trial_count=1),
            source_decoder_manifest_sha256="a" * 64,
            original_task_evaluation_sha256="b" * 64,
            source_language_manifest_sha256="e" * 64,
            source_language_result_sha256="f" * 64,
            prepared_at=NOW,
        )


def test_research_builder_creates_exact_subject_profile_balanced_sample(
    tmp_path: Path,
) -> None:
    spec = CounterfactualPreparationSpec(
        schema_version="2.0",
        preparation_revision="subject-profile-balanced-paired-input-v2",
        experiment_id="counterfactual-input-research-test",
        evidence_tier="research",
        maximum_messages=None,
        eeg_trial_order="seeded-subject-sha256-v1",
        sampling_revision="subject-profile-balanced-complete-message-v1",
        required_profile_ids=("synthetic-casual", "synthetic-concise"),
        required_eeg_subject_ids=("P_02", "P_11"),
        messages_per_profile_per_eeg_subject=1,
        required_message_span_count=2,
        inference_scope="study-p-dataset-specific-descriptive",
    )
    language_trials = tuple(
        language_trial(
            span_index,
            profile_id=profile_id,
            message_id=f"{profile_id}-message-{message_index}",
        )
        for profile_id in spec.required_profile_ids
        for message_index in range(2)
        for span_index in range(2)
    )
    research_language_result = HeldOutLanguageResult.model_construct(
        trials=language_trials,
        claim_eligible=True,
    )
    research_decoder = multi_subject_decoder_evaluation(
        spec.required_eeg_subject_ids,
        trials_per_subject=5,
    )
    experiment_input = CounterfactualInputBuilder(spec, fusion_spec()).build(
        language_result=research_language_result,
        decoder_evaluation=research_decoder,
        source_decoder_manifest_sha256="a" * 64,
        original_task_evaluation_sha256="b" * 64,
        source_language_manifest_sha256="e" * 64,
        source_language_result_sha256="f" * 64,
        prepared_at=NOW,
    )

    assert spec.planned_counterfactual_trial_count == 8
    assert spec.planned_trials_per_eeg_subject == 4
    assert len(experiment_input.trials) == 8
    assert {
        profile_id: sum(
            trial.synthetic_profile_id == profile_id for trial in experiment_input.trials
        )
        for profile_id in spec.required_profile_ids
    } == {"synthetic-casual": 4, "synthetic-concise": 4}
    assert {
        subject_id: sum(
            trial.flash_trial.subject_id == subject_id for trial in experiment_input.trials
        )
        for subject_id in spec.required_eeg_subject_ids
    } == {"P_02": 4, "P_11": 4}
    for message_id in {trial.message_id for trial in experiment_input.trials}:
        message_subjects = {
            trial.flash_trial.subject_id
            for trial in experiment_input.trials
            if trial.message_id == message_id
        }
        assert len(message_subjects) == 1
    assert experiment_input.source_evidence_claim_eligible is True
    assert "preregistered balanced subset" in experiment_input.preparation_limitations[1]
    assert "not population or clinical inference" in experiment_input.preparation_limitations[2]
    repeated = CounterfactualInputBuilder(spec, fusion_spec()).build(
        language_result=research_language_result,
        decoder_evaluation=research_decoder,
        source_decoder_manifest_sha256="a" * 64,
        original_task_evaluation_sha256="b" * 64,
        source_language_manifest_sha256="e" * 64,
        source_language_result_sha256="f" * 64,
        prepared_at=NOW,
    )
    assert repeated.canonical_json() == experiment_input.canonical_json()
    manifest = write_counterfactual_input_artifacts(
        experiment_input,
        spec,
        tmp_path,
        git_sha="b4cfd22",
        package_versions={"python": "test"},
        device={"platform": "test"},
    )
    restored, _ = read_counterfactual_input_artifacts(tmp_path)
    assert restored == experiment_input
    assert manifest.metadata["profile_trial_counts"] == {
        "synthetic-casual": 4,
        "synthetic-concise": 4,
    }
    assert manifest.metadata["eeg_subject_trial_counts"] == {
        "P_02": 4,
        "P_11": 4,
    }
    assert manifest.metadata["inference_scope"] == "study-p-dataset-specific-descriptive"


def test_balanced_research_builder_fails_closed_on_subject_capacity() -> None:
    spec = CounterfactualPreparationSpec(
        schema_version="2.0",
        preparation_revision="subject-profile-balanced-paired-input-v2",
        experiment_id="counterfactual-input-research-capacity-test",
        evidence_tier="research",
        maximum_messages=None,
        eeg_trial_order="seeded-subject-sha256-v1",
        sampling_revision="subject-profile-balanced-complete-message-v1",
        required_profile_ids=("synthetic-concise",),
        required_eeg_subject_ids=("P_02", "P_11"),
        messages_per_profile_per_eeg_subject=1,
        required_message_span_count=2,
        inference_scope="study-p-dataset-specific-descriptive",
    )
    language_trials = tuple(
        language_trial(
            span_index,
            message_id=f"capacity-message-{message_index}",
        )
        for message_index in range(2)
        for span_index in range(2)
    )

    with pytest.raises(
        ValueError,
        match="EEG subject P_11 provides 0 usable selection trials",
    ):
        CounterfactualInputBuilder(spec, fusion_spec()).build(
            language_result=HeldOutLanguageResult.model_construct(
                trials=language_trials,
                claim_eligible=True,
            ),
            decoder_evaluation=decoder_evaluation(trial_count=3),
            source_decoder_manifest_sha256="a" * 64,
            original_task_evaluation_sha256="b" * 64,
            source_language_manifest_sha256="e" * 64,
            source_language_result_sha256="f" * 64,
            prepared_at=NOW,
        )


def test_preparation_rejects_ineligible_protocols_and_invalid_recipes(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires the balanced v2 protocol"):
        CounterfactualPreparationSpec(
            experiment_id="invalid-research-limit",
            evidence_tier="research",
            maximum_messages=1,
        )
    with pytest.raises(ValueError, match="v1 preparation cannot declare balanced"):
        CounterfactualPreparationSpec(
            experiment_id="invalid-v1-balanced-fields",
            required_profile_ids=("synthetic-concise",),
        )
    with pytest.raises(ValueError, match="requires complete sampling parameters"):
        CounterfactualPreparationSpec(
            schema_version="2.0",
            preparation_revision="subject-profile-balanced-paired-input-v2",
            experiment_id="incomplete-balanced-research",
            evidence_tier="research",
            maximum_messages=None,
        )
    balanced_fields = {
        "schema_version": "2.0",
        "preparation_revision": "subject-profile-balanced-paired-input-v2",
        "evidence_tier": "research",
        "maximum_messages": None,
        "eeg_trial_order": "seeded-subject-sha256-v1",
        "sampling_revision": "subject-profile-balanced-complete-message-v1",
        "messages_per_profile_per_eeg_subject": 1,
        "required_message_span_count": 2,
        "inference_scope": "study-p-dataset-specific-descriptive",
    }
    with pytest.raises(ValueError, match="profile IDs must be unique"):
        CounterfactualPreparationSpec.model_validate(
            {
                **balanced_fields,
                "experiment_id": "duplicate-balanced-profiles",
                "required_profile_ids": (
                    "synthetic-concise",
                    "synthetic-concise",
                ),
                "required_eeg_subject_ids": ("P_02",),
            }
        )
    with pytest.raises(ValueError, match="EEG subject IDs must be unique"):
        CounterfactualPreparationSpec.model_validate(
            {
                **balanced_fields,
                "experiment_id": "duplicate-balanced-subjects",
                "required_profile_ids": ("synthetic-concise",),
                "required_eeg_subject_ids": ("P_02", "P_02"),
            }
        )
    invalid_config = tmp_path / "invalid-preparation.yaml"
    invalid_config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_counterfactual_preparation_spec(invalid_config)

    legacy_fusion = CounterfactualFusionSpec(
        schema_version="1.0",
        protocol_revision="offline-counterfactual-fusion-v1",
        experiment_id="legacy-counterfactual",
        conditions=(EvaluationCondition.A_BCI_ONLY,),
        personalization_evidence_kind="controlled_fixture",
    )
    with pytest.raises(ValueError, match="requires counterfactual protocol v2"):
        CounterfactualInputBuilder(preparation_spec(), legacy_fusion).build(
            language_result=language_result(),
            decoder_evaluation=decoder_evaluation(),
            source_decoder_manifest_sha256="a" * 64,
            original_task_evaluation_sha256="b" * 64,
            source_language_manifest_sha256="e" * 64,
            source_language_result_sha256="f" * 64,
            prepared_at=NOW,
        )
    controlled_fusion = fusion_spec().model_copy(
        update={"personalization_evidence_kind": "controlled_fixture"}
    )
    with pytest.raises(ValueError, match="held-out-adapter evidence"):
        CounterfactualInputBuilder(preparation_spec(), controlled_fusion).build(
            language_result=language_result(),
            decoder_evaluation=decoder_evaluation(),
            source_decoder_manifest_sha256="a" * 64,
            original_task_evaluation_sha256="b" * 64,
            source_language_manifest_sha256="e" * 64,
            source_language_result_sha256="f" * 64,
            prepared_at=NOW,
        )


def test_prepared_input_artifacts_round_trip_and_detect_tampering(
    tmp_path: Path,
) -> None:
    spec, experiment_input = prepared_input()
    with pytest.raises(ValueError, match="does not agree"):
        write_counterfactual_input_artifacts(
            experiment_input,
            spec.model_copy(update={"seed": spec.seed + 1}),
            tmp_path,
            git_sha="b4cfd22",
            package_versions={"python": "test"},
            device={"platform": "test"},
        )
    with pytest.raises(ValueError, match="missing preparation or language provenance"):
        write_counterfactual_input_artifacts(
            experiment_input.model_copy(
                update={
                    "source_language_manifest_sha256": None,
                    "source_language_result_sha256": None,
                }
            ),
            spec,
            tmp_path,
            git_sha="b4cfd22",
            package_versions={"python": "test"},
            device={"platform": "test"},
        )
    manifest = write_counterfactual_input_artifacts(
        experiment_input,
        spec,
        tmp_path,
        git_sha="b4cfd22",
        package_versions={"python": "test"},
        device={"platform": "test"},
    )

    restored, restored_manifest = read_counterfactual_input_artifacts(tmp_path)

    assert restored == experiment_input
    assert restored_manifest == manifest
    assert manifest.random_seeds == {"language_message_pairing": spec.seed}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_counterfactual_input_artifacts(
            experiment_input,
            spec,
            tmp_path,
            git_sha="b4cfd22",
            package_versions={"python": "test"},
            device={"platform": "test"},
        )
    (tmp_path / "input.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_counterfactual_input_artifacts(tmp_path)


def test_tracked_development_preparation_and_fusion_configs() -> None:
    spec = load_counterfactual_preparation_spec(
        "configs/experiments/counterfactual_input_development.yaml"
    )
    fusion = load_counterfactual_spec("configs/experiments/counterfactual_fusion_development.yaml")

    assert spec.evidence_tier == "development"
    assert spec.maximum_messages == 1
    assert spec.digest() == "74864c2aa83abdabd98a453d21216d0f6f6e438f8a0b5c306fbb39d865131d7a"
    assert EvaluationCondition.F_COMPLETE_SYSTEM in fusion.conditions
    assert EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL not in fusion.conditions
