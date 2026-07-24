from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neuroselect.decoding import (
    BinaryDecoderMetrics,
    ClassicalDecoderConfig,
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    DecoderTrainingSummary,
    EpochPrediction,
)
from neuroselect.eeg import P300Label, PreprocessingConfig
from neuroselect.evaluation import EvaluationCondition, load_counterfactual_spec
from neuroselect.evaluation.counterfactual_preparation import (
    load_counterfactual_preparation_spec,
)
from neuroselect.evaluation.language_benchmark import (
    load_held_out_language_spec,
)
from neuroselect.evaluation.research_readiness import (
    PRIMARY_RESEARCH_CONDITIONS,
    ResearchExpansionReadiness,
    ResearchReadinessCheck,
    _decoder_state,
    assess_research_expansion,
    load_research_expansion_spec,
)
from neuroselect.language import load_lora_training_config
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus

ROOT = Path(__file__).parents[2]


def test_tracked_research_expansion_protocols_are_unlimited_and_strict() -> None:
    expansion = load_research_expansion_spec(
        ROOT / "configs/experiments/research_evidence_expansion.yaml"
    )
    language = load_held_out_language_spec(ROOT / expansion.language_evaluation_config)
    training = load_lora_training_config(ROOT / expansion.language_training_config)
    preparation = load_counterfactual_preparation_spec(
        ROOT / expansion.counterfactual_preparation_config
    )
    fusion = load_counterfactual_spec(ROOT / expansion.counterfactual_fusion_config)

    assert language.evidence_tier == "research"
    assert language.maximum_messages_per_profile is None
    assert training.trainer_revision == "neuroselect-mlx-lora-v1"
    assert preparation.evidence_tier == "research"
    assert preparation.maximum_messages is None
    assert preparation.sampling_revision == ("subject-profile-balanced-complete-message-v1")
    assert preparation.inference_scope == "study-p-dataset-specific-descriptive"
    assert preparation.planned_counterfactual_trial_count == 144
    assert preparation.planned_trials_per_eeg_subject == 48
    assert tuple(condition.value for condition in fusion.conditions) == PRIMARY_RESEARCH_CONDITIONS
    assert fusion.conditions[-1] is EvaluationCondition.F_COMPLETE_SYSTEM


def test_research_readiness_fails_closed_on_current_limited_artifacts() -> None:
    spec = load_research_expansion_spec(
        ROOT / "configs/experiments/research_evidence_expansion.yaml"
    )
    readiness = assess_research_expansion(
        spec.model_copy(
            update={
                "benchmark_spec": ROOT / spec.benchmark_spec,
                "profiles": ROOT / spec.profiles,
                "language_evaluation_config": (ROOT / spec.language_evaluation_config),
                "language_training_config": ROOT / spec.language_training_config,
                "language_model_config": ROOT / spec.language_model_config,
                "counterfactual_preparation_config": (
                    ROOT / spec.counterfactual_preparation_config
                ),
                "counterfactual_fusion_config": (ROOT / spec.counterfactual_fusion_config),
                "adapter_root": ROOT / "artifacts/does-not-exist/adapters",
                "language_artifacts": (ROOT / "artifacts/does-not-exist/language"),
                "decoder_artifacts": ROOT / "artifacts/does-not-exist/decoder",
            }
        )
    )

    assert readiness.ready is False
    assert readiness.required_message_count == 1_000
    assert readiness.required_language_trial_count == 3_990
    assert readiness.planned_counterfactual_trial_count == 144
    assert readiness.planned_trials_per_eeg_subject == 48
    assert readiness.available_p300_trial_count == 0
    blocked = {check.check_id for check in readiness.checks if not check.ready}
    assert {
        "research-adapters",
        "full-language-evaluation",
        "decoder-provenance",
        "decoder-training-subjects",
        "decoder-validation-subjects",
        "held-out-eeg-subjects",
        "balanced-p300-capacity",
    }.issubset(blocked)
    assert '"ready":false' in readiness.canonical_json()


def test_research_readiness_status_and_yaml_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must agree"):
        ResearchExpansionReadiness(
            expansion_id="invalid",
            evaluated_at=load_research_expansion_spec(
                ROOT / "configs/experiments/research_evidence_expansion.yaml"
            ).evaluated_at,
            config_sha256="a" * 64,
            benchmark_source_sha256="b" * 64,
            required_message_count=1,
            required_language_trial_count=1,
            planned_counterfactual_trial_count=1,
            planned_trials_per_eeg_subject=1,
            available_p300_trial_count=1,
            ready=True,
            checks=(
                ResearchReadinessCheck(
                    check_id="blocked",
                    ready=False,
                    observed="zero",
                    required="one",
                    detail="fixture",
                ),
            ),
        )
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_research_expansion_spec(invalid)


def test_decoder_readiness_verifies_json_without_loading_checkpoint(
    tmp_path: Path,
) -> None:
    evaluation = DecoderEvaluation(
        dataset_sha256="d" * 64,
        predictions=(
            EpochPrediction(
                epoch_id="epoch-target",
                event_id="event-target",
                selection_trial_id="selection-1",
                recording_id="P_02:SE001:fixture",
                subject_id="P_02",
                session_id="SE001",
                true_label=P300Label.TARGET,
                target_probability=0.9,
                predicted_target=True,
                onset_seconds=0.0,
                stimulus_code=10,
            ),
            EpochPrediction(
                epoch_id="epoch-nontarget",
                event_id="event-nontarget",
                selection_trial_id="selection-1",
                recording_id="P_02:SE001:fixture",
                subject_id="P_02",
                session_id="SE001",
                true_label=P300Label.NON_TARGET,
                target_probability=0.1,
                predicted_target=False,
                onset_seconds=0.2,
                stimulus_code=11,
            ),
        ),
        labeled_epoch_count=2,
        unknown_epoch_count=0,
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
    evaluation_content = evaluation.model_dump_json()
    (tmp_path / "evaluation.json").write_text(evaluation_content, encoding="utf-8")
    tracked = load_research_expansion_spec(
        ROOT / "configs/experiments/research_evidence_expansion.yaml"
    )
    decoder_config = ClassicalDecoderConfig()
    metadata = DecoderCheckpointMetadata(
        config=decoder_config,
        training_summary=DecoderTrainingSummary(
            model_revision=decoder_config.model_revision,
            config_sha256=decoder_config.digest(),
            training_dataset_sha256="a" * 64,
            calibration_dataset_sha256="b" * 64,
            training_epoch_count=10,
            calibration_epoch_count=5,
            excluded_unknown_training_count=0,
            excluded_unknown_calibration_count=0,
            training_subject_ids=("P_01", "P_03"),
            calibration_subject_ids=("P_06",),
            channel_names=("Cz",),
            sampling_rate_hz=128.0,
            epoch_sample_count=16,
            preprocessing_config=PreprocessingConfig(),
        ),
    )
    metadata_content = metadata.model_dump_json()
    (tmp_path / "decoder.json").write_text(metadata_content, encoding="utf-8")
    manifest = RunManifest(
        run_id="research-readiness-decoder-fixture",
        run_kind=RunKind.EEG_ORIGINAL_TASK,
        status=RunStatus.COMPLETED,
        started_at=tracked.evaluated_at,
        completed_at=tracked.evaluated_at,
        git_sha="b4321f7",
        config_sha256=decoder_config.digest(),
        random_seeds={"global": 1},
        package_versions={"python": "3.12"},
        device={"system": "test"},
        outputs=(
            ArtifactRef(
                artifact_id="evaluation",
                uri="artifact://evaluation.json",
                sha256=hashlib.sha256(evaluation_content.encode()).hexdigest(),
            ),
            ArtifactRef(
                artifact_id="metadata",
                uri="artifact://decoder.json",
                sha256=hashlib.sha256(metadata_content.encode()).hexdigest(),
            ),
        ),
        metadata={"working_tree_dirty": False},
    )
    (tmp_path / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    spec = tracked.model_copy(update={"decoder_artifacts": tmp_path})

    state = _decoder_state(spec)

    assert state.trial_count == 1
    assert state.trial_counts_by_subject == {"P_02": 1}
    assert state.training_subject_ids == {"P_01", "P_03"}
    assert state.validation_subject_ids == {"P_06"}
    assert state.clean is True
    assert "without loading the checkpoint" in state.detail

    (tmp_path / "evaluation.json").write_text(evaluation_content + "\n", encoding="utf-8")
    state = _decoder_state(spec)
    assert state.trial_count == 0
    assert state.trial_counts_by_subject == {}
    assert state.clean is False
    assert "checksum does not match" in state.detail
