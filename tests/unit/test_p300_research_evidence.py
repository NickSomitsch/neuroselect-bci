from __future__ import annotations

import hashlib
import json
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
from neuroselect.decoding.research_evidence import (
    P300ResearchCheck,
    StudyPResearchEvidenceAudit,
    StudyPResearchEvidenceSpec,
    audit_study_p_research_evidence,
    load_study_p_research_evidence_spec,
)
from neuroselect.eeg import EpochMetadata, P300Label, PreprocessingConfig
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus

ROOT = Path(__file__).parents[2]


def fixture_spec(tmp_path: Path) -> StudyPResearchEvidenceSpec:
    tracked = load_study_p_research_evidence_spec(
        ROOT / "configs/experiments/study_p_research_evidence.yaml"
    )
    return tracked.model_copy(
        update={
            "processed_root": tmp_path / "processed",
            "decoder_artifacts": tmp_path / "decoder",
            "decoder_config": ROOT / tracked.decoder_config,
            "required_sessions": ("SE001",),
            "required_recordings_per_subject_per_session": 1,
            "required_training_subject_ids": ("P_01",),
            "required_validation_subject_ids": ("P_06",),
            "required_test_subject_ids": ("P_02",),
            "minimum_usable_test_trials_per_subject": 1,
        }
    )


def write_epoch_fixture(
    root: Path,
    *,
    subject_id: str,
    session_id: str = "SE001",
) -> None:
    run_id = f"{subject_id}_{session_id}_PredictiveSpelling_Train01"
    directory = root / subject_id / session_id / run_id
    directory.mkdir(parents=True)
    recording_id = f"{subject_id}:{session_id}:{run_id}"
    selection_id = f"{recording_id}:selection-00001"
    epochs = (
        EpochMetadata(
            epoch_id=f"{recording_id}:target",
            event_id=f"{recording_id}:target-event",
            selection_trial_id=selection_id,
            recording_id=recording_id,
            subject_id=subject_id,
            session_id=session_id,
            label=P300Label.TARGET,
            onset_sample=10,
            onset_seconds=0.1,
            stimulus_code=10,
        ),
        EpochMetadata(
            epoch_id=f"{recording_id}:non-target",
            event_id=f"{recording_id}:non-target-event",
            selection_trial_id=selection_id,
            recording_id=recording_id,
            subject_id=subject_id,
            session_id=session_id,
            label=P300Label.NON_TARGET,
            onset_sample=20,
            onset_seconds=0.2,
            stimulus_code=11,
        ),
    )
    data_content = b"safe-npz-fixture"
    metadata_content = (
        json.dumps(
            {
                "schema_version": "1.1",
                "epochs": [item.model_dump(mode="json") for item in epochs],
            },
            sort_keys=True,
        )
        + "\n"
    )
    (directory / "epochs.npz").write_bytes(data_content)
    (directory / "epochs.json").write_text(metadata_content, encoding="utf-8")
    (directory / "epoch-checksums.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "files": {
                    "epochs.npz": hashlib.sha256(data_content).hexdigest(),
                    "epochs.json": hashlib.sha256(metadata_content.encode()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )


def write_decoder_fixture(spec: StudyPResearchEvidenceSpec) -> None:
    config = ClassicalDecoderConfig()
    metadata = DecoderCheckpointMetadata(
        config=config,
        training_summary=DecoderTrainingSummary(
            model_revision=config.model_revision,
            config_sha256=config.digest(),
            training_dataset_sha256="1" * 64,
            calibration_dataset_sha256="2" * 64,
            training_epoch_count=2,
            calibration_epoch_count=2,
            excluded_unknown_training_count=0,
            excluded_unknown_calibration_count=0,
            training_subject_ids=spec.required_training_subject_ids,
            calibration_subject_ids=spec.required_validation_subject_ids,
            channel_names=("Cz",),
            sampling_rate_hz=128.0,
            epoch_sample_count=16,
            preprocessing_config=PreprocessingConfig(),
        ),
    )
    predictions = (
        EpochPrediction(
            epoch_id="test-target",
            event_id="test-target-event",
            selection_trial_id="P_02-selection-1",
            recording_id="P_02:SE001:fixture",
            subject_id="P_02",
            session_id="SE001",
            true_label=P300Label.TARGET,
            target_probability=0.9,
            predicted_target=True,
            onset_seconds=0.1,
            stimulus_code=10,
        ),
        EpochPrediction(
            epoch_id="test-nontarget",
            event_id="test-nontarget-event",
            selection_trial_id="P_02-selection-1",
            recording_id="P_02:SE001:fixture",
            subject_id="P_02",
            session_id="SE001",
            true_label=P300Label.NON_TARGET,
            target_probability=0.1,
            predicted_target=False,
            onset_seconds=0.2,
            stimulus_code=11,
        ),
    )
    evaluation = DecoderEvaluation(
        dataset_sha256="3" * 64,
        predictions=predictions,
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
    directory = spec.decoder_artifacts
    directory.mkdir(parents=True)
    contents = {
        "decoder.json": metadata.model_dump_json().encode(),
        "evaluation.json": evaluation.model_dump_json().encode(),
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    manifest = RunManifest(
        run_id="p300-research-evidence-fixture",
        run_kind=RunKind.EEG_ORIGINAL_TASK,
        status=RunStatus.COMPLETED,
        started_at=spec.evaluated_at,
        completed_at=spec.evaluated_at,
        git_sha="b4321f7",
        config_sha256=config.digest(),
        random_seeds={"global": config.random_seed},
        package_versions={"python": "3.12"},
        device={"system": "test"},
        outputs=tuple(
            ArtifactRef(
                artifact_id=name,
                uri=f"artifact://{name}",
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for name, content in contents.items()
        ),
        metadata={"working_tree_dirty": False},
    )
    (directory / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )


def test_tracked_step_9_protocol_covers_full_subject_split() -> None:
    spec = load_study_p_research_evidence_spec(
        ROOT / "configs/experiments/study_p_research_evidence.yaml"
    )

    assert len(spec.required_subject_ids) == 19
    assert spec.required_recording_count == 190
    assert spec.minimum_usable_test_trials_per_subject == 48
    assert spec.required_test_subject_ids == ("P_02", "P_11", "P_13")


def test_step_9_data_and_decoder_audit_succeeds_on_verified_fixture(
    tmp_path: Path,
) -> None:
    spec = fixture_spec(tmp_path)
    for subject_id in spec.required_subject_ids:
        write_epoch_fixture(spec.processed_root, subject_id=subject_id)

    data_audit = audit_study_p_research_evidence(spec, include_decoder=False)

    assert data_audit.ready is True
    assert data_audit.decoder_ready is None
    assert data_audit.prepared_recording_count == 3
    assert data_audit.usable_test_trials_by_subject == {"P_02": 1}

    write_decoder_fixture(spec)
    audit = audit_study_p_research_evidence(spec)

    assert audit.ready is True
    assert audit.decoder_ready is True
    assert all(check.ready for check in audit.checks)
    assert '"ready":true' in audit.canonical_json()


def test_step_9_audit_fails_closed_on_missing_and_tampered_evidence(
    tmp_path: Path,
) -> None:
    spec = fixture_spec(tmp_path)
    missing = audit_study_p_research_evidence(spec)

    assert missing.ready is False
    assert {check.check_id for check in missing.checks if not check.ready} == {
        "data-recording-coverage",
        "data-test-replay-capacity",
        "decoder-artifacts",
    }

    for subject_id in spec.required_subject_ids:
        write_epoch_fixture(spec.processed_root, subject_id=subject_id)
    epoch_path = next(spec.processed_root.rglob("epochs.json"))
    epoch_path.write_text("tampered\n", encoding="utf-8")
    tampered = audit_study_p_research_evidence(spec, include_decoder=False)

    assert tampered.ready is False
    assert tampered.prepared_recording_count == 2
    assert any("checksum mismatch" in check.detail for check in tampered.checks)


def test_step_9_models_and_yaml_reject_inconsistent_status(
    tmp_path: Path,
) -> None:
    spec = fixture_spec(tmp_path)
    with pytest.raises(ValueError, match="splits must be disjoint"):
        StudyPResearchEvidenceSpec.model_validate(
            {
                **spec.model_dump(),
                "required_test_subject_ids": ("P_01",),
            }
        )
    with pytest.raises(ValueError, match="data status"):
        StudyPResearchEvidenceAudit(
            evidence_id="invalid",
            evaluated_at=spec.evaluated_at,
            config_sha256="a" * 64,
            include_decoder=False,
            prepared_recording_count=0,
            usable_test_trials_by_subject={},
            data_ready=True,
            decoder_ready=None,
            ready=True,
            checks=(
                P300ResearchCheck(
                    check_id="data-missing",
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
        load_study_p_research_evidence_spec(invalid)
