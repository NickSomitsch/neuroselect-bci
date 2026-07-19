from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from neuroselect.bci import EpochReplay
from neuroselect.decoding import (
    EEGNetConfig,
    EEGNetP300Decoder,
    EEGNetTrainingSummary,
    adapt_eegnet_head,
    evaluate_chronological_session_drift,
    evaluate_decoder,
    fit_eegnet_decoder,
    load_eegnet_config,
    read_eegnet_artifacts,
    write_eegnet_artifacts,
)
from neuroselect.eeg import (
    EpochBatch,
    EpochMetadata,
    P300Label,
    PreprocessingConfig,
    PreprocessingReport,
    SessionFold,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def neural_config(**updates: object) -> EEGNetConfig:
    defaults: dict[str, object] = {
        "device": "cpu",
        "temporal_filters": 2,
        "depth_multiplier": 1,
        "pointwise_filters": 4,
        "temporal_kernel_samples": 15,
        "separable_kernel_samples": 7,
        "first_pool_size": 2,
        "second_pool_size": 2,
        "dropout": 0.0,
        "batch_size": 64,
        "max_epochs": 12,
        "learning_rate": 0.005,
        "early_stopping_patience": 12,
        "adaptation_max_epochs": 8,
        "adaptation_patience": 3,
        "minimum_adaptation_trials": 4,
    }
    return EEGNetConfig.model_validate({**defaults, **updates})


def make_neural_batch(
    subject_id: str,
    session_id: str,
    *,
    seed: int,
    unknown: bool = False,
    trial_count: int = 8,
) -> EpochBatch:
    generator = np.random.default_rng(seed)
    channel_count = 4
    sample_count = 64
    codes = (1, 2, 3, 4)
    repeats = 2
    rows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[EpochMetadata] = []
    recording_id = f"{subject_id}:{session_id}:{'unknown' if unknown else 'labeled'}"
    index = 0
    for trial_index in range(trial_count):
        target_code = codes[trial_index % len(codes)]
        for _ in range(repeats):
            for code in codes:
                is_target = code == target_code
                epoch = generator.normal(0.0, 0.3, (channel_count, sample_count)).astype(np.float32)
                if is_target:
                    epoch[0, 18:36] += 3.0
                    epoch[1, 22:40] += 2.0
                numeric_label = -1 if unknown else int(is_target)
                event_id = f"{recording_id}:event-{index:04d}"
                rows.append(epoch)
                labels.append(numeric_label)
                metadata.append(
                    EpochMetadata(
                        epoch_id=f"{event_id}:epoch",
                        event_id=event_id,
                        selection_trial_id=f"{recording_id}:selection-{trial_index:03d}",
                        recording_id=recording_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        label=(
                            P300Label.UNKNOWN
                            if unknown
                            else P300Label.TARGET
                            if is_target
                            else P300Label.NON_TARGET
                        ),
                        onset_sample=index * 64,
                        onset_seconds=index * 0.25,
                        stimulus_code=code,
                        current_target=target_code,
                        selected_target=target_code,
                    )
                )
                index += 1
    return EpochBatch(
        data=np.stack(rows),
        labels=np.asarray(labels, dtype=np.int8),
        channel_names=("Fz", "Cz", "Pz", "Oz"),
        sampling_rate_hz=128.0,
        metadata=tuple(metadata),
        config=PreprocessingConfig(),
        report=PreprocessingReport(
            recording_id=recording_id,
            input_event_count=len(rows),
            accepted_event_count=len(rows),
        ),
    )


@pytest.fixture(scope="module")
def neural_bundle() -> tuple[
    EEGNetP300Decoder,
    EEGNetTrainingSummary,
    EpochBatch,
    EpochBatch,
]:
    training = make_neural_batch("P_01", "SE001", seed=1)
    calibration = make_neural_batch("P_02", "SE001", seed=2)
    decoder, summary = fit_eegnet_decoder((training,), (calibration,), neural_config())
    return decoder, summary, training, calibration


def test_eegnet_is_deterministic_calibrated_and_cpu_compatible(
    neural_bundle: tuple[
        EEGNetP300Decoder,
        EEGNetTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, summary, training, calibration = neural_bundle
    repeated, repeated_summary = fit_eegnet_decoder((training,), (calibration,), neural_config())
    test = make_neural_batch("P_03", "SE002", seed=3)
    evaluation = evaluate_decoder(decoder, (test,))

    assert summary == repeated_summary
    assert summary.training_device == "cpu"
    assert np.allclose(
        decoder.predict_probabilities(test.data),
        repeated.predict_probabilities(test.data),
        atol=1e-7,
    )
    assert evaluation.metrics is not None
    assert evaluation.metrics.auroc > 0.8
    assert evaluation.selection_trial_count == 8
    probabilities = decoder.predict_probabilities(test.data)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_eegnet_excludes_unknown_epochs_and_rejects_leakage() -> None:
    training = make_neural_batch("P_04", "SE001", seed=4)
    unknown = make_neural_batch("P_04", "SE002", seed=5, unknown=True)
    calibration = make_neural_batch("P_05", "SE001", seed=6)
    decoder, summary = fit_eegnet_decoder(
        (training, unknown), (calibration,), neural_config(max_epochs=2)
    )

    assert summary.excluded_unknown_training_count == len(unknown.labels)
    with pytest.raises(ValueError, match="leakage"):
        fit_eegnet_decoder((training,), (training,), neural_config(max_epochs=1))
    with pytest.raises(ValueError, match="overlaps EEGNet development"):
        evaluate_decoder(decoder, (calibration,))
    replay_only = evaluate_decoder(
        decoder, (make_neural_batch("P_06", "SE001", seed=7, unknown=True),)
    )
    assert replay_only.metrics is None
    assert replay_only.unknown_epoch_count == len(replay_only.predictions)


def test_subject_adapter_freezes_features_and_keeps_target_session_untouched(
    neural_bundle: tuple[
        EEGNetP300Decoder,
        EEGNetTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, _, _, _ = neural_bundle
    source = make_neural_batch("P_03", "SE001", seed=8)
    source_unknown = make_neural_batch("P_03", "SE001", seed=9, unknown=True)
    target = make_neural_batch("P_03", "SE002", seed=10)

    adapted, summary = adapt_eegnet_head(decoder, (source, source_unknown))
    evaluation = evaluate_decoder(adapted, (target,))

    assert summary.source_session_id == "SE001"
    assert summary.target_session_id == "SE002"
    assert summary.excluded_unknown_count == len(source_unknown.labels)
    assert summary.feature_extractor_sha256_before == summary.feature_extractor_sha256_after
    assert set(summary.trained_parameters) == {
        "classifier.weight",
        "classifier.bias",
        "temperature",
    }
    assert evaluation.metrics is not None
    assert evaluation.dataset_sha256 not in {
        summary.head_dataset_sha256,
        summary.calibration_dataset_sha256,
    }
    with pytest.raises(ValueError, match="overlaps EEGNet development"):
        evaluate_decoder(adapted, (source,))
    replay = EpochReplay(target, decoder=adapted)
    replay.start()
    assert replay.next_frame() is not None


def test_chronological_drift_report_is_se001_to_se002_only(
    neural_bundle: tuple[
        EEGNetP300Decoder,
        EEGNetTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, _, _, _ = neural_bundle
    session_one = make_neural_batch("P_03", "SE001", seed=11)
    session_two = make_neural_batch("P_03", "SE002", seed=12)

    report = evaluate_chronological_session_drift(decoder, (session_two, session_one))

    assert report.fold.train_sessions == ("SE001",)
    assert report.fold.test_sessions == ("SE002",)
    assert report.subjects[0].adaptation is not None
    assert report.subjects[0].adaptation.source_session_id == "SE001"
    assert report.subjects[0].adaptation.target_session_id == "SE002"
    assert report.adapted_subject_count == 1
    assert report.fallback_subject_count == 0
    with pytest.raises(ValueError, match="SE001-to-SE002"):
        evaluate_chronological_session_drift(
            decoder,
            (session_one, session_two),
            SessionFold(
                fold_id="reverse-sensitivity-only",
                train_sessions=("SE002",),
                test_sessions=("SE001",),
            ),
        )
    with pytest.raises(ValueError, match="insufficient"):
        adapt_eegnet_head(
            decoder,
            (make_neural_batch("P_04", "SE001", seed=13, trial_count=2),),
        )
    fallback = evaluate_chronological_session_drift(
        decoder,
        (
            make_neural_batch("P_04", "SE001", seed=13, trial_count=2),
            make_neural_batch("P_04", "SE002", seed=14),
        ),
    )
    assert fallback.fallback_subject_count == 1
    assert fallback.subjects[0].adaptation is None
    assert fallback.subjects[0].conservative_abstention_required is True


def test_eegnet_artifacts_round_trip_and_verify_hashes(
    tmp_path: Path,
    neural_bundle: tuple[
        EEGNetP300Decoder,
        EEGNetTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, summary, _, _ = neural_bundle
    session_one = make_neural_batch("P_03", "SE001", seed=14)
    session_two = make_neural_batch("P_03", "SE002", seed=15)
    evaluation = evaluate_decoder(decoder, (session_one, session_two))
    drift = evaluate_chronological_session_drift(decoder, (session_one, session_two))
    manifest = write_eegnet_artifacts(
        decoder,
        summary,
        evaluation,
        tmp_path,
        git_sha="d733e27",
        run_time=NOW,
        drift_report=drift,
    )

    restored, metadata, restored_evaluation, restored_drift, restored_manifest = (
        read_eegnet_artifacts(tmp_path)
    )
    assert metadata.training_summary == summary
    assert restored_evaluation == evaluation
    assert restored_drift == drift
    assert restored_manifest == manifest
    assert np.allclose(
        restored.predict_probabilities(session_two.data),
        decoder.predict_probabilities(session_two.data),
        atol=1e-7,
    )
    assert manifest.package_versions["torch"] == torch.__version__
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_eegnet_artifacts(
            decoder,
            summary,
            evaluation,
            tmp_path,
            git_sha="d733e27",
            run_time=NOW,
        )
    (tmp_path / "eegnet.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_eegnet_artifacts(tmp_path)


def test_eegnet_config_is_strict_and_tracked(tmp_path: Path) -> None:
    config = load_eegnet_config()
    assert config.model_revision == "eegnet-temperature-v1"
    assert len(config.digest()) == 64
    with pytest.raises(ValidationError):
        EEGNetConfig(device="cuda")  # type: ignore[arg-type]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_eegnet_config(invalid)
