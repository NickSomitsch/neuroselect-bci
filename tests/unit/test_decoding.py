from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neuroselect.bci import EpochReplay, ReplayState
from neuroselect.decoding import (
    CalibratedP300Decoder,
    ClassicalDecoderConfig,
    DecoderEvaluation,
    DecoderTrainingSummary,
    evaluate_decoder,
    fit_calibrated_decoder,
    load_classical_decoder_config,
    load_partitioned_epoch_batches,
    read_decoder_artifacts,
    write_decoder_artifacts,
)
from neuroselect.eeg import (
    DataSplit,
    EpochBatch,
    EpochMetadata,
    P300Label,
    PreprocessingConfig,
    PreprocessingReport,
    SubjectSplit,
    write_epoch_batch,
)

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def make_batch(
    subject_id: str,
    *,
    seed: int,
    unknown: bool = False,
    recording_suffix: str = "labeled",
    reverse: bool = False,
) -> EpochBatch:
    generator = np.random.default_rng(seed)
    channel_count = 4
    sample_count = 48
    codes = tuple(range(1, 7))
    repeats = 2
    trial_count = 8
    rows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[EpochMetadata] = []
    recording_id = f"{subject_id}:SE001:{recording_suffix}"
    index = 0
    for trial_index in range(trial_count):
        target_code = codes[trial_index % len(codes)]
        for _ in range(repeats):
            for code in codes:
                is_target = code == target_code
                epoch = generator.normal(0.0, 0.35, (channel_count, sample_count)).astype(
                    np.float32
                )
                if is_target:
                    epoch[0, 14:28] += 2.8
                    epoch[1, 18:32] += 1.8
                rows.append(epoch)
                numeric_label = -1 if unknown else int(is_target)
                labels.append(numeric_label)
                event_id = f"{recording_id}:event-{index:04d}"
                metadata.append(
                    EpochMetadata(
                        epoch_id=f"{event_id}:epoch",
                        event_id=event_id,
                        selection_trial_id=(f"{recording_id}:selection-{trial_index:03d}"),
                        recording_id=recording_id,
                        subject_id=subject_id,
                        session_id="SE001",
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
    order = np.arange(len(rows) - 1, -1, -1) if reverse else np.arange(len(rows))
    ordered_metadata = tuple(metadata[int(item)] for item in order)
    return EpochBatch(
        data=np.stack([rows[int(item)] for item in order]),
        labels=np.asarray([labels[int(item)] for item in order], dtype=np.int8),
        channel_names=("Fz", "Cz", "Pz", "Oz"),
        sampling_rate_hz=128.0,
        metadata=ordered_metadata,
        config=PreprocessingConfig(),
        report=PreprocessingReport(
            recording_id=recording_id,
            input_event_count=len(rows),
            accepted_event_count=len(rows),
        ),
    )


@pytest.fixture(scope="module")
def decoder_bundle() -> tuple[
    CalibratedP300Decoder,
    DecoderTrainingSummary,
    EpochBatch,
    EpochBatch,
]:
    training = make_batch("P_01", seed=1)
    calibration = make_batch("P_02", seed=2)
    decoder, summary = fit_calibrated_decoder((training,), (calibration,))
    return decoder, summary, training, calibration


def test_calibrated_decoder_is_deterministic_and_scores_only_known_labels(
    decoder_bundle: tuple[
        CalibratedP300Decoder,
        DecoderTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, summary, training, calibration = decoder_bundle
    repeated, repeated_summary = fit_calibrated_decoder((training,), (calibration,))
    test_batch = make_batch("P_03", seed=3)

    first = evaluate_decoder(decoder, (test_batch,))
    second = evaluate_decoder(repeated, (test_batch,))

    assert summary == repeated_summary
    assert np.allclose(
        [item.target_probability for item in first.predictions],
        [item.target_probability for item in second.predictions],
        atol=1e-12,
    )
    assert first.labeled_epoch_count == len(test_batch.labels)
    assert first.unknown_epoch_count == 0
    assert first.metrics is not None
    assert first.metrics.auroc > 0.98
    assert first.metrics.balanced_accuracy > 0.9
    assert first.metrics.brier_score < 0.1
    assert first.selection_trial_count == 8
    assert first.selection_code_set_accuracy == 1.0
    assert first.selection_metrics is not None
    assert first.selection_metrics.exact_target_event_set_accuracy == 1.0
    assert first.selection_metrics.target_event_recall_at_k == 1.0
    assert first.selection_metrics.target_event_average_precision == 1.0
    assert first.selection_metrics.top_event_hit_rate == 1.0

    unknown = make_batch("P_04", seed=4, unknown=True)
    replay_only = evaluate_decoder(decoder, (unknown,))
    assert replay_only.labeled_epoch_count == 0
    assert replay_only.unknown_epoch_count == len(unknown.labels)
    assert replay_only.metrics is None
    assert replay_only.selection_trial_count == 0
    assert all(prediction.true_label is P300Label.UNKNOWN for prediction in replay_only.predictions)


def test_unknown_epochs_are_excluded_from_fit_and_calibration() -> None:
    training = make_batch("P_05", seed=5)
    training_unknown = make_batch("P_05", seed=6, unknown=True, recording_suffix="unknown")
    calibration = make_batch("P_06", seed=7)
    calibration_unknown = make_batch("P_06", seed=8, unknown=True, recording_suffix="unknown")

    _, summary = fit_calibrated_decoder(
        (training, training_unknown), (calibration, calibration_unknown)
    )

    assert summary.training_epoch_count == len(training.labels)
    assert summary.calibration_epoch_count == len(calibration.labels)
    assert summary.excluded_unknown_training_count == len(training_unknown.labels)
    assert summary.excluded_unknown_calibration_count == len(calibration_unknown.labels)


def test_decoder_rejects_leakage_incompatible_tensors_and_invalid_labels(
    decoder_bundle: tuple[
        CalibratedP300Decoder,
        DecoderTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, _, training, calibration = decoder_bundle
    with pytest.raises(ValueError, match="leakage"):
        fit_calibrated_decoder((training,), (training,))
    with pytest.raises(ValueError, match="overlaps decoder development"):
        evaluate_decoder(decoder, (calibration,))

    incompatible = make_batch("P_03", seed=9)
    incompatible.channel_names = ("Cz", "Fz", "Pz", "Oz")
    with pytest.raises(ValueError, match="channel order"):
        evaluate_decoder(decoder, (incompatible,))
    with pytest.raises(ValueError, match="shape"):
        decoder.predict_probabilities(np.zeros((1, 4, 47), dtype=np.float32))
    invalid = np.zeros((1, 4, 48), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        decoder.predict_probabilities(invalid)

    mismatched_preprocessing = make_batch("P_13", seed=18)
    mismatched_preprocessing.config = mismatched_preprocessing.config.model_copy(
        update={"reject_peak_to_peak_v": 100e-6}
    )
    with pytest.raises(ValueError, match="preprocessing config"):
        evaluate_decoder(decoder, (mismatched_preprocessing,))
    with pytest.raises(ValueError, match="tensor contracts"):
        fit_calibrated_decoder((training,), (mismatched_preprocessing,))

    all_unknown = make_batch("P_07", seed=10, unknown=True)
    with pytest.raises(ValueError, match="both non-target and target"):
        fit_calibrated_decoder((all_unknown,), (make_batch("P_08", seed=11),))


def test_decoder_config_and_evaluation_models_are_strict(tmp_path: Path) -> None:
    config = load_classical_decoder_config()
    assert config.model_revision == "xdawn-shrinkage-lda-platt-v1"
    assert len(config.digest()) == 64
    with pytest.raises(ValidationError):
        ClassicalDecoderConfig(decision_threshold=1.0)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- invalid\n- config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_classical_decoder_config(invalid)
    with pytest.raises(ValidationError, match="cover every prediction"):
        DecoderEvaluation(
            dataset_sha256="a" * 64,
            predictions=(
                evaluate_decoder(
                    fit_calibrated_decoder(
                        (make_batch("P_09", seed=12),),
                        (make_batch("P_10", seed=13),),
                    )[0],
                    (make_batch("P_11", seed=14),),
                ).predictions[0],
            ),
            labeled_epoch_count=0,
            unknown_epoch_count=0,
        )


def test_decoder_artifacts_round_trip_and_detect_tampering(
    tmp_path: Path,
    decoder_bundle: tuple[
        CalibratedP300Decoder,
        DecoderTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, summary, _, _ = decoder_bundle
    evaluation = evaluate_decoder(decoder, (make_batch("P_03", seed=15),))
    manifest = write_decoder_artifacts(
        decoder,
        summary,
        evaluation,
        tmp_path,
        git_sha="61d6c6e",
        run_time=NOW,
    )

    restored, metadata, restored_evaluation, restored_manifest = read_decoder_artifacts(tmp_path)
    assert metadata.training_summary == summary
    assert restored_evaluation == evaluation
    assert restored_manifest == manifest
    assert restored.config == decoder.config
    assert manifest.package_versions["scikit-learn"]
    assert manifest.run_kind.value == "eeg_original_task"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_decoder_artifacts(
            decoder,
            summary,
            evaluation,
            tmp_path,
            git_sha="61d6c6e",
            run_time=NOW,
        )
    with pytest.raises(ValueError, match="timezone"):
        write_decoder_artifacts(
            decoder,
            summary,
            evaluation,
            tmp_path / "naive",
            git_sha="61d6c6e",
            run_time=datetime(2026, 7, 19, 12, 0),
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["datasets"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest does not agree"):
        read_decoder_artifacts(tmp_path)
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")

    (tmp_path / "decoder.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_decoder_artifacts(tmp_path)


def test_virtual_replay_preserves_order_labels_and_control_state(
    decoder_bundle: tuple[
        CalibratedP300Decoder,
        DecoderTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, _, _, _ = decoder_bundle
    batch = make_batch("P_04", seed=16, unknown=True, reverse=True)
    replay = EpochReplay(batch, speed=2.0, decoder=decoder)

    assert replay.state is ReplayState.READY
    assert replay.frame_count == len(batch.labels)
    with pytest.raises(RuntimeError, match="must be running"):
        replay.next_frame()
    replay.start()
    first = replay.next_frame()
    assert first is not None
    assert first.sequence_index == 0
    assert first.source_offset_seconds == 0.0
    assert first.replay_offset_seconds == 0.0
    assert first.label is P300Label.UNKNOWN
    assert first.supervised_label_available is False
    assert first.target_probability is not None
    assert first.data.flags.writeable is False

    replay.pause()
    with pytest.raises(RuntimeError, match="must be running"):
        replay.next_frame()
    replay.seek_seconds(1.0)
    assert replay.position == 4
    replay.start()
    sought = replay.next_frame()
    assert sought is not None
    assert sought.source_offset_seconds == 1.0
    assert sought.replay_offset_seconds == 0.5
    replay.set_speed(4.0)
    replay.seek_index(replay.frame_count - 1)
    replay.start()
    assert replay.next_frame() is not None
    assert replay.state.value == ReplayState.FINISHED.value
    assert replay.next_frame() is None
    with pytest.raises(RuntimeError, match="reset or seek"):
        replay.start()
    replay.reset()
    assert replay.position == 0
    assert replay.state is ReplayState.READY


def test_virtual_replay_rejects_invalid_timing_and_controls() -> None:
    batch = make_batch("P_12", seed=17, unknown=True)
    with pytest.raises(ValueError, match="speed"):
        EpochReplay(batch, speed=0.0)
    replay = EpochReplay(batch)
    with pytest.raises(RuntimeError, match="running replay"):
        replay.pause()
    with pytest.raises(ValueError, match="outside"):
        replay.seek_index(-1)
    with pytest.raises(ValueError, match="non-negative"):
        replay.seek_seconds(float("nan"))

    metadata = list(batch.metadata)
    metadata[0] = metadata[0].model_copy(update={"onset_seconds": None})
    batch.metadata = tuple(metadata)
    with pytest.raises(ValueError, match="onset seconds"):
        EpochReplay(batch)


def test_virtual_replay_rejects_decoder_tensor_mismatch(
    decoder_bundle: tuple[
        CalibratedP300Decoder,
        DecoderTrainingSummary,
        EpochBatch,
        EpochBatch,
    ],
) -> None:
    decoder, _, _, _ = decoder_bundle
    batch = make_batch("P_14", seed=19, unknown=True)
    batch.channel_names = ("Cz", "Fz", "Pz", "Oz")
    with pytest.raises(ValueError, match="does not match"):
        EpochReplay(batch, decoder=decoder)
    batch.channel_names = decoder.channel_names
    batch.config = batch.config.model_copy(update={"reject_peak_to_peak_v": 100e-6})
    with pytest.raises(ValueError, match="does not match"):
        EpochReplay(batch, decoder=decoder)


def test_training_command_loader_uses_the_tracked_subject_split(tmp_path: Path) -> None:
    split = SubjectSplit(
        seed=1,
        train_subjects=("P_15",),
        validation_subjects=("P_16",),
        test_subjects=("P_17",),
    )
    (tmp_path / "subject-split.json").write_text(split.model_dump_json(), encoding="utf-8")
    for subject, seed in (("P_15", 20), ("P_16", 21), ("P_17", 22)):
        write_epoch_batch(make_batch(subject, seed=seed), tmp_path / subject)

    partitions = load_partitioned_epoch_batches(tmp_path)

    assert {partition: len(batches) for partition, batches in partitions.items()} == {
        DataSplit.TRAIN: 1,
        DataSplit.VALIDATION: 1,
        DataSplit.TEST: 1,
    }
