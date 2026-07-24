from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import mne
import numpy as np
import numpy.typing as npt
import pytest
import yaml
from pydantic import ValidationError

import neuroselect.eeg.study_p as study_p_module
from neuroselect.eeg import (
    CHECKSUM_MANIFEST_SHA256,
    DataSplit,
    EpochBatch,
    EpochMetadata,
    P300Label,
    PreprocessingConfig,
    PreprocessingReport,
    SessionFold,
    SourcePartition,
    SpellingCondition,
    StandardizedRecording,
    StudyPSourceFile,
    SubjectSplit,
    assign_epochs_by_subject,
    cross_session_folds,
    download_pinned_inventory,
    download_source_files,
    load_pinned_inventory,
    make_subject_split,
    parse_checksum_inventory,
    preprocess_recording,
    read_epoch_batch,
    read_standardized_recording,
    select_source_files,
    sha256_file,
    split_for_subject,
    standardize_study_p_raw,
    validate_split_integrity,
    verify_sha256,
    write_epoch_batch,
    write_standardized_recording,
)
from neuroselect.eeg.models import ChannelMetadata, P300Event, RecordingMetadata


def make_source(
    *,
    subject_id: str = "P_01",
    session_id: str = "SE001",
    partition: SourcePartition = SourcePartition.TRAIN,
    condition: SpellingCondition = SpellingCondition.PREDICTIVE,
    digest: str = "0" * 64,
    suffix: str = "01",
) -> StudyPSourceFile:
    partition_path = partition.value.title()
    condition_path = {
        SpellingCondition.PREDICTIVE: "PredictiveSpelling",
        SpellingCondition.NON_PREDICTIVE: "NonpredictiveSpelling",
    }[condition]
    run_id = f"{subject_id}_{session_id}_{condition_path}_{partition_path}{suffix}"
    return StudyPSourceFile(
        relative_path=(
            f"bigP3BCI-data/StudyP/{subject_id}/{session_id}/{partition_path}/"
            f"{condition_path}/{run_id}.edf"
        ),
        sha256=digest,
        subject_id=subject_id,
        session_id=session_id,
        source_partition=partition,
        condition=condition,
        run_id=run_id,
    )


def make_source_raw(
    *,
    sampling_rate_hz: float = 256.0,
    include_markers: bool = True,
    include_phase: bool = True,
    include_events: bool = True,
    artifact: bool = False,
    negative_optional_value: bool = False,
    unknown_labels: bool = False,
) -> mne.io.RawArray:
    sample_count = 2_400
    times = np.arange(sample_count) / sampling_rate_hz
    eeg = np.asarray(
        [
            (8 + index) * 1e-6 * np.sin(2 * np.pi * (4 + index) * times + index / 3)
            for index in range(4)
        ]
    )
    if artifact:
        eeg[0, 620:660] += 1e-3
    channel_names = ["EEG_Fz", "EEG_Cz", "EEG_Pz", "EEG_Oz"]
    channel_types = ["eeg"] * 4
    data = [*eeg]
    if include_markers:
        stimulus_begin = np.zeros(sample_count)
        stimulus_type = np.zeros(sample_count)
        stimulus_code = np.zeros(sample_count)
        current_target = np.zeros(sample_count)
        selected_target = np.zeros(sample_count)
        event_samples = (128, 384, 640, 896, 1_152) if include_events else ()
        for index, sample in enumerate(event_samples):
            stimulus_begin[sample : sample + 2] = 1
            stimulus_type[sample : sample + 2] = 0 if unknown_labels else index % 2
            stimulus_code[sample : sample + 2] = index + 1
            current_target[sample : sample + 2] = 9
            selected_target[sample : sample + 2] = 9
        if negative_optional_value and event_samples:
            stimulus_code[event_samples[0]] = -1
        data.extend([stimulus_begin, stimulus_type, stimulus_code, current_target, selected_target])
        channel_names.extend(
            [
                "StimulusBegin",
                "StimulusType",
                "StimulusCode",
                "CurrentTarget",
                "SelectedTarget",
            ]
        )
        channel_types.extend(["misc"] * 5)
        if include_phase:
            phase = np.zeros(sample_count)
            phase[80:720] = 1
            phase[720:2_000] = 2
            data.append(phase)
            channel_names.append("PhaseInSequence")
            channel_types.append("misc")
    info = mne.create_info(channel_names, sampling_rate_hz, channel_types)
    return mne.io.RawArray(np.asarray(data), info, verbose=False)


def make_recording(**raw_options: Any) -> StandardizedRecording:
    return standardize_study_p_raw(
        make_source_raw(**raw_options), make_source(), expected_channel_count=4
    )


def inventory_line(source: StudyPSourceFile) -> str:
    return f"{source.sha256} {source.relative_path}"


def make_epoch(
    subject_id: str,
    partition_index: int,
    *,
    session_id: str = "SE001",
    trial_id: str | None = None,
) -> EpochMetadata:
    recording_id = f"{subject_id}:{session_id}:run-{partition_index}"
    event_id = f"{recording_id}:event-{partition_index}"
    return EpochMetadata(
        epoch_id=f"{event_id}:epoch",
        event_id=event_id,
        selection_trial_id=trial_id or f"{recording_id}:selection-{partition_index}",
        recording_id=recording_id,
        subject_id=subject_id,
        session_id=session_id,
        label=P300Label.TARGET if partition_index % 2 else P300Label.NON_TARGET,
        onset_sample=partition_index,
    )


def rebuild_batch(
    batch: EpochBatch,
    *,
    data: npt.NDArray[np.floating] | None = None,
    labels: npt.NDArray[np.integer] | None = None,
) -> EpochBatch:
    return EpochBatch(
        data=batch.data if data is None else data,
        labels=batch.labels if labels is None else labels,
        channel_names=batch.channel_names,
        sampling_rate_hz=batch.sampling_rate_hz,
        metadata=batch.metadata,
        config=batch.config,
        report=batch.report,
    )


def test_parse_and_select_checksum_inventory() -> None:
    first = make_source()
    second = make_source(
        subject_id="P_02",
        session_id="SE002",
        partition=SourcePartition.TEST,
        condition=SpellingCondition.NON_PREDICTIVE,
        digest="a" * 64,
    )
    content = "\n".join((inventory_line(second), inventory_line(first)))

    parsed = parse_checksum_inventory(content, require_complete_study=False)

    assert parsed == (first, second)
    assert select_source_files(
        parsed,
        subject_ids=["P_02"],
        session_ids=["SE002"],
        source_partitions=[SourcePartition.TEST],
    ) == (second,)
    with pytest.raises(ValueError, match="unknown Study P IDs"):
        select_source_files(parsed, subject_ids=["P_99"])
    with pytest.raises(ValueError, match="contains no Study P"):
        select_source_files(parsed, subject_ids=["P_03"])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-a-checksum path", "invalid SHA-256"),
        (f"{'0' * 64} ../unsafe.edf", "unsafe inventory path"),
        ("", "contains no Study P"),
    ],
)
def test_inventory_rejects_malformed_input(content: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_checksum_inventory(content, require_complete_study=False)


def test_inventory_rejects_duplicates_and_incomplete_release() -> None:
    source = make_source()
    duplicate = "\n".join((inventory_line(source), inventory_line(source)))
    with pytest.raises(ValueError, match="duplicate inventory path"):
        parse_checksum_inventory(duplicate, require_complete_study=False)
    with pytest.raises(ValueError, match="must contain 228"):
        parse_checksum_inventory(inventory_line(source))


def test_source_file_model_rejects_path_metadata_mismatch() -> None:
    payload = make_source().model_dump()
    payload["subject_id"] = "P_02"
    with pytest.raises(ValidationError, match="subject/session"):
        StudyPSourceFile.model_validate(payload)


def test_pinned_inventory_and_hash_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source()
    inventory = tmp_path / "SHA256SUMS.txt"
    inventory.write_text(inventory_line(source) + "\n", encoding="utf-8")
    digest = sha256_file(inventory)
    monkeypatch.setattr(study_p_module, "CHECKSUM_MANIFEST_SHA256", digest)
    monkeypatch.setattr(study_p_module, "EXPECTED_SOURCE_FILE_COUNT", 1)
    monkeypatch.setattr(study_p_module, "EXPECTED_SUBJECT_IDS", ("P_01",))
    monkeypatch.setattr(study_p_module, "EXPECTED_SESSION_IDS", ("SE001",))

    assert load_pinned_inventory(inventory) == (source,)
    verify_sha256(inventory, digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(inventory, "f" * 64)
    assert digest != CHECKSUM_MANIFEST_SHA256


def test_explicit_download_is_checksum_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    remote_file = remote_root / make_source().relative_path
    remote_file.parent.mkdir(parents=True)
    remote_file.write_bytes(b"immutable-edf-fixture")
    digest = hashlib.sha256(remote_file.read_bytes()).hexdigest()
    source = make_source(digest=digest)
    monkeypatch.setattr(study_p_module, "DOWNLOAD_BASE_URL", remote_root.as_uri() + "/")

    with pytest.raises(PermissionError, match="accept_license"):
        download_source_files((source,), local_root, accept_license=False)
    downloaded = download_source_files(
        (source,),
        local_root,
        accept_license=True,
        workers=2,
    )
    assert downloaded[0].read_bytes() == b"immutable-edf-fixture"
    assert download_source_files((source,), local_root, accept_license=True) == downloaded
    with pytest.raises(ValueError, match="workers must lie"):
        download_source_files(
            (source,),
            local_root,
            accept_license=True,
            workers=17,
        )

    bad_source = source.model_copy(update={"sha256": "f" * 64})
    downloaded[0].unlink()
    with pytest.raises(ValueError, match="downloaded SHA-256 mismatch"):
        download_source_files((bad_source,), local_root, accept_license=True)
    assert not downloaded[0].exists()


def test_source_download_retries_transient_network_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    remote_file = remote_root / make_source().relative_path
    remote_file.parent.mkdir(parents=True)
    remote_file.write_bytes(b"retry-safe-edf-fixture")
    source = make_source(digest=hashlib.sha256(remote_file.read_bytes()).hexdigest())
    attempts = 0

    def flaky_urlopen(request: str | Request, timeout: float | None = None) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionResetError("transient fixture failure")
        return urlopen(request, timeout=timeout)

    monkeypatch.setattr(study_p_module, "DOWNLOAD_BASE_URL", remote_root.as_uri() + "/")
    monkeypatch.setattr("neuroselect.eeg.study_p.urlopen", flaky_urlopen)
    monkeypatch.setattr("neuroselect.eeg.study_p.time.sleep", lambda _: None)

    downloaded = download_source_files((source,), local_root, accept_license=True)

    assert downloaded[0].read_bytes() == b"retry-safe-edf-fixture"
    assert attempts == 2

    downloaded[0].unlink()
    attempts = 0
    delays: list[int] = []

    def unavailable_urlopen(request: str | Request, timeout: float | None = None) -> Any:
        del request, timeout
        nonlocal attempts
        attempts += 1
        raise ConnectionResetError("persistent fixture failure")

    monkeypatch.setattr("neuroselect.eeg.study_p.urlopen", unavailable_urlopen)
    monkeypatch.setattr("neuroselect.eeg.study_p.time.sleep", delays.append)

    with pytest.raises(ConnectionResetError, match="persistent"):
        download_source_files((source,), local_root, accept_license=True)

    assert attempts == 3
    assert delays == [1, 2]
    assert not tuple(local_root.rglob("tmp*"))


def test_pinned_manifest_download_requires_acceptance_and_reuses_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.txt"
    destination.write_text("fixture", encoding="utf-8")
    digest = sha256_file(destination)
    monkeypatch.setattr(study_p_module, "CHECKSUM_MANIFEST_SHA256", digest)
    with pytest.raises(PermissionError, match="accept_license"):
        download_pinned_inventory(destination, accept_license=False)
    assert download_pinned_inventory(destination, accept_license=True) == destination


def test_standardize_source_shaped_raw_preserves_event_provenance() -> None:
    recording = make_recording(negative_optional_value=True)

    assert recording.raw.ch_names == ["Fz", "Cz", "Pz", "Oz"]
    assert recording.raw.get_channel_types() == ["eeg"] * 4
    assert recording.metadata.key.recording_id.startswith("P_01:SE001:")
    assert recording.metadata.sampling_rate_hz == 256
    assert len(recording.metadata.events) == 5
    assert {event.label for event in recording.metadata.events} == {
        P300Label.NON_TARGET,
        P300Label.TARGET,
    }
    assert recording.metadata.labels_available is True
    assert recording.metadata.events[0].stimulus_code is None
    assert recording.metadata.events[1].stimulus_code == 2
    assert len({event.selection_trial_id for event in recording.metadata.events}) == 2
    assert all(np.isfinite(channel.position_m).all() for channel in recording.metadata.channels)


def test_standardize_without_phase_uses_one_leakage_group_per_event() -> None:
    recording = make_recording(include_phase=False)
    trial_ids = {event.selection_trial_id for event in recording.metadata.events}
    assert len(trial_ids) == len(recording.metadata.events)


def test_unlabeled_test_block_is_not_silently_treated_as_non_target() -> None:
    source = make_source(partition=SourcePartition.TEST)
    recording = standardize_study_p_raw(
        make_source_raw(unknown_labels=True), source, expected_channel_count=4
    )

    assert recording.metadata.labels_available is False
    assert {event.label for event in recording.metadata.events} == {P300Label.UNKNOWN}
    batch = preprocess_recording(recording)
    assert np.array_equal(batch.labels, np.full(len(batch.labels), -1, dtype=np.int8))

    with pytest.raises(ValueError, match="Train recording contains no target labels"):
        standardize_study_p_raw(
            make_source_raw(unknown_labels=True), make_source(), expected_channel_count=4
        )


def test_standardize_rejects_invalid_source_contract() -> None:
    with pytest.raises(ValueError, match="256 Hz"):
        standardize_study_p_raw(
            make_source_raw(sampling_rate_hz=128.0), make_source(), expected_channel_count=4
        )
    with pytest.raises(ValueError, match="contain 5 EEG"):
        standardize_study_p_raw(make_source_raw(), make_source(), expected_channel_count=5)
    with pytest.raises(ValueError, match="missing StimulusBegin"):
        standardize_study_p_raw(
            make_source_raw(include_markers=False), make_source(), expected_channel_count=4
        )
    with pytest.raises(ValueError, match="no stimulus-onset"):
        standardize_study_p_raw(
            make_source_raw(include_events=False), make_source(), expected_channel_count=4
        )


def test_standardized_recording_checks_raw_alignment() -> None:
    recording = make_recording()
    with pytest.raises(ValueError, match="channel order"):
        StandardizedRecording(
            raw=recording.raw.copy().pick(["Cz", "Fz", "Pz", "Oz"]), metadata=recording.metadata
        )
    with pytest.raises(ValueError, match="sample count"):
        StandardizedRecording(raw=recording.raw.copy().crop(tmax=1), metadata=recording.metadata)


def test_recording_metadata_rejects_invalid_event_invariants() -> None:
    recording = make_recording()
    payload = recording.metadata.model_dump()
    payload["events"][1]["onset_seconds"] = 99
    with pytest.raises(ValidationError, match="event seconds"):
        RecordingMetadata.model_validate(payload)
    payload = recording.metadata.model_dump()
    payload["events"] = [payload["events"][0]]
    with pytest.raises(ValidationError, match="labeled recordings require both classes"):
        RecordingMetadata.model_validate(payload)
    payload = recording.metadata.model_dump()
    payload["channels"][1]["name"] = payload["channels"][0]["name"]
    with pytest.raises(ValidationError, match="channel names"):
        RecordingMetadata.model_validate(payload)


def test_preprocessing_produces_finite_decoder_ready_epochs() -> None:
    batch = preprocess_recording(make_recording())

    assert batch.data.dtype == np.float32
    assert batch.data.shape[0] == 5
    assert batch.data.shape[1] == 4
    assert batch.sampling_rate_hz == 128
    assert np.isfinite(batch.data).all()
    assert np.array_equal(batch.labels, np.asarray([0, 1, 0, 1, 0], dtype=np.int8))
    assert batch.report.accepted_event_count == 5
    assert not batch.report.rejected_epochs


def test_preprocessing_rejects_artifact_epoch_and_reports_reason() -> None:
    batch = preprocess_recording(make_recording(artifact=True))

    assert batch.report.accepted_event_count < batch.report.input_event_count
    assert batch.report.rejected_epochs
    assert any("Fz" in rejected.reasons for rejected in batch.report.rejected_epochs)


def test_preprocessing_supports_no_notch_and_native_output_rate() -> None:
    config = PreprocessingConfig(notch_hz=None, output_sampling_rate_hz=256)
    batch = preprocess_recording(make_recording(), config)
    assert batch.sampling_rate_hz == 256
    assert batch.config is config


def test_preprocessing_rejects_every_event_when_threshold_is_too_low() -> None:
    config = PreprocessingConfig(reject_peak_to_peak_v=1e-9, flat_peak_to_peak_v=0)
    with (
        pytest.warns(RuntimeWarning, match="All epochs were dropped"),
        pytest.raises(ValueError, match="rejected every event"),
    ):
        preprocess_recording(make_recording(), config)


@pytest.mark.parametrize(
    "payload",
    [
        {"low_cut_hz": 20, "high_cut_hz": 20},
        {"high_cut_hz": 64, "output_sampling_rate_hz": 128},
        {"baseline_seconds": (0.0, 0.1)},
        {"flat_peak_to_peak_v": 2e-4, "reject_peak_to_peak_v": 1e-4},
    ],
)
def test_preprocessing_config_rejects_invalid_ranges(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PreprocessingConfig.model_validate(payload)


def test_epoch_batch_validates_shape_labels_and_report() -> None:
    batch = preprocess_recording(make_recording())
    with pytest.raises(ValueError, match="shape"):
        rebuild_batch(batch, data=batch.data[0])
    with pytest.raises(ValueError, match="one value"):
        rebuild_batch(batch, labels=batch.labels[:-1])
    with pytest.raises(ValueError, match="numeric labels"):
        rebuild_batch(batch, labels=1 - batch.labels)
    with pytest.raises(ValueError, match="non-finite"):
        invalid = batch.data.copy()
        invalid[0, 0, 0] = np.nan
        rebuild_batch(batch, data=invalid)


def test_standardized_and_epoch_artifacts_round_trip(tmp_path: Path) -> None:
    recording = make_recording()
    destination = write_standardized_recording(recording, tmp_path)
    restored = read_standardized_recording(destination)
    assert restored.metadata == recording.metadata
    assert np.allclose(restored.raw.get_data(), recording.raw.get_data(), atol=1e-12)

    batch = preprocess_recording(recording)
    write_epoch_batch(batch, destination)
    restored_batch = read_epoch_batch(destination)
    assert np.array_equal(restored_batch.data, batch.data)
    assert np.array_equal(restored_batch.labels, batch.labels)
    assert restored_batch.metadata == batch.metadata
    assert restored_batch.metadata[0].onset_seconds == batch.metadata[0].onset_seconds
    assert restored_batch.metadata[0].stimulus_code == batch.metadata[0].stimulus_code
    epoch_payload = json.loads((destination / "epochs.json").read_text(encoding="utf-8"))
    assert epoch_payload["schema_version"] == "1.1"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_standardized_recording(recording, tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_epoch_batch(batch, destination)


def test_artifact_reader_detects_checksum_tampering(tmp_path: Path) -> None:
    recording = make_recording()
    destination = write_standardized_recording(recording, tmp_path)
    (destination / "recording.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_standardized_recording(destination)

    destination = write_standardized_recording(recording, tmp_path, overwrite=True)
    batch = preprocess_recording(recording)
    write_epoch_batch(batch, destination)
    checksums_path = destination / "epoch-checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["files"].pop("epochs.json")
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum set is incomplete"):
        read_epoch_batch(destination)


def test_subject_split_is_deterministic_and_complete() -> None:
    subjects = tuple(f"P_{index:02d}" for index in range(1, 20))
    first = make_subject_split(subjects)
    second = make_subject_split(reversed(subjects))

    assert first == second
    assert len(first.train_subjects) == 13
    assert len(first.validation_subjects) == 3
    assert len(first.test_subjects) == 3
    assert split_for_subject(first.test_subjects[0], first) is DataSplit.TEST
    with pytest.raises(ValueError, match="absent"):
        split_for_subject("P_99", first)


def test_tracked_dataset_config_matches_code_pins() -> None:
    payload = yaml.safe_load(
        Path("configs/datasets/bigp3bci_study_p.yaml").read_text(encoding="utf-8")
    )
    assert payload["source_version"] == study_p_module.SOURCE_VERSION
    assert payload["doi"] == study_p_module.SOURCE_DOI
    assert payload["checksum_inventory"]["sha256"] == CHECKSUM_MANIFEST_SHA256
    assert payload["source_file_count"] == study_p_module.EXPECTED_SOURCE_FILE_COUNT
    assert tuple(payload["subject_ids"]) == study_p_module.EXPECTED_SUBJECT_IDS
    assert tuple(payload["sessions"]) == study_p_module.EXPECTED_SESSION_IDS
    assert PreprocessingConfig.model_validate(payload["preprocessing"]) == PreprocessingConfig()
    split_payload = {
        key: value for key, value in payload["model_split"].items() if key != "strategy"
    }
    tracked_split = SubjectSplit.model_validate({"schema_version": "1.0", **split_payload})
    assert tracked_split == make_subject_split(study_p_module.EXPECTED_SUBJECT_IDS)


def test_subject_split_rejects_bad_requests_and_overlap() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_subject_split(("P_01", "P_01", "P_02"), validation_count=1, test_count=1)
    with pytest.raises(ValueError, match="non-empty"):
        make_subject_split(("P_01", "P_02", "P_03"), validation_count=0, test_count=1)
    with pytest.raises(ValueError, match="training subject"):
        make_subject_split(("P_01", "P_02"), validation_count=1, test_count=1)
    with pytest.raises(ValidationError, match="disjoint"):
        SubjectSplit(
            seed=1,
            train_subjects=("P_01",),
            validation_subjects=("P_01",),
            test_subjects=("P_02",),
        )


def test_epoch_assignment_prevents_subject_and_trial_leakage() -> None:
    split = SubjectSplit(
        seed=1,
        train_subjects=("P_01",),
        validation_subjects=("P_02",),
        test_subjects=("P_03",),
    )
    epochs = tuple(make_epoch(f"P_0{index}", index) for index in range(1, 4))
    assigned = assign_epochs_by_subject(epochs, split)
    assert [item.subject_id for item in assigned[DataSplit.TRAIN]] == ["P_01"]

    leaked = {
        DataSplit.TRAIN: (make_epoch("P_01", 1, trial_id="shared"),),
        DataSplit.VALIDATION: (make_epoch("P_02", 2, trial_id="shared"),),
        DataSplit.TEST: (make_epoch("P_03", 3),),
    }
    with pytest.raises(ValueError, match="selection trial leakage"):
        validate_split_integrity(leaked, require_subject_disjoint=True)

    same_subject = {
        DataSplit.TRAIN: (make_epoch("P_01", 1),),
        DataSplit.VALIDATION: (make_epoch("P_01", 2, session_id="SE002"),),
        DataSplit.TEST: (make_epoch("P_03", 3),),
    }
    with pytest.raises(ValueError, match="subject leakage"):
        validate_split_integrity(same_subject, require_subject_disjoint=True)
    validate_split_integrity(same_subject, require_subject_disjoint=False)


def test_split_integrity_rejects_missing_partition_and_duplicate_epoch() -> None:
    with pytest.raises(ValueError, match="must define"):
        validate_split_integrity({DataSplit.TRAIN: ()}, require_subject_disjoint=True)
    duplicate = make_epoch("P_01", 1)
    partitions = {
        DataSplit.TRAIN: (duplicate, duplicate),
        DataSplit.VALIDATION: (),
        DataSplit.TEST: (),
    }
    with pytest.raises(ValueError, match="duplicate epoch"):
        validate_split_integrity(partitions, require_subject_disjoint=True)


def test_cross_session_folds_are_bidirectional_and_disjoint() -> None:
    folds = cross_session_folds()
    assert folds[0].train_sessions == folds[1].test_sessions
    assert folds[0].test_sessions == folds[1].train_sessions
    with pytest.raises(ValidationError, match="must be disjoint"):
        SessionFold(fold_id="bad", train_sessions=("SE001",), test_sessions=("SE001",))


def test_low_level_models_reject_inconsistent_metadata() -> None:
    with pytest.raises(ValidationError, match="position_m"):
        ChannelMetadata.model_validate({"name": "Cz", "position_m": (0.0, 0.0)})
    recording = make_recording()
    event_payload = recording.metadata.events[0].model_dump()
    event_payload["onset_sample"] = -1
    with pytest.raises(ValidationError):
        P300Event.model_validate(event_payload)
    with pytest.raises(ValidationError, match="account for every"):
        PreprocessingReport(
            recording_id="recording",
            input_event_count=2,
            accepted_event_count=1,
            rejected_epochs=(),
        )
