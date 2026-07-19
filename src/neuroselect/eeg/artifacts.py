"""Checksum-addressed FIF and NumPy artifacts for the standardized EEG layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np

from neuroselect.eeg.models import (
    EpochMetadata,
    PreprocessingConfig,
    PreprocessingReport,
    RecordingMetadata,
)
from neuroselect.eeg.preprocessing import EpochBatch
from neuroselect.eeg.study_p import StandardizedRecording, sha256_file, verify_sha256


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _require_writable(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite EEG artifacts: {existing}")


def recording_artifact_directory(destination_root: str | Path, metadata: RecordingMetadata) -> Path:
    return (
        Path(destination_root)
        / metadata.key.subject_id
        / metadata.key.session_id
        / metadata.key.run_id
    )


def write_standardized_recording(
    recording: StandardizedRecording,
    destination_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write EEG-only MNE FIF, a typed sidecar, and output checksums."""

    destination = recording_artifact_directory(destination_root, recording.metadata)
    raw_path = destination / "recording-raw.fif"
    metadata_path = destination / "recording.json"
    checksum_path = destination / "checksums.json"
    _require_writable((raw_path, metadata_path, checksum_path), overwrite)
    destination.mkdir(parents=True, exist_ok=True)
    recording.raw.save(raw_path, overwrite=overwrite, verbose=False)
    metadata_path.write_text(
        _canonical_json(recording.metadata.model_dump(mode="json")), encoding="utf-8"
    )
    checksum_path.write_text(
        _canonical_json(
            {
                "schema_version": "1.0",
                "files": {
                    raw_path.name: sha256_file(raw_path),
                    metadata_path.name: sha256_file(metadata_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return destination


def _load_checksums(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError(f"invalid checksum sidecar: {path}")
    files = payload.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) for name, digest in files.items()
    ):
        raise ValueError(f"invalid checksum entries: {path}")
    return files


def read_standardized_recording(directory: str | Path) -> StandardizedRecording:
    source = Path(directory)
    raw_path = source / "recording-raw.fif"
    metadata_path = source / "recording.json"
    checksums = _load_checksums(source / "checksums.json")
    if set(checksums) != {raw_path.name, metadata_path.name}:
        raise ValueError("standardized recording checksum set is incomplete")
    verify_sha256(raw_path, checksums[raw_path.name])
    verify_sha256(metadata_path, checksums[metadata_path.name])
    metadata = RecordingMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    return StandardizedRecording(raw=raw, metadata=metadata)


def write_epoch_batch(
    batch: EpochBatch,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write decoder-ready arrays separately from typed epoch provenance."""

    destination = Path(directory)
    data_path = destination / "epochs.npz"
    metadata_path = destination / "epochs.json"
    checksum_path = destination / "epoch-checksums.json"
    _require_writable((data_path, metadata_path, checksum_path), overwrite)
    destination.mkdir(parents=True, exist_ok=True)
    with data_path.open("wb") as artifact:
        np.savez_compressed(artifact, data=batch.data, labels=batch.labels)
    metadata_path.write_text(
        _canonical_json(
            {
                "schema_version": "1.1",
                "channel_names": batch.channel_names,
                "sampling_rate_hz": batch.sampling_rate_hz,
                "config": batch.config.model_dump(mode="json"),
                "report": batch.report.model_dump(mode="json"),
                "epochs": [item.model_dump(mode="json") for item in batch.metadata],
            }
        ),
        encoding="utf-8",
    )
    checksum_path.write_text(
        _canonical_json(
            {
                "schema_version": "1.0",
                "files": {
                    data_path.name: sha256_file(data_path),
                    metadata_path.name: sha256_file(metadata_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return destination


def read_epoch_batch(directory: str | Path) -> EpochBatch:
    source = Path(directory)
    data_path = source / "epochs.npz"
    metadata_path = source / "epochs.json"
    checksums = _load_checksums(source / "epoch-checksums.json")
    if set(checksums) != {data_path.name, metadata_path.name}:
        raise ValueError("epoch checksum set is incomplete")
    verify_sha256(data_path, checksums[data_path.name])
    verify_sha256(metadata_path, checksums[metadata_path.name])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.1":
        raise ValueError("invalid epoch metadata schema")
    with np.load(data_path, allow_pickle=False) as arrays:
        data = arrays["data"]
        labels = arrays["labels"]
    return EpochBatch(
        data=data,
        labels=labels,
        channel_names=tuple(payload["channel_names"]),
        sampling_rate_hz=float(payload["sampling_rate_hz"]),
        metadata=tuple(EpochMetadata.model_validate(item) for item in payload["epochs"]),
        config=PreprocessingConfig.model_validate(payload["config"]),
        report=PreprocessingReport.model_validate(payload["report"]),
    )
