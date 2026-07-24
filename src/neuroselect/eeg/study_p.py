"""Pinned bigP3BCI Study P inventory, download, and EDF standardization."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

import mne
import numpy as np
from mne.io import BaseRaw

from neuroselect.eeg.models import (
    ChannelMetadata,
    P300Event,
    P300Label,
    RecordingKey,
    RecordingMetadata,
    RecordingProvenance,
    SourcePartition,
    SpellingCondition,
    StudyPSourceFile,
)

DATASET_ID = "bigp3bci-study-p"
SOURCE_VERSION = "1.0.0"
SOURCE_DOI = "10.13026/0byy-ry86"
SOURCE_BASE_URL = "https://physionet.org/files/bigp3bci/1.0.0/"
DOWNLOAD_BASE_URL = "https://physionet-open.s3.amazonaws.com/bigp3bci/1.0.0/"
CHECKSUM_MANIFEST_URL = f"{SOURCE_BASE_URL}SHA256SUMS.txt"
CHECKSUM_MANIFEST_SHA256 = "75ce052ae8626a73b43887c994c4c0d17e5b0d775ad3083f759af20028e32fbb"
EXPECTED_SUBJECT_IDS = tuple(f"P_{index:02d}" for index in range(1, 20))
EXPECTED_SESSION_IDS = ("SE001", "SE002")
EXPECTED_CHANNEL_COUNT = 32
EXPECTED_SOURCE_FILE_COUNT = 228
DOWNLOAD_ATTEMPTS = 3

_SOURCE_PATTERN = re.compile(
    r"^bigP3BCI-data/StudyP/(?P<subject>P_[0-9]{2})/(?P<session>SE[0-9]{3})/"
    r"(?P<partition>Train|Test)/(?P<condition>PredictiveSpelling|NonpredictiveSpelling)/"
    r"(?P<filename>[^/]+\.edf)$"
)


class StandardizedRecording:
    """An unfiltered, EEG-only MNE recording plus its validated sidecar."""

    def __init__(self, *, raw: BaseRaw, metadata: RecordingMetadata) -> None:
        if raw.ch_names != [channel.name for channel in metadata.channels]:
            raise ValueError("raw channel order must match recording metadata")
        if raw.n_times != metadata.sample_count:
            raise ValueError("raw sample count must match recording metadata")
        self.raw = raw
        self.metadata = metadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> None:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {observed}")


def _source_from_inventory(path: str, digest: str) -> StudyPSourceFile | None:
    match = _SOURCE_PATTERN.fullmatch(path)
    if match is None:
        return None
    condition = {
        "PredictiveSpelling": SpellingCondition.PREDICTIVE,
        "NonpredictiveSpelling": SpellingCondition.NON_PREDICTIVE,
    }[match.group("condition")]
    return StudyPSourceFile(
        relative_path=path,
        sha256=digest,
        subject_id=match.group("subject"),
        session_id=match.group("session"),
        source_partition=SourcePartition(match.group("partition").lower()),
        condition=condition,
        run_id=match.group("filename").removesuffix(".edf"),
    )


def parse_checksum_inventory(
    content: str, *, require_complete_study: bool = True
) -> tuple[StudyPSourceFile, ...]:
    """Parse Study P records from PhysioNet's complete SHA-256 inventory."""

    sources: list[StudyPSourceFile] = []
    seen_paths: set[str] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise ValueError(f"invalid SHA-256 inventory line {line_number}")
        digest, relative_path = parts
        posix_path = PurePosixPath(relative_path)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            raise ValueError(f"unsafe inventory path on line {line_number}")
        if relative_path in seen_paths:
            raise ValueError(f"duplicate inventory path on line {line_number}")
        seen_paths.add(relative_path)
        source = _source_from_inventory(relative_path, digest)
        if source is not None:
            sources.append(source)

    sources.sort(key=lambda item: item.relative_path)
    result = tuple(sources)
    if not result:
        raise ValueError("checksum inventory contains no Study P EDF records")
    if require_complete_study:
        if len(result) != EXPECTED_SOURCE_FILE_COUNT:
            raise ValueError(
                f"Study P inventory must contain {EXPECTED_SOURCE_FILE_COUNT} EDF files"
            )
        if {item.subject_id for item in result} != set(EXPECTED_SUBJECT_IDS):
            raise ValueError("Study P inventory subject IDs do not match the pinned release")
        for subject_id in EXPECTED_SUBJECT_IDS:
            sessions = {item.session_id for item in result if item.subject_id == subject_id}
            if sessions != set(EXPECTED_SESSION_IDS):
                raise ValueError(f"Study P subject {subject_id} does not contain both sessions")
    return result


def load_pinned_inventory(path: str | Path) -> tuple[StudyPSourceFile, ...]:
    verify_sha256(path, CHECKSUM_MANIFEST_SHA256)
    return parse_checksum_inventory(Path(path).read_text(encoding="utf-8"))


def select_source_files(
    sources: Iterable[StudyPSourceFile],
    *,
    subject_ids: Iterable[str] | None = None,
    session_ids: Iterable[str] | None = None,
    source_partitions: Iterable[SourcePartition] | None = None,
) -> tuple[StudyPSourceFile, ...]:
    selected_subjects = set(subject_ids or EXPECTED_SUBJECT_IDS)
    selected_sessions = set(session_ids or EXPECTED_SESSION_IDS)
    selected_partitions = set(source_partitions or SourcePartition)
    unknown_subjects = selected_subjects - set(EXPECTED_SUBJECT_IDS)
    unknown_sessions = selected_sessions - set(EXPECTED_SESSION_IDS)
    if unknown_subjects or unknown_sessions:
        raise ValueError(
            f"unknown Study P IDs: subjects={sorted(unknown_subjects)}, "
            f"sessions={sorted(unknown_sessions)}"
        )
    result = tuple(
        source
        for source in sources
        if source.subject_id in selected_subjects
        and source.session_id in selected_sessions
        and source.source_partition in selected_partitions
    )
    if not result:
        raise ValueError("selection contains no Study P source files")
    return result


def _download_verified_file(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "NeuroSelect/0.1 dataset preparation"})
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    with urlopen(request, timeout=60) as response:
                        while chunk := response.read(1024 * 1024):
                            digest.update(chunk)
                            temporary.write(chunk)
                    observed = digest.hexdigest()
                    if observed != expected_sha256:
                        raise ValueError(
                            f"downloaded SHA-256 mismatch for {url}: "
                            f"expected {expected_sha256}, got {observed}"
                        )
                    os.replace(temporary_path, destination)
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise
            return
        except OSError:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(attempt)


def download_pinned_inventory(destination: str | Path, *, accept_license: bool) -> Path:
    """Explicitly download the small inventory; never runs during setup or tests."""

    if not accept_license:
        raise PermissionError("pass accept_license=True after reviewing CC-BY-4.0")
    path = Path(destination)
    if path.exists():
        verify_sha256(path, CHECKSUM_MANIFEST_SHA256)
        return path
    _download_verified_file(CHECKSUM_MANIFEST_URL, path, CHECKSUM_MANIFEST_SHA256)
    return path


def download_source_files(
    sources: Iterable[StudyPSourceFile],
    destination_root: str | Path,
    *,
    accept_license: bool,
    workers: int = 1,
) -> tuple[Path, ...]:
    """Download only selected EDFs, verifying every official checksum before use."""

    if not accept_license:
        raise PermissionError("pass accept_license=True after reviewing CC-BY-4.0")
    if workers < 1 or workers > 16:
        raise ValueError("download workers must lie in [1, 16]")
    root = Path(destination_root)

    def fetch(source: StudyPSourceFile) -> Path:
        destination = root / source.relative_path
        if destination.exists():
            verify_sha256(destination, source.sha256)
        else:
            _download_verified_file(
                f"{DOWNLOAD_BASE_URL}{source.relative_path}", destination, source.sha256
            )
        return destination

    selected = tuple(sources)
    if workers == 1:
        return tuple(fetch(source) for source in selected)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(fetch, selected))


def _sample_integer(raw: BaseRaw, channel_name: str, sample: int) -> int | None:
    if channel_name not in raw.ch_names:
        return None
    value = float(raw.get_data(picks=[channel_name], start=sample, stop=sample + 1)[0, 0])
    if not np.isfinite(value) or value < 0:
        return None
    return round(value)


def _extract_events(raw: BaseRaw, recording_id: str) -> tuple[tuple[P300Event, ...], bool]:
    required = {"StimulusBegin", "StimulusType"}
    if not required.issubset(raw.ch_names):
        raise ValueError("recording is missing StimulusBegin or StimulusType")
    stimulus_begin, stimulus_type = raw.get_data(picks=["StimulusBegin", "StimulusType"])
    active = stimulus_begin > 0.5
    onsets = np.flatnonzero(np.diff(np.concatenate(([False], active)).astype(np.int8)) == 1)
    if len(onsets) == 0:
        raise ValueError("recording contains no stimulus-onset events")
    labels_available = bool(np.any(stimulus_type[onsets] > 0.5))

    if "PhaseInSequence" in raw.ch_names:
        phase = raw.get_data(picks=["PhaseInSequence"])[0]
        boundaries = np.flatnonzero(np.diff(phase) != 0) + 1
        selection_segments = np.searchsorted(boundaries, onsets, side="right")
    else:
        selection_segments = np.arange(len(onsets))

    sampling_rate = float(raw.info["sfreq"])
    events = []
    for event_index, (onset, segment) in enumerate(
        zip(onsets.tolist(), selection_segments.tolist(), strict=True)
    ):
        events.append(
            P300Event(
                event_id=f"{recording_id}:event-{event_index:05d}",
                selection_trial_id=f"{recording_id}:selection-{segment:05d}",
                onset_sample=onset,
                onset_seconds=onset / sampling_rate,
                label=(P300Label.TARGET if stimulus_type[onset] > 0.5 else P300Label.NON_TARGET)
                if labels_available
                else P300Label.UNKNOWN,
                stimulus_code=_sample_integer(raw, "StimulusCode", onset),
                current_target=_sample_integer(raw, "CurrentTarget", onset),
                selected_target=_sample_integer(raw, "SelectedTarget", onset),
            )
        )
    return tuple(events), labels_available


def standardize_study_p_raw(
    raw: BaseRaw,
    source: StudyPSourceFile,
    *,
    expected_channel_count: int = EXPECTED_CHANNEL_COUNT,
) -> StandardizedRecording:
    """Convert one source-shaped EDF recording to EEG-only MNE plus JSON metadata."""

    sampling_rate = float(raw.info["sfreq"])
    if not np.isclose(sampling_rate, 256.0):
        raise ValueError(f"Study P source sampling rate must be 256 Hz, got {sampling_rate}")
    eeg_source_names = [name for name in raw.ch_names if name.startswith("EEG_")]
    if len(eeg_source_names) != expected_channel_count:
        raise ValueError(
            f"Study P recording must contain {expected_channel_count} EEG channels, "
            f"got {len(eeg_source_names)}"
        )

    key = RecordingKey(
        subject_id=source.subject_id,
        session_id=source.session_id,
        run_id=source.run_id,
        source_partition=source.source_partition,
        condition=source.condition,
    )
    events, labels_available = _extract_events(raw, key.recording_id)
    if source.source_partition is SourcePartition.TRAIN and not labels_available:
        raise ValueError("Study P Train recording contains no target labels")

    standardized = raw.copy().pick(eeg_source_names).load_data()
    rename = {
        name: {"EEG_FP1": "Fp1", "EEG_FP2": "Fp2"}.get(name, name.removeprefix("EEG_"))
        for name in eeg_source_names
    }
    standardized.rename_channels(rename)
    standardized.set_channel_types(dict.fromkeys(standardized.ch_names, "eeg"))
    standardized.set_montage("standard_1020", on_missing="raise")
    channels = tuple(
        ChannelMetadata(
            name=name,
            position_m=(
                float(channel["loc"][0]),
                float(channel["loc"][1]),
                float(channel["loc"][2]),
            ),
        )
        for name, channel in zip(standardized.ch_names, standardized.info["chs"], strict=True)
    )
    provenance = RecordingProvenance(
        source_url=f"{SOURCE_BASE_URL}{source.relative_path}",
        source_relative_path=source.relative_path,
        source_sha256=source.sha256,
        checksum_manifest_url=CHECKSUM_MANIFEST_URL,
        checksum_manifest_sha256=CHECKSUM_MANIFEST_SHA256,
    )
    metadata = RecordingMetadata(
        key=key,
        sampling_rate_hz=sampling_rate,
        sample_count=standardized.n_times,
        channels=channels,
        events=events,
        labels_available=labels_available,
        provenance=provenance,
    )
    return StandardizedRecording(raw=standardized, metadata=metadata)


def load_study_p_edf(
    source: StudyPSourceFile,
    dataset_root: str | Path,
) -> StandardizedRecording:
    """Checksum, load, and standardize an immutable local Study P EDF."""

    path = Path(dataset_root) / source.relative_path
    verify_sha256(path, source.sha256)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Channels contain different (highpass|lowpass) filters.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return standardize_study_p_raw(raw, source)
