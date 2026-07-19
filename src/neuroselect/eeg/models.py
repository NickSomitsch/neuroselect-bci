"""Typed contracts for immutable EEG sources and leakage-safe P300 epochs."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from neuroselect.provenance.manifest import Sha256

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class P300Label(StrEnum):
    """Stimulus label, retaining unknown online/test-block events explicitly."""

    UNKNOWN = "unknown"
    NON_TARGET = "non_target"
    TARGET = "target"


class SourcePartition(StrEnum):
    """The dataset author's block label, not a NeuroSelect model split."""

    TRAIN = "train"
    TEST = "test"


class SpellingCondition(StrEnum):
    PREDICTIVE = "predictive"
    NON_PREDICTIVE = "non_predictive"


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class StudyPSourceFile(BaseModel):
    """One immutable Study P EDF entry from the official checksum inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(
        pattern=(
            r"^bigP3BCI-data/StudyP/P_[0-9]{2}/SE[0-9]{3}/"
            r"(Train|Test)/(PredictiveSpelling|NonpredictiveSpelling)/[^/]+\.edf$"
        )
    )
    sha256: Sha256
    subject_id: Annotated[str, StringConstraints(pattern=r"^P_[0-9]{2}$")]
    session_id: Annotated[str, StringConstraints(pattern=r"^SE[0-9]{3}$")]
    source_partition: SourcePartition
    condition: SpellingCondition
    run_id: Identifier

    @model_validator(mode="after")
    def require_path_alignment(self) -> StudyPSourceFile:
        parts = self.relative_path.split("/")
        expected_partition = self.source_partition.value.title()
        expected_condition = {
            SpellingCondition.PREDICTIVE: "PredictiveSpelling",
            SpellingCondition.NON_PREDICTIVE: "NonpredictiveSpelling",
        }[self.condition]
        if parts[2] != self.subject_id or parts[3] != self.session_id:
            raise ValueError("source subject/session IDs must match the relative path")
        if parts[4] != expected_partition or parts[5] != expected_condition:
            raise ValueError("source partition/condition must match the relative path")
        if parts[-1].removesuffix(".edf") != self.run_id:
            raise ValueError("run ID must equal the EDF filename stem")
        return self


class RecordingKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["bigp3bci-study-p"] = "bigp3bci-study-p"
    source_version: Literal["1.0.0"] = "1.0.0"
    subject_id: Annotated[str, StringConstraints(pattern=r"^P_[0-9]{2}$")]
    session_id: Annotated[str, StringConstraints(pattern=r"^SE[0-9]{3}$")]
    run_id: Identifier
    source_partition: SourcePartition
    condition: SpellingCondition

    @property
    def recording_id(self) -> str:
        return f"{self.subject_id}:{self.session_id}:{self.run_id}"


class ChannelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Identifier
    channel_type: Literal["eeg"] = "eeg"
    unit: Literal["V"] = "V"
    position_m: tuple[float, float, float]


class P300Event(BaseModel):
    """One flash event and its enclosing character-selection trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: Identifier
    selection_trial_id: Identifier
    onset_sample: int = Field(ge=0)
    onset_seconds: float = Field(ge=0.0)
    label: P300Label
    stimulus_code: int | None = Field(default=None, ge=0)
    current_target: int | None = Field(default=None, ge=0)
    selected_target: int | None = Field(default=None, ge=0)


class RecordingProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: str = Field(min_length=1, max_length=2048)
    source_relative_path: str = Field(min_length=1, max_length=2048)
    source_sha256: Sha256
    checksum_manifest_url: str = Field(min_length=1, max_length=2048)
    checksum_manifest_sha256: Sha256
    license_id: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    doi: Literal["10.13026/0byy-ry86"] = "10.13026/0byy-ry86"


class RecordingMetadata(BaseModel):
    """JSON-serializable sidecar for the standardized MNE FIF recording."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    key: RecordingKey
    sampling_rate_hz: float = Field(gt=0.0)
    sample_count: int = Field(ge=1)
    channels: tuple[ChannelMetadata, ...] = Field(min_length=1)
    events: tuple[P300Event, ...] = Field(min_length=1)
    labels_available: bool
    provenance: RecordingProvenance

    @model_validator(mode="after")
    def validate_recording_invariants(self) -> RecordingMetadata:
        names = [channel.name for channel in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("channel names must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")
        samples = [event.onset_sample for event in self.events]
        if samples != sorted(samples) or samples[-1] >= self.sample_count:
            raise ValueError("events must be sorted and inside the recording")
        for event in self.events:
            expected_seconds = event.onset_sample / self.sampling_rate_hz
            if abs(event.onset_seconds - expected_seconds) > 1e-9:
                raise ValueError("event seconds must agree with sample index and sampling rate")
        labels = {event.label for event in self.events}
        expected = (
            {P300Label.NON_TARGET, P300Label.TARGET}
            if self.labels_available
            else {P300Label.UNKNOWN}
        )
        if labels != expected:
            raise ValueError(
                "labeled recordings require both classes; unlabeled recordings require "
                "only unknown labels"
            )
        return self


class PreprocessingConfig(BaseModel):
    """Pinned, CPU-compatible ERP preprocessing recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    low_cut_hz: float = Field(default=0.5, gt=0.0)
    high_cut_hz: float = Field(default=20.0, gt=0.0)
    notch_hz: float | None = Field(default=60.0, gt=0.0)
    reference: Literal["average"] = "average"
    epoch_start_seconds: float = Field(default=-0.1, ge=-1.0, lt=0.0)
    epoch_end_seconds: float = Field(default=0.8, gt=0.0, le=2.0)
    baseline_seconds: tuple[float, float] = (-0.1, 0.0)
    output_sampling_rate_hz: float = Field(default=128.0, gt=0.0)
    reject_peak_to_peak_v: float = Field(default=150e-6, gt=0.0)
    flat_peak_to_peak_v: float = Field(default=0.5e-6, ge=0.0)

    @model_validator(mode="after")
    def validate_signal_ranges(self) -> PreprocessingConfig:
        if self.low_cut_hz >= self.high_cut_hz:
            raise ValueError("low-cut frequency must be below high-cut frequency")
        if self.high_cut_hz >= self.output_sampling_rate_hz / 2:
            raise ValueError("high-cut frequency must be below the output Nyquist frequency")
        baseline_start, baseline_end = self.baseline_seconds
        if not (
            self.epoch_start_seconds <= baseline_start < baseline_end <= 0.0
            and baseline_end <= self.epoch_end_seconds
        ):
            raise ValueError("baseline must be an ordered interval inside the pre-stimulus epoch")
        if self.flat_peak_to_peak_v >= self.reject_peak_to_peak_v:
            raise ValueError("flat threshold must be below the rejection threshold")
        return self


class EpochMetadata(BaseModel):
    """Leakage boundary and provenance for one accepted P300 epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch_id: Identifier
    event_id: Identifier
    selection_trial_id: Identifier
    recording_id: Identifier
    subject_id: Annotated[str, StringConstraints(pattern=r"^P_[0-9]{2}$")]
    session_id: Annotated[str, StringConstraints(pattern=r"^SE[0-9]{3}$")]
    label: P300Label
    onset_sample: int = Field(ge=0)
    onset_seconds: float | None = Field(default=None, ge=0.0)
    stimulus_code: int | None = Field(default=None, ge=0)
    current_target: int | None = Field(default=None, ge=0)
    selected_target: int | None = Field(default=None, ge=0)


class RejectedEpoch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: Identifier
    reasons: tuple[Identifier, ...] = Field(min_length=1)


class PreprocessingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recording_id: Identifier
    input_event_count: int = Field(ge=1)
    accepted_event_count: int = Field(ge=0)
    rejected_epochs: tuple[RejectedEpoch, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> PreprocessingReport:
        if self.accepted_event_count + len(self.rejected_epochs) != self.input_event_count:
            raise ValueError("accepted and rejected events must account for every input event")
        return self


class SubjectSplit(BaseModel):
    """Primary cross-subject split; source Train/Test folder names are ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    seed: int = Field(ge=0)
    train_subjects: tuple[str, ...] = Field(min_length=1)
    validation_subjects: tuple[str, ...] = Field(min_length=1)
    test_subjects: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disjoint_subjects(self) -> SubjectSplit:
        partitions = (
            set(self.train_subjects),
            set(self.validation_subjects),
            set(self.test_subjects),
        )
        if any(not all(subject.startswith("P_") for subject in group) for group in partitions):
            raise ValueError("Study P subject IDs must start with P_")
        if any(
            left & right
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ):
            raise ValueError("subject split partitions must be disjoint")
        return self


class SessionFold(BaseModel):
    """One within-subject, cross-session drift fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold_id: Identifier
    train_sessions: tuple[str, ...] = Field(min_length=1)
    test_sessions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disjoint_sessions(self) -> SessionFold:
        if set(self.train_sessions) & set(self.test_sessions):
            raise ValueError("session fold train and test sessions must be disjoint")
        return self
