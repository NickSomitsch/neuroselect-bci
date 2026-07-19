"""Typed configuration, predictions, and metrics for the classical P300 baseline."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.eeg import P300Label, PreprocessingConfig, SessionFold


class ClassicalDecoderConfig(BaseModel):
    """Locked xDAWN, shrinkage-LDA, and held-validation calibration recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    model_revision: Literal["xdawn-shrinkage-lda-platt-v1"] = "xdawn-shrinkage-lda-platt-v1"
    random_seed: int = Field(default=20260719, ge=0)
    xdawn_components: int = Field(default=2, ge=1, le=8)
    xdawn_regularization: float = Field(default=0.1, ge=0.0, le=1.0)
    lda_shrinkage: Literal["auto"] = "auto"
    calibration_c: float = Field(default=1.0, gt=0.0, le=1_000.0)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    require_subject_disjoint: bool = True

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class DecoderTrainingSummary(BaseModel):
    """Data boundary and exclusion counts used to fit one checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_revision: str = Field(min_length=1, max_length=160)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_epoch_count: int = Field(ge=1)
    calibration_epoch_count: int = Field(ge=1)
    excluded_unknown_training_count: int = Field(ge=0)
    excluded_unknown_calibration_count: int = Field(ge=0)
    training_subject_ids: tuple[str, ...] = Field(min_length=1)
    calibration_subject_ids: tuple[str, ...] = Field(min_length=1)
    channel_names: tuple[str, ...] = Field(min_length=1)
    sampling_rate_hz: float = Field(gt=0.0)
    epoch_sample_count: int = Field(ge=2)
    preprocessing_config: PreprocessingConfig


class EpochPrediction(BaseModel):
    """One calibrated target probability retaining original event provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch_id: str = Field(min_length=1, max_length=200)
    event_id: str = Field(min_length=1, max_length=200)
    selection_trial_id: str = Field(min_length=1, max_length=200)
    recording_id: str = Field(min_length=1, max_length=200)
    subject_id: str = Field(min_length=1, max_length=32)
    session_id: str = Field(min_length=1, max_length=32)
    true_label: P300Label
    target_probability: float = Field(ge=0.0, le=1.0)
    predicted_target: bool
    onset_seconds: float | None = Field(default=None, ge=0.0)
    stimulus_code: int | None = Field(default=None, ge=0)


class BinaryDecoderMetrics(BaseModel):
    """Original-task labeled-epoch metrics; unknown events are never included."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    auroc: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    negative_log_likelihood: float = Field(ge=0.0)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)


class DecoderEvaluation(BaseModel):
    """Predictions and metrics for a leakage-safe held-out epoch collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions: tuple[EpochPrediction, ...] = Field(min_length=1)
    labeled_epoch_count: int = Field(ge=0)
    unknown_epoch_count: int = Field(ge=0)
    metrics: BinaryDecoderMetrics | None = None
    selection_trial_count: int = Field(default=0, ge=0)
    selection_code_set_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> DecoderEvaluation:
        if self.labeled_epoch_count + self.unknown_epoch_count != len(self.predictions):
            raise ValueError("labeled and unknown counts must cover every prediction")
        if (self.labeled_epoch_count == 0) != (self.metrics is None):
            raise ValueError("metrics are present exactly when labeled epochs are present")
        if (self.selection_trial_count == 0) != (self.selection_code_set_accuracy is None):
            raise ValueError("selection accuracy requires at least one scorable trial")
        return self


class DecoderCheckpointMetadata(BaseModel):
    """Safe JSON metadata stored beside the trusted local joblib checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    config: ClassicalDecoderConfig
    training_summary: DecoderTrainingSummary

    @model_validator(mode="after")
    def validate_config_alignment(self) -> DecoderCheckpointMetadata:
        if (
            self.training_summary.config_sha256 != self.config.digest()
            or self.training_summary.model_revision != self.config.model_revision
        ):
            raise ValueError("decoder configuration and training summary do not agree")
        return self


class EEGNetConfig(BaseModel):
    """Locked compact EEGNet training, calibration, and adaptation recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    model_revision: Literal["eegnet-temperature-v1"] = "eegnet-temperature-v1"
    adapter_revision: Literal["eegnet-linear-head-temperature-v1"] = (
        "eegnet-linear-head-temperature-v1"
    )
    random_seed: int = Field(default=20260720, ge=0)
    device: Literal["auto", "cpu", "mps"] = "auto"
    temporal_filters: int = Field(default=8, ge=1, le=64)
    depth_multiplier: int = Field(default=2, ge=1, le=8)
    pointwise_filters: int = Field(default=16, ge=1, le=128)
    temporal_kernel_samples: int = Field(default=31, ge=3, le=255)
    separable_kernel_samples: int = Field(default=15, ge=3, le=127)
    first_pool_size: int = Field(default=4, ge=1, le=16)
    second_pool_size: int = Field(default=4, ge=1, le=16)
    dropout: float = Field(default=0.25, ge=0.0, lt=1.0)
    batch_size: int = Field(default=64, ge=2, le=1024)
    max_epochs: int = Field(default=100, ge=1, le=2_000)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(default=12, ge=1, le=200)
    early_stopping_min_delta: float = Field(default=1e-5, ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    calibration_bins: int = Field(default=10, ge=2, le=50)
    adaptation_head_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    minimum_adaptation_trials: int = Field(default=4, ge=2, le=1_000)
    adaptation_max_epochs: int = Field(default=100, ge=1, le=2_000)
    adaptation_learning_rate: float = Field(default=5e-3, gt=0.0, le=1.0)
    adaptation_patience: int = Field(default=12, ge=1, le=200)
    minimum_temperature: float = Field(default=0.05, gt=0.0, lt=1.0)
    maximum_temperature: float = Field(default=10.0, gt=1.0, le=100.0)
    require_subject_disjoint: bool = True

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class EEGNetTrainingSummary(BaseModel):
    """Data, tensor, and optimization provenance for a subject-independent EEGNet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_revision: str = Field(min_length=1, max_length=160)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_epoch_count: int = Field(ge=1)
    calibration_epoch_count: int = Field(ge=1)
    excluded_unknown_training_count: int = Field(ge=0)
    excluded_unknown_calibration_count: int = Field(ge=0)
    training_subject_ids: tuple[str, ...] = Field(min_length=1)
    calibration_subject_ids: tuple[str, ...] = Field(min_length=1)
    channel_names: tuple[str, ...] = Field(min_length=1)
    sampling_rate_hz: float = Field(gt=0.0)
    epoch_sample_count: int = Field(ge=2)
    preprocessing_config: PreprocessingConfig
    selected_epoch: int = Field(ge=1)
    validation_loss: float = Field(ge=0.0)
    temperature: float = Field(gt=0.0)
    training_device: Literal["cpu", "mps"]


class EEGNetCheckpointMetadata(BaseModel):
    """Safe metadata used to reconstruct an EEGNet state-dict checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    config: EEGNetConfig
    training_summary: EEGNetTrainingSummary
    development_groups: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_config_alignment(self) -> EEGNetCheckpointMetadata:
        if (
            self.training_summary.config_sha256 != self.config.digest()
            or self.training_summary.model_revision != self.config.model_revision
        ):
            raise ValueError("EEGNet configuration and training summary do not agree")
        if set(self.development_groups) != {
            "epoch",
            "selection_trial",
            "recording",
            "subject",
        }:
            raise ValueError("EEGNet metadata must include every development leakage group")
        return self


class SubjectAdaptationSummary(BaseModel):
    """Proof that one subject adapter used only an earlier calibration session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adapter_revision: str = Field(min_length=1, max_length=160)
    subject_id: str = Field(pattern=r"^P_[0-9]{2}$")
    source_session_id: str = Field(pattern=r"^SE[0-9]{3}$")
    target_session_id: str = Field(pattern=r"^SE[0-9]{3}$")
    head_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    head_epoch_count: int = Field(ge=1)
    calibration_epoch_count: int = Field(ge=1)
    head_trial_count: int = Field(ge=1)
    calibration_trial_count: int = Field(ge=1)
    excluded_unknown_count: int = Field(ge=0)
    temperature: float = Field(gt=0.0)
    selected_epoch: int = Field(ge=1)
    validation_loss: float = Field(ge=0.0)
    feature_extractor_sha256_before: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_extractor_sha256_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    trained_parameters: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_frozen_features(self) -> SubjectAdaptationSummary:
        if self.source_session_id == self.target_session_id:
            raise ValueError("adaptation source and drift target sessions must differ")
        if self.feature_extractor_sha256_before != self.feature_extractor_sha256_after:
            raise ValueError("subject adaptation must not modify EEGNet feature layers")
        if set(self.trained_parameters) != {"classifier.weight", "classifier.bias", "temperature"}:
            raise ValueError("subject adaptation may train only the linear head and temperature")
        return self


class SubjectDriftEvaluation(BaseModel):
    """Base and adapted evaluations for one chronological subject/session pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject_id: str = Field(pattern=r"^P_[0-9]{2}$")
    fold: SessionFold
    adaptation: SubjectAdaptationSummary | None = None
    fallback_reason: str | None = Field(default=None, min_length=1, max_length=500)
    conservative_abstention_required: bool = False
    subject_independent: DecoderEvaluation
    adapted: DecoderEvaluation

    @model_validator(mode="after")
    def validate_fallback(self) -> SubjectDriftEvaluation:
        if (self.adaptation is None) != (self.fallback_reason is not None):
            raise ValueError("drift result must contain either adaptation or a fallback reason")
        if self.conservative_abstention_required != (self.adaptation is None):
            raise ValueError("only subject-independent fallback requires conservative abstention")
        return self


class ChronologicalDriftReport(BaseModel):
    """Per-subject SE001-to-SE002 drift results kept separate from reverse sensitivity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    protocol_revision: Literal["study-p-chronological-session-drift-v1"] = (
        "study-p-chronological-session-drift-v1"
    )
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold: SessionFold
    subjects: tuple[SubjectDriftEvaluation, ...] = Field(min_length=1)
    mean_auroc_delta: float
    mean_brier_delta: float
    adapted_subject_count: int = Field(ge=0)
    fallback_subject_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_chronology(self) -> ChronologicalDriftReport:
        if self.fold.train_sessions != ("SE001",) or self.fold.test_sessions != ("SE002",):
            raise ValueError("primary chronological drift must adapt on SE001 and test on SE002")
        if any(item.fold != self.fold for item in self.subjects):
            raise ValueError("every subject drift result must use the report fold")
        identifiers = [item.subject_id for item in self.subjects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("chronological drift subjects must be unique")
        if self.adapted_subject_count + self.fallback_subject_count != len(self.subjects):
            raise ValueError("adapted and fallback counts must cover every drift subject")
        return self
