"""Typed configuration, predictions, and metrics for the classical P300 baseline."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.eeg import P300Label, PreprocessingConfig


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
