"""Leakage-safe xDAWN, shrinkage-LDA, and held-validation calibration baseline."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from mne.decoding import Vectorizer, XdawnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from neuroselect.decoding.models import (
    BinaryDecoderMetrics,
    ClassicalDecoderConfig,
    DecoderEvaluation,
    DecoderTrainingSummary,
    EpochPrediction,
)
from neuroselect.eeg import (
    DataSplit,
    EpochBatch,
    EpochMetadata,
    P300Label,
    PreprocessingConfig,
)
from neuroselect.eeg.splits import validate_split_integrity

DEFAULT_CLASSICAL_DECODER_CONFIG = Path("configs/decoding/xdawn_lda.yaml")


def load_classical_decoder_config(
    path: str | Path = DEFAULT_CLASSICAL_DECODER_CONFIG,
) -> ClassicalDecoderConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("classical decoder configuration must contain a YAML mapping")
    return ClassicalDecoderConfig.model_validate(payload)


@dataclass(frozen=True)
class _EpochCollection:
    data: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int8]
    metadata: tuple[EpochMetadata, ...]
    channel_names: tuple[str, ...]
    sampling_rate_hz: float
    preprocessing_config: PreprocessingConfig
    dataset_sha256: str


def _combine_batches(batches: Sequence[EpochBatch]) -> _EpochCollection:
    if not batches:
        raise ValueError("at least one epoch batch is required")
    reference = batches[0]
    sample_count = reference.data.shape[2]
    for batch in batches[1:]:
        if batch.channel_names != reference.channel_names:
            raise ValueError("all decoder batches must use the same channel order")
        if batch.data.shape[2] != sample_count:
            raise ValueError("all decoder batches must use the same epoch sample count")
        if not np.isclose(batch.sampling_rate_hz, reference.sampling_rate_hz):
            raise ValueError("all decoder batches must use the same sampling rate")
        if batch.config != reference.config:
            raise ValueError("all decoder batches must use the same preprocessing config")

    rows = [
        (metadata.epoch_id, batch.data[index], int(batch.labels[index]), metadata)
        for batch in batches
        for index, metadata in enumerate(batch.metadata)
    ]
    rows.sort(key=lambda item: item[0])
    epoch_ids = [item[0] for item in rows]
    if len(epoch_ids) != len(set(epoch_ids)):
        raise ValueError("decoder batches contain duplicate epoch IDs")
    data = np.ascontiguousarray(np.stack([item[1] for item in rows]), dtype=np.float32)
    labels = np.asarray([item[2] for item in rows], dtype=np.int8)
    metadata = tuple(item[3] for item in rows)
    digest = hashlib.sha256()
    digest.update(data.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "channel_names": reference.channel_names,
                "sampling_rate_hz": reference.sampling_rate_hz,
                "preprocessing": reference.config.model_dump(mode="json"),
                "epochs": [item.model_dump(mode="json") for item in metadata],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return _EpochCollection(
        data=data,
        labels=labels,
        metadata=metadata,
        channel_names=reference.channel_names,
        sampling_rate_hz=reference.sampling_rate_hz,
        preprocessing_config=reference.config,
        dataset_sha256=digest.hexdigest(),
    )


def _require_binary_labels(labels: npt.NDArray[np.int8], purpose: str) -> None:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{purpose} requires both non-target and target labeled epochs")


def _labeled(collection: _EpochCollection) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    mask = collection.labels >= 0
    return collection.data[mask], collection.labels[mask]


class CalibratedP300Decoder:
    """Fitted local checkpoint; probabilities are calibrated on held-out subjects."""

    def __init__(
        self,
        *,
        config: ClassicalDecoderConfig,
        base_model: Pipeline,
        calibrator: LogisticRegression,
        channel_names: tuple[str, ...],
        sampling_rate_hz: float,
        epoch_sample_count: int,
        preprocessing_config: PreprocessingConfig,
        development_groups: dict[str, frozenset[str]],
    ) -> None:
        self.config = config
        self.base_model = base_model
        self.calibrator = calibrator
        self.channel_names = channel_names
        self.sampling_rate_hz = sampling_rate_hz
        self.epoch_sample_count = epoch_sample_count
        self.preprocessing_config = preprocessing_config
        self.development_groups = development_groups

    def predict_probabilities(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        epochs = np.asarray(data, dtype=np.float32)
        expected = (len(self.channel_names), self.epoch_sample_count)
        if epochs.ndim != 3 or epochs.shape[1:] != expected:
            raise ValueError(
                "decoder input must have shape (epochs, channels, samples) matching training"
            )
        if not np.isfinite(epochs).all():
            raise ValueError("decoder input contains non-finite values")
        scores = np.asarray(self.base_model.decision_function(epochs), dtype=np.float64).reshape(
            -1, 1
        )
        return np.asarray(self.calibrator.predict_proba(scores)[:, 1], dtype=np.float64)

    def validate_collection(self, collection: _EpochCollection) -> None:
        if collection.channel_names != self.channel_names:
            raise ValueError("evaluation channel order does not match the checkpoint")
        if collection.data.shape[2] != self.epoch_sample_count:
            raise ValueError("evaluation epoch length does not match the checkpoint")
        if not np.isclose(collection.sampling_rate_hz, self.sampling_rate_hz):
            raise ValueError("evaluation sampling rate does not match the checkpoint")
        if collection.preprocessing_config != self.preprocessing_config:
            raise ValueError("evaluation preprocessing config does not match the checkpoint")
        metadata_groups = {
            "epoch": {item.epoch_id for item in collection.metadata},
            "selection_trial": {item.selection_trial_id for item in collection.metadata},
            "recording": {item.recording_id for item in collection.metadata},
            "subject": {item.subject_id for item in collection.metadata},
        }
        checked: tuple[str, ...] = ("epoch", "selection_trial", "recording")
        if self.config.require_subject_disjoint:
            checked += ("subject",)
        for group_name in checked:
            overlap = metadata_groups[group_name] & self.development_groups[group_name]
            if overlap:
                raise ValueError(
                    f"evaluation {group_name} overlaps decoder development data: "
                    f"{sorted(overlap)[:3]}"
                )


def fit_calibrated_decoder(
    training_batches: Sequence[EpochBatch],
    calibration_batches: Sequence[EpochBatch],
    config: ClassicalDecoderConfig | None = None,
) -> tuple[CalibratedP300Decoder, DecoderTrainingSummary]:
    """Fit only on labeled training epochs and calibrate only on held validation epochs."""

    recipe = config or load_classical_decoder_config()
    training = _combine_batches(training_batches)
    calibration = _combine_batches(calibration_batches)
    validate_split_integrity(
        {
            DataSplit.TRAIN: training.metadata,
            DataSplit.VALIDATION: calibration.metadata,
            DataSplit.TEST: (),
        },
        require_subject_disjoint=recipe.require_subject_disjoint,
    )
    if (
        training.channel_names != calibration.channel_names
        or training.data.shape[2] != calibration.data.shape[2]
        or not np.isclose(training.sampling_rate_hz, calibration.sampling_rate_hz)
        or training.preprocessing_config != calibration.preprocessing_config
    ):
        raise ValueError("training and calibration tensor contracts must match")
    train_data, train_labels = _labeled(training)
    calibration_data, calibration_labels = _labeled(calibration)
    _require_binary_labels(train_labels, "decoder fitting")
    _require_binary_labels(calibration_labels, "probability calibration")

    base_model = Pipeline(
        steps=(
            (
                "xdawn",
                XdawnTransformer(
                    n_components=recipe.xdawn_components,
                    reg=recipe.xdawn_regularization,
                ),
            ),
            ("vectorize", Vectorizer()),
            (
                "lda",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage=recipe.lda_shrinkage),
            ),
        )
    )
    base_model.fit(train_data, train_labels)
    calibration_scores = np.asarray(
        base_model.decision_function(calibration_data), dtype=np.float64
    ).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=recipe.calibration_c,
        random_state=recipe.random_seed,
        solver="lbfgs",
    )
    calibrator.fit(calibration_scores, calibration_labels)
    development_metadata = (*training.metadata, *calibration.metadata)
    development_groups = {
        "epoch": frozenset(item.epoch_id for item in development_metadata),
        "selection_trial": frozenset(item.selection_trial_id for item in development_metadata),
        "recording": frozenset(item.recording_id for item in development_metadata),
        "subject": frozenset(item.subject_id for item in development_metadata),
    }
    decoder = CalibratedP300Decoder(
        config=recipe,
        base_model=base_model,
        calibrator=calibrator,
        channel_names=training.channel_names,
        sampling_rate_hz=training.sampling_rate_hz,
        epoch_sample_count=training.data.shape[2],
        preprocessing_config=training.preprocessing_config,
        development_groups=development_groups,
    )
    summary = DecoderTrainingSummary(
        model_revision=recipe.model_revision,
        config_sha256=recipe.digest(),
        training_dataset_sha256=training.dataset_sha256,
        calibration_dataset_sha256=calibration.dataset_sha256,
        training_epoch_count=len(train_labels),
        calibration_epoch_count=len(calibration_labels),
        excluded_unknown_training_count=int(np.sum(training.labels < 0)),
        excluded_unknown_calibration_count=int(np.sum(calibration.labels < 0)),
        training_subject_ids=tuple(sorted({item.subject_id for item in training.metadata})),
        calibration_subject_ids=tuple(sorted({item.subject_id for item in calibration.metadata})),
        channel_names=training.channel_names,
        sampling_rate_hz=training.sampling_rate_hz,
        epoch_sample_count=training.data.shape[2],
        preprocessing_config=training.preprocessing_config,
    )
    return decoder, summary


def _expected_calibration_error(
    labels: npt.NDArray[np.int8], probabilities: npt.NDArray[np.float64], bins: int
) -> float:
    total = len(labels)
    result = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            result += (
                float(np.sum(mask))
                / total
                * abs(float(np.mean(labels[mask])) - float(np.mean(probabilities[mask])))
            )
    return result


def _selection_accuracy(predictions: tuple[EpochPrediction, ...]) -> tuple[int, float | None]:
    trials: dict[str, list[EpochPrediction]] = defaultdict(list)
    for prediction in predictions:
        trials[prediction.selection_trial_id].append(prediction)
    outcomes: list[bool] = []
    for trial in trials.values():
        if any(
            item.true_label is P300Label.UNKNOWN or item.stimulus_code is None for item in trial
        ):
            continue
        target_codes = {item.stimulus_code for item in trial if item.true_label is P300Label.TARGET}
        if not target_codes:
            continue
        by_code: dict[int, list[float]] = defaultdict(list)
        for item in trial:
            assert item.stimulus_code is not None
            by_code[item.stimulus_code].append(item.target_probability)
        predicted_codes = {
            code
            for code, _ in sorted(
                by_code.items(),
                key=lambda item: (-float(np.mean(item[1])), item[0]),
            )[: len(target_codes)]
        }
        outcomes.append(predicted_codes == target_codes)
    return len(outcomes), float(np.mean(outcomes)) if outcomes else None


def evaluate_decoder(
    decoder: CalibratedP300Decoder,
    batches: Sequence[EpochBatch],
) -> DecoderEvaluation:
    """Predict every event but compute original-task metrics from known labels only."""

    collection = _combine_batches(batches)
    decoder.validate_collection(collection)
    probabilities = decoder.predict_probabilities(collection.data)
    predictions = tuple(
        EpochPrediction(
            epoch_id=metadata.epoch_id,
            event_id=metadata.event_id,
            selection_trial_id=metadata.selection_trial_id,
            recording_id=metadata.recording_id,
            subject_id=metadata.subject_id,
            session_id=metadata.session_id,
            true_label=metadata.label,
            target_probability=float(probability),
            predicted_target=probability >= decoder.config.decision_threshold,
            onset_seconds=metadata.onset_seconds,
            stimulus_code=metadata.stimulus_code,
        )
        for metadata, probability in zip(collection.metadata, probabilities, strict=True)
    )
    labeled_mask = collection.labels >= 0
    labeled_count = int(np.sum(labeled_mask))
    metrics = None
    if labeled_count:
        labels = collection.labels[labeled_mask]
        _require_binary_labels(labels, "decoder evaluation")
        labeled_probabilities = probabilities[labeled_mask]
        predicted = labeled_probabilities >= decoder.config.decision_threshold
        metrics = BinaryDecoderMetrics(
            auroc=float(roc_auc_score(labels, labeled_probabilities)),
            balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
            brier_score=float(brier_score_loss(labels, labeled_probabilities)),
            negative_log_likelihood=float(log_loss(labels, labeled_probabilities, labels=(0, 1))),
            expected_calibration_error=_expected_calibration_error(
                labels, labeled_probabilities, decoder.config.calibration_bins
            ),
        )
    trial_count, selection_accuracy = _selection_accuracy(predictions)
    return DecoderEvaluation(
        dataset_sha256=collection.dataset_sha256,
        predictions=predictions,
        labeled_epoch_count=labeled_count,
        unknown_epoch_count=len(collection.labels) - labeled_count,
        metrics=metrics,
        selection_trial_count=trial_count,
        selection_code_set_accuracy=selection_accuracy,
    )
