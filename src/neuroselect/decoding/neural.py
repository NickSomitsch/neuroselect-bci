"""Deterministic EEGNet training, temperature calibration, and head-only adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit
from torch import Tensor, nn

from neuroselect.decoding.classical import (
    _combine_batches,
    _EpochCollection,
    _labeled,
    _require_binary_labels,
    evaluate_decoder,
)
from neuroselect.decoding.models import (
    ChronologicalDriftReport,
    EEGNetConfig,
    EEGNetTrainingSummary,
    SubjectAdaptationSummary,
    SubjectDriftEvaluation,
)
from neuroselect.eeg import DataSplit, EpochBatch, PreprocessingConfig, SessionFold
from neuroselect.eeg.splits import validate_split_integrity

DEFAULT_EEGNET_CONFIG = Path("configs/decoding/eegnet.yaml")


class InsufficientAdaptationDataError(ValueError):
    """Raised when chronological calibration cannot support a safe subject adapter."""


def load_eegnet_config(path: str | Path = DEFAULT_EEGNET_CONFIG) -> EEGNetConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("EEGNet configuration must contain a YAML mapping")
    return EEGNetConfig.model_validate(payload)


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("EEGNet configuration requested MPS but it is unavailable")
        return torch.device("mps")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class EEGNetFeatureExtractor(nn.Module):
    """Compact EEGNet feature stack with temporal, spatial, and separable convolutions."""

    def __init__(self, channel_count: int, config: EEGNetConfig) -> None:
        super().__init__()
        spatial_filters = config.temporal_filters * config.depth_multiplier
        self.layers = nn.Sequential(
            nn.Conv2d(
                1,
                config.temporal_filters,
                kernel_size=(1, config.temporal_kernel_samples),
                padding=(0, config.temporal_kernel_samples // 2),
                bias=False,
            ),
            nn.BatchNorm2d(config.temporal_filters),
            nn.Conv2d(
                config.temporal_filters,
                spatial_filters,
                kernel_size=(channel_count, 1),
                groups=config.temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, config.first_pool_size)),
            nn.Dropout(config.dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, config.separable_kernel_samples),
                padding=(0, config.separable_kernel_samples // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, config.pointwise_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(config.pointwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, config.second_pool_size)),
            nn.Dropout(config.dropout),
        )

    def forward(self, epochs: Tensor) -> Tensor:
        return torch.flatten(self.layers(epochs), start_dim=1)


class EEGNetBinaryModel(nn.Module):
    """EEGNet feature extractor and replaceable binary linear output head."""

    input_mean: Tensor
    input_scale: Tensor

    def __init__(
        self,
        channel_count: int,
        sample_count: int,
        config: EEGNetConfig,
        input_mean: npt.NDArray[np.float32],
        input_scale: npt.NDArray[np.float32],
    ) -> None:
        super().__init__()
        self.feature_extractor = EEGNetFeatureExtractor(channel_count, config)
        self.register_buffer(
            "input_mean",
            torch.from_numpy(input_mean).reshape(1, 1, channel_count, 1),
        )
        self.register_buffer(
            "input_scale",
            torch.from_numpy(input_scale).reshape(1, 1, channel_count, 1),
        )
        with torch.no_grad():
            feature_count = self.feature_extractor(
                torch.zeros((1, 1, channel_count, sample_count), dtype=torch.float32)
            ).shape[1]
        self.classifier = nn.Linear(feature_count, 1)

    def extract_features(self, epochs: Tensor) -> Tensor:
        normalized = (epochs.unsqueeze(1) - self.input_mean) / self.input_scale
        return cast(Tensor, self.feature_extractor(normalized))

    def forward(self, epochs: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.extract_features(epochs))).squeeze(1)


def _input_normalization(data: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], ...]:
    mean = np.asarray(np.mean(data, axis=(0, 2)), dtype=np.float32)
    scale = np.asarray(np.std(data, axis=(0, 2)), dtype=np.float32)
    scale = np.maximum(scale, np.finfo(np.float32).eps)
    return mean, scale


def _class_weight(labels: npt.NDArray[np.int8]) -> float:
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    return negative / positive


def _tensor(data: npt.NDArray[np.float32], device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(data)).to(device=device, dtype=torch.float32)


def _labels(labels: npt.NDArray[np.int8], device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(labels)).to(device=device, dtype=torch.float32)


def _binary_loss(logits: Tensor, labels: Tensor, positive_weight: float) -> Tensor:
    return functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
    )


def _state_copy(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _fit_model(
    model: EEGNetBinaryModel,
    training_data: npt.NDArray[np.float32],
    training_labels: npt.NDArray[np.int8],
    validation_data: npt.NDArray[np.float32],
    validation_labels: npt.NDArray[np.int8],
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    parameters: Sequence[nn.Parameter] | None = None,
    freeze_features: bool = False,
) -> tuple[int, float]:
    train_x = _tensor(training_data, device)
    train_y = _labels(training_labels, device)
    validation_x = _tensor(validation_data, device)
    validation_y = _labels(validation_labels, device)
    optimized = tuple(model.parameters()) if parameters is None else tuple(parameters)
    optimizer = torch.optim.AdamW(
        optimized,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positive_weight = _class_weight(training_labels)
    generator = np.random.default_rng(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state = _state_copy(model)
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        if freeze_features:
            model.feature_extractor.eval()
        permutation = generator.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            index = torch.from_numpy(indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _binary_loss(model(train_x[index]), train_y[index], positive_weight)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _binary_loss(model(validation_x), validation_y, positive_weight).cpu()
            )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_copy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, best_loss


def _model_logits(
    model: EEGNetBinaryModel,
    data: npt.NDArray[np.float32],
    device: torch.device,
) -> npt.NDArray[np.float64]:
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(data, device)).detach().cpu().numpy()
    return np.asarray(logits, dtype=np.float64)


def _fit_temperature(
    logits: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int8],
    config: EEGNetConfig,
) -> float:
    lower = math.log(config.minimum_temperature)
    upper = math.log(config.maximum_temperature)

    def objective(log_temperature: float) -> float:
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("temperature calibration did not converge")
    return float(math.exp(result.x))


def _development_groups(collections: Sequence[_EpochCollection]) -> dict[str, frozenset[str]]:
    metadata = tuple(item for collection in collections for item in collection.metadata)
    return {
        "epoch": frozenset(item.epoch_id for item in metadata),
        "selection_trial": frozenset(item.selection_trial_id for item in metadata),
        "recording": frozenset(item.recording_id for item in metadata),
        "subject": frozenset(item.subject_id for item in metadata),
    }


class EEGNetP300Decoder:
    """Fitted EEGNet checkpoint with a held-data temperature scaler."""

    def __init__(
        self,
        *,
        config: EEGNetConfig,
        model: EEGNetBinaryModel,
        temperature: float,
        channel_names: tuple[str, ...],
        sampling_rate_hz: float,
        epoch_sample_count: int,
        preprocessing_config: PreprocessingConfig,
        development_groups: dict[str, frozenset[str]],
        require_subject_disjoint_evaluation: bool,
    ) -> None:
        self.config = config
        self.model = model.cpu().eval()
        self.temperature = temperature
        self.channel_names = channel_names
        self.sampling_rate_hz = sampling_rate_hz
        self.epoch_sample_count = epoch_sample_count
        self.preprocessing_config = preprocessing_config
        self.development_groups = development_groups
        self.require_subject_disjoint_evaluation = require_subject_disjoint_evaluation

    @property
    def decision_threshold(self) -> float:
        return self.config.decision_threshold

    @property
    def calibration_bins(self) -> int:
        return self.config.calibration_bins

    def predict_probabilities(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        epochs = np.asarray(data, dtype=np.float32)
        expected = (len(self.channel_names), self.epoch_sample_count)
        if epochs.ndim != 3 or epochs.shape[1:] != expected:
            raise ValueError(
                "decoder input must have shape (epochs, channels, samples) matching training"
            )
        if not np.isfinite(epochs).all():
            raise ValueError("decoder input contains non-finite values")
        logits = _model_logits(self.model, epochs, torch.device("cpu")) / self.temperature
        return np.asarray(expit(logits), dtype=np.float64)

    def validate_collection(self, collection: _EpochCollection) -> None:
        if collection.channel_names != self.channel_names:
            raise ValueError("evaluation channel order does not match the EEGNet checkpoint")
        if collection.data.shape[2] != self.epoch_sample_count:
            raise ValueError("evaluation epoch length does not match the EEGNet checkpoint")
        if not np.isclose(collection.sampling_rate_hz, self.sampling_rate_hz):
            raise ValueError("evaluation sampling rate does not match the EEGNet checkpoint")
        if collection.preprocessing_config != self.preprocessing_config:
            raise ValueError("evaluation preprocessing config does not match the EEGNet checkpoint")
        metadata_groups = {
            "epoch": {item.epoch_id for item in collection.metadata},
            "selection_trial": {item.selection_trial_id for item in collection.metadata},
            "recording": {item.recording_id for item in collection.metadata},
            "subject": {item.subject_id for item in collection.metadata},
        }
        checked: tuple[str, ...] = ("epoch", "selection_trial", "recording")
        if self.require_subject_disjoint_evaluation:
            checked += ("subject",)
        for group_name in checked:
            overlap = metadata_groups[group_name] & self.development_groups[group_name]
            if overlap:
                raise ValueError(
                    f"evaluation {group_name} overlaps EEGNet development data: "
                    f"{sorted(overlap)[:3]}"
                )


def fit_eegnet_decoder(
    training_batches: Sequence[EpochBatch],
    calibration_batches: Sequence[EpochBatch],
    config: EEGNetConfig | None = None,
) -> tuple[EEGNetP300Decoder, EEGNetTrainingSummary]:
    """Fit EEGNet on train subjects and temperature-scale it on validation subjects."""

    recipe = config or load_eegnet_config()
    _seed_training(recipe.random_seed)
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
        raise ValueError("EEGNet training and calibration tensor contracts must match")
    training_data, training_labels = _labeled(training)
    calibration_data, calibration_labels = _labeled(calibration)
    _require_binary_labels(training_labels, "EEGNet fitting")
    _require_binary_labels(calibration_labels, "EEGNet calibration")
    mean, scale = _input_normalization(training_data)
    model = EEGNetBinaryModel(
        len(training.channel_names),
        training.data.shape[2],
        recipe,
        mean,
        scale,
    )
    device = _resolve_device(recipe.device)
    model.to(device)
    selected_epoch, validation_loss = _fit_model(
        model,
        training_data,
        training_labels,
        calibration_data,
        calibration_labels,
        device=device,
        seed=recipe.random_seed,
        batch_size=recipe.batch_size,
        max_epochs=recipe.max_epochs,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        patience=recipe.early_stopping_patience,
        min_delta=recipe.early_stopping_min_delta,
    )
    calibration_logits = _model_logits(model, calibration_data, device)
    temperature = _fit_temperature(calibration_logits, calibration_labels, recipe)
    decoder = EEGNetP300Decoder(
        config=recipe,
        model=model,
        temperature=temperature,
        channel_names=training.channel_names,
        sampling_rate_hz=training.sampling_rate_hz,
        epoch_sample_count=training.data.shape[2],
        preprocessing_config=training.preprocessing_config,
        development_groups=_development_groups((training, calibration)),
        require_subject_disjoint_evaluation=recipe.require_subject_disjoint,
    )
    training_device: Literal["cpu", "mps"] = "mps" if device.type == "mps" else "cpu"
    summary = EEGNetTrainingSummary(
        model_revision=recipe.model_revision,
        config_sha256=recipe.digest(),
        training_dataset_sha256=training.dataset_sha256,
        calibration_dataset_sha256=calibration.dataset_sha256,
        training_epoch_count=len(training_labels),
        calibration_epoch_count=len(calibration_labels),
        excluded_unknown_training_count=int(np.sum(training.labels < 0)),
        excluded_unknown_calibration_count=int(np.sum(calibration.labels < 0)),
        training_subject_ids=tuple(sorted({item.subject_id for item in training.metadata})),
        calibration_subject_ids=tuple(sorted({item.subject_id for item in calibration.metadata})),
        channel_names=training.channel_names,
        sampling_rate_hz=training.sampling_rate_hz,
        epoch_sample_count=training.data.shape[2],
        preprocessing_config=training.preprocessing_config,
        selected_epoch=selected_epoch,
        validation_loss=validation_loss,
        temperature=temperature,
        training_device=training_device,
    )
    return decoder, summary


def _subset_collection(collection: _EpochCollection, indices: Sequence[int]) -> _EpochCollection:
    selected = np.asarray(indices, dtype=np.int64)
    data = np.ascontiguousarray(collection.data[selected], dtype=np.float32)
    labels = np.asarray(collection.labels[selected], dtype=np.int8)
    metadata = tuple(collection.metadata[index] for index in indices)
    digest = hashlib.sha256()
    digest.update(data.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    digest.update(
        json.dumps(
            [item.model_dump(mode="json") for item in metadata],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return _EpochCollection(
        data=data,
        labels=labels,
        metadata=metadata,
        channel_names=collection.channel_names,
        sampling_rate_hz=collection.sampling_rate_hz,
        preprocessing_config=collection.preprocessing_config,
        dataset_sha256=digest.hexdigest(),
    )


def _chronological_trial_split(
    collection: _EpochCollection,
    config: EEGNetConfig,
) -> tuple[_EpochCollection, _EpochCollection]:
    trial_times: dict[str, float] = {}
    for index, item in enumerate(collection.metadata):
        if collection.labels[index] < 0:
            continue
        onset = item.onset_seconds if item.onset_seconds is not None else float(item.onset_sample)
        trial_times[item.selection_trial_id] = min(
            trial_times.get(item.selection_trial_id, math.inf), onset
        )
    ordered_trials = tuple(sorted(trial_times, key=lambda item: (trial_times[item], item)))
    if len(ordered_trials) < config.minimum_adaptation_trials:
        raise InsufficientAdaptationDataError(
            "subject adaptation has insufficient chronological selection trials: "
            f"{len(ordered_trials)} < {config.minimum_adaptation_trials}"
        )
    boundary = min(
        len(ordered_trials) - 1,
        max(1, int(len(ordered_trials) * config.adaptation_head_fraction)),
    )
    head_trials = set(ordered_trials[:boundary])
    head_indices = [
        index
        for index, item in enumerate(collection.metadata)
        if item.selection_trial_id in head_trials and collection.labels[index] >= 0
    ]
    calibration_indices = [
        index
        for index, item in enumerate(collection.metadata)
        if item.selection_trial_id not in head_trials and collection.labels[index] >= 0
    ]
    head = _subset_collection(collection, head_indices)
    calibration = _subset_collection(collection, calibration_indices)
    try:
        _require_binary_labels(head.labels, "subject head adaptation")
        _require_binary_labels(calibration.labels, "subject temperature calibration")
    except ValueError as error:
        raise InsufficientAdaptationDataError(str(error)) from error
    return head, calibration


def _feature_extractor_digest(model: EEGNetBinaryModel) -> str:
    digest = hashlib.sha256()
    state: Mapping[str, Tensor] = model.state_dict()
    for name in sorted(state):
        if name.startswith("classifier."):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def adapt_eegnet_head(
    base_decoder: EEGNetP300Decoder,
    source_batches: Sequence[EpochBatch],
    *,
    target_session_id: str = "SE002",
) -> tuple[EEGNetP300Decoder, SubjectAdaptationSummary]:
    """Freeze EEGNet features and fit a subject head plus temperature chronologically."""

    source = _combine_batches(source_batches)
    base_decoder.validate_collection(source)
    subject_ids = {item.subject_id for item in source.metadata}
    source_sessions = {item.session_id for item in source.metadata}
    if len(subject_ids) != 1 or len(source_sessions) != 1:
        raise ValueError("subject adaptation requires exactly one subject and source session")
    source_session = next(iter(source_sessions))
    if source_session == target_session_id:
        raise ValueError("adaptation source and target sessions must differ")
    head, calibration = _chronological_trial_split(source, base_decoder.config)
    recipe = base_decoder.config
    seed = recipe.random_seed + int(next(iter(subject_ids)).removeprefix("P_"))
    _seed_training(seed)
    model = copy.deepcopy(base_decoder.model)
    before = _feature_extractor_digest(model)
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False
    model.input_mean.requires_grad = False
    model.input_scale.requires_grad = False
    device = _resolve_device(recipe.device)
    model.to(device)
    selected_epoch, validation_loss = _fit_model(
        model,
        head.data,
        head.labels,
        calibration.data,
        calibration.labels,
        device=device,
        seed=seed,
        batch_size=recipe.batch_size,
        max_epochs=recipe.adaptation_max_epochs,
        learning_rate=recipe.adaptation_learning_rate,
        weight_decay=recipe.weight_decay,
        patience=recipe.adaptation_patience,
        min_delta=recipe.early_stopping_min_delta,
        parameters=tuple(model.classifier.parameters()),
        freeze_features=True,
    )
    temperature = _fit_temperature(
        _model_logits(model, calibration.data, device), calibration.labels, recipe
    )
    after = _feature_extractor_digest(model)
    groups = {
        name: base_decoder.development_groups[name] | values
        for name, values in _development_groups((source,)).items()
    }
    adapted = EEGNetP300Decoder(
        config=recipe,
        model=model,
        temperature=temperature,
        channel_names=source.channel_names,
        sampling_rate_hz=source.sampling_rate_hz,
        epoch_sample_count=source.data.shape[2],
        preprocessing_config=source.preprocessing_config,
        development_groups=groups,
        require_subject_disjoint_evaluation=False,
    )
    summary = SubjectAdaptationSummary(
        adapter_revision=recipe.adapter_revision,
        subject_id=next(iter(subject_ids)),
        source_session_id=source_session,
        target_session_id=target_session_id,
        head_dataset_sha256=head.dataset_sha256,
        calibration_dataset_sha256=calibration.dataset_sha256,
        head_epoch_count=len(head.labels),
        calibration_epoch_count=len(calibration.labels),
        head_trial_count=len({item.selection_trial_id for item in head.metadata}),
        calibration_trial_count=len({item.selection_trial_id for item in calibration.metadata}),
        excluded_unknown_count=int(np.sum(source.labels < 0)),
        temperature=temperature,
        selected_epoch=selected_epoch,
        validation_loss=validation_loss,
        feature_extractor_sha256_before=before,
        feature_extractor_sha256_after=after,
        trained_parameters=("classifier.weight", "classifier.bias", "temperature"),
    )
    return adapted, summary


def evaluate_chronological_session_drift(
    base_decoder: EEGNetP300Decoder,
    batches: Sequence[EpochBatch],
    fold: SessionFold | None = None,
) -> ChronologicalDriftReport:
    """Adapt each held-out subject on SE001 and evaluate untouched SE002 epochs."""

    protocol = fold or SessionFold(
        fold_id="study-p-se001-to-se002",
        train_sessions=("SE001",),
        test_sessions=("SE002",),
    )
    if protocol.train_sessions != ("SE001",) or protocol.test_sessions != ("SE002",):
        raise ValueError("primary chronological drift requires the SE001-to-SE002 fold")
    grouped: dict[str, list[EpochBatch]] = {}
    for batch in batches:
        subjects = {item.subject_id for item in batch.metadata}
        if len(subjects) != 1:
            raise ValueError("chronological drift batches must contain exactly one subject")
        grouped.setdefault(next(iter(subjects)), []).append(batch)
    results: list[SubjectDriftEvaluation] = []
    for subject_id, subject_batches in sorted(grouped.items()):
        source = [
            batch
            for batch in subject_batches
            if {item.session_id for item in batch.metadata} == {"SE001"}
        ]
        target = [
            batch
            for batch in subject_batches
            if {item.session_id for item in batch.metadata} == {"SE002"}
        ]
        if not source or not target or len(source) + len(target) != len(subject_batches):
            raise ValueError(
                f"subject {subject_id} must contain only non-empty SE001 and SE002 batches"
            )
        fallback_reason: str | None
        try:
            adapted, adaptation = adapt_eegnet_head(base_decoder, source, target_session_id="SE002")
        except InsufficientAdaptationDataError as error:
            adaptation = None
            fallback_reason = str(error)
        else:
            fallback_reason = None
        subject_independent = evaluate_decoder(base_decoder, target)
        if adaptation is None:
            adapted_evaluation = subject_independent
        else:
            adapted_evaluation = evaluate_decoder(adapted, target)
        if subject_independent.metrics is None or adapted_evaluation.metrics is None:
            raise ValueError("chronological drift target sessions require both labeled classes")
        results.append(
            SubjectDriftEvaluation(
                subject_id=subject_id,
                fold=protocol,
                adaptation=adaptation,
                fallback_reason=fallback_reason,
                conservative_abstention_required=adaptation is None,
                subject_independent=subject_independent,
                adapted=adapted_evaluation,
            )
        )
    if not results:
        raise ValueError("chronological drift requires at least one held-out subject")
    auroc_deltas = [
        item.adapted.metrics.auroc - item.subject_independent.metrics.auroc
        for item in results
        if item.adapted.metrics is not None and item.subject_independent.metrics is not None
    ]
    brier_deltas = [
        item.adapted.metrics.brier_score - item.subject_independent.metrics.brier_score
        for item in results
        if item.adapted.metrics is not None and item.subject_independent.metrics is not None
    ]
    return ChronologicalDriftReport(
        config_sha256=base_decoder.config.digest(),
        fold=protocol,
        subjects=tuple(results),
        mean_auroc_delta=float(np.mean(auroc_deltas)),
        mean_brier_delta=float(np.mean(brier_deltas)),
        adapted_subject_count=sum(item.adaptation is not None for item in results),
        fallback_subject_count=sum(item.adaptation is None for item in results),
    )
