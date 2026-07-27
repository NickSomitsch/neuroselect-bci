"""Classical P300 decoding, calibration, evaluation, and checkpoint artifacts."""

from neuroselect.decoding.artifacts import read_decoder_artifacts, write_decoder_artifacts
from neuroselect.decoding.classical import (
    CalibratedP300Decoder,
    evaluate_decoder,
    fit_calibrated_decoder,
    load_classical_decoder_config,
    selection_metrics_from_predictions,
)
from neuroselect.decoding.datasets import load_partitioned_epoch_batches
from neuroselect.decoding.models import (
    BinaryDecoderMetrics,
    ChronologicalDriftReport,
    ClassicalDecoderConfig,
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    DecoderTrainingSummary,
    EEGNetCheckpointMetadata,
    EEGNetConfig,
    EEGNetTrainingSummary,
    EpochPrediction,
    SelectionDecoderMetrics,
    SubjectAdaptationSummary,
    SubjectDriftEvaluation,
)
from neuroselect.decoding.neural import (
    EEGNetP300Decoder,
    InsufficientAdaptationDataError,
    adapt_eegnet_head,
    evaluate_chronological_session_drift,
    fit_eegnet_decoder,
    load_eegnet_config,
)
from neuroselect.decoding.neural_artifacts import (
    read_eegnet_artifacts,
    write_eegnet_artifacts,
)

__all__ = [
    "BinaryDecoderMetrics",
    "CalibratedP300Decoder",
    "ChronologicalDriftReport",
    "ClassicalDecoderConfig",
    "DecoderCheckpointMetadata",
    "DecoderEvaluation",
    "DecoderTrainingSummary",
    "EEGNetCheckpointMetadata",
    "EEGNetConfig",
    "EEGNetP300Decoder",
    "EEGNetTrainingSummary",
    "EpochPrediction",
    "InsufficientAdaptationDataError",
    "SelectionDecoderMetrics",
    "SubjectAdaptationSummary",
    "SubjectDriftEvaluation",
    "adapt_eegnet_head",
    "evaluate_chronological_session_drift",
    "evaluate_decoder",
    "fit_calibrated_decoder",
    "fit_eegnet_decoder",
    "load_classical_decoder_config",
    "load_eegnet_config",
    "load_partitioned_epoch_batches",
    "read_decoder_artifacts",
    "read_eegnet_artifacts",
    "selection_metrics_from_predictions",
    "write_decoder_artifacts",
    "write_eegnet_artifacts",
]
