"""Classical P300 decoding, calibration, evaluation, and checkpoint artifacts."""

from neuroselect.decoding.artifacts import read_decoder_artifacts, write_decoder_artifacts
from neuroselect.decoding.classical import (
    CalibratedP300Decoder,
    evaluate_decoder,
    fit_calibrated_decoder,
    load_classical_decoder_config,
)
from neuroselect.decoding.datasets import load_partitioned_epoch_batches
from neuroselect.decoding.models import (
    BinaryDecoderMetrics,
    ClassicalDecoderConfig,
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    DecoderTrainingSummary,
    EpochPrediction,
)

__all__ = [
    "BinaryDecoderMetrics",
    "CalibratedP300Decoder",
    "ClassicalDecoderConfig",
    "DecoderCheckpointMetadata",
    "DecoderEvaluation",
    "DecoderTrainingSummary",
    "EpochPrediction",
    "evaluate_decoder",
    "fit_calibrated_decoder",
    "load_classical_decoder_config",
    "load_partitioned_epoch_batches",
    "read_decoder_artifacts",
    "write_decoder_artifacts",
]
