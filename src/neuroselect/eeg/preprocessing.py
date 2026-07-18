"""MNE-based, artifact-aware P300 preprocessing and epoch validation."""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
import numpy.typing as npt

from neuroselect.eeg.models import (
    EpochMetadata,
    P300Label,
    PreprocessingConfig,
    PreprocessingReport,
    RejectedEpoch,
)
from neuroselect.eeg.study_p import StandardizedRecording


class EpochBatch:
    """Validated decoder-ready tensor with one provenance record per epoch."""

    def __init__(
        self,
        *,
        data: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer],
        channel_names: Sequence[str],
        sampling_rate_hz: float,
        metadata: Sequence[EpochMetadata],
        config: PreprocessingConfig,
        report: PreprocessingReport,
    ) -> None:
        float_data = np.asarray(data, dtype=np.float32)
        integer_labels = np.asarray(labels, dtype=np.int8)
        if float_data.ndim != 3:
            raise ValueError("epoch data must have shape (epochs, channels, samples)")
        if integer_labels.ndim != 1 or len(integer_labels) != len(float_data):
            raise ValueError("labels must contain exactly one value per epoch")
        if float_data.shape[1] != len(channel_names):
            raise ValueError("channel names must match the epoch tensor")
        if len(metadata) != len(float_data):
            raise ValueError("metadata must contain exactly one record per epoch")
        if not np.isfinite(float_data).all():
            raise ValueError("epoch tensor contains non-finite values")
        if not set(integer_labels.tolist()).issubset({-1, 0, 1}):
            raise ValueError("P300 labels must be encoded as unknown=-1, non-target=0, or target=1")
        label_codes = {
            P300Label.UNKNOWN: -1,
            P300Label.NON_TARGET: 0,
            P300Label.TARGET: 1,
        }
        expected_labels = np.asarray([label_codes[item.label] for item in metadata], dtype=np.int8)
        if not np.array_equal(integer_labels, expected_labels):
            raise ValueError("numeric labels must agree with epoch metadata")
        if sampling_rate_hz <= 0:
            raise ValueError("epoch sampling rate must be positive")
        if report.accepted_event_count != len(float_data):
            raise ValueError("preprocessing report must agree with the epoch tensor")

        self.data = float_data
        self.labels = integer_labels
        self.channel_names = tuple(channel_names)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.metadata = tuple(metadata)
        self.config = config
        self.report = report


def preprocess_recording(
    recording: StandardizedRecording,
    config: PreprocessingConfig | None = None,
) -> EpochBatch:
    """Band-limit, reference, baseline, reject artifacts, and resample one recording."""

    recipe = config or PreprocessingConfig()
    raw = recording.raw.copy().load_data()
    if recipe.notch_hz is not None:
        raw.notch_filter(freqs=[recipe.notch_hz], verbose=False)
    raw.filter(l_freq=recipe.low_cut_hz, h_freq=recipe.high_cut_hz, verbose=False)
    raw.set_eeg_reference(recipe.reference, projection=False, verbose=False)

    source_events = recording.metadata.events
    mne_event_codes = {
        P300Label.NON_TARGET: 1,
        P300Label.TARGET: 2,
        P300Label.UNKNOWN: 3,
    }
    events = np.asarray(
        [
            [
                event.onset_sample,
                0,
                mne_event_codes[event.label],
            ]
            for event in source_events
        ],
        dtype=np.int64,
    )
    present_labels = {event.label for event in source_events}
    epochs = mne.Epochs(
        raw,
        events,
        event_id={
            label.value: code for label, code in mne_event_codes.items() if label in present_labels
        },
        tmin=recipe.epoch_start_seconds,
        tmax=recipe.epoch_end_seconds,
        baseline=recipe.baseline_seconds,
        reject={"eeg": recipe.reject_peak_to_peak_v},
        flat={"eeg": recipe.flat_peak_to_peak_v},
        preload=True,
        reject_by_annotation=True,
        event_repeated="error",
        verbose=False,
    )
    accepted_indices = tuple(int(index) for index in epochs.selection)
    if not accepted_indices:
        raise ValueError("preprocessing rejected every event in the recording")
    accepted_set = set(accepted_indices)
    rejected = tuple(
        RejectedEpoch(event_id=source_events[index].event_id, reasons=tuple(drop_reasons))
        for index, drop_reasons in enumerate(epochs.drop_log)
        if index not in accepted_set
    )

    if not np.isclose(float(epochs.info["sfreq"]), recipe.output_sampling_rate_hz):
        epochs.resample(recipe.output_sampling_rate_hz, npad="auto", verbose=False)
    accepted_events = tuple(source_events[index] for index in accepted_indices)
    epoch_metadata = tuple(
        EpochMetadata(
            epoch_id=f"{event.event_id}:epoch",
            event_id=event.event_id,
            selection_trial_id=event.selection_trial_id,
            recording_id=recording.metadata.key.recording_id,
            subject_id=recording.metadata.key.subject_id,
            session_id=recording.metadata.key.session_id,
            label=event.label,
            onset_sample=event.onset_sample,
        )
        for event in accepted_events
    )
    label_codes = {
        P300Label.UNKNOWN: -1,
        P300Label.NON_TARGET: 0,
        P300Label.TARGET: 1,
    }
    labels = np.asarray([label_codes[event.label] for event in accepted_events], dtype=np.int8)
    report = PreprocessingReport(
        recording_id=recording.metadata.key.recording_id,
        input_event_count=len(source_events),
        accepted_event_count=len(epoch_metadata),
        rejected_epochs=rejected,
    )
    return EpochBatch(
        data=epochs.get_data(copy=True),
        labels=labels,
        channel_names=epochs.ch_names,
        sampling_rate_hz=float(epochs.info["sfreq"]),
        metadata=epoch_metadata,
        config=recipe,
        report=report,
    )
