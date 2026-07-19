"""Pull-based virtual-clock replay for chronological preprocessed P300 events."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

from neuroselect.eeg import EpochBatch, EpochMetadata, P300Label, PreprocessingConfig


class ProbabilityDecoder(Protocol):
    channel_names: tuple[str, ...]
    sampling_rate_hz: float
    epoch_sample_count: int
    preprocessing_config: PreprocessingConfig

    def predict_probabilities(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]: ...


class ReplayState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass(frozen=True)
class ReplayFrame:
    """One epoch emitted at a deterministic virtual-clock offset."""

    sequence_index: int
    source_offset_seconds: float
    replay_offset_seconds: float
    metadata: EpochMetadata
    label: P300Label
    data: npt.NDArray[np.float32]
    target_probability: float | None

    @property
    def supervised_label_available(self) -> bool:
        return self.label is not P300Label.UNKNOWN


class EpochReplay:
    """Replay one recording without sleeping or changing source timestamps/order."""

    def __init__(
        self,
        batch: EpochBatch,
        *,
        speed: float = 1.0,
        decoder: ProbabilityDecoder | None = None,
    ) -> None:
        recording_ids = {item.recording_id for item in batch.metadata}
        if len(recording_ids) != 1:
            raise ValueError("one replay stream must contain exactly one recording")
        if any(item.onset_seconds is None for item in batch.metadata):
            raise ValueError("replay requires preserved source onset seconds")
        if decoder is not None and (
            decoder.channel_names != batch.channel_names
            or not np.isclose(decoder.sampling_rate_hz, batch.sampling_rate_hz)
            or decoder.epoch_sample_count != batch.data.shape[2]
            or decoder.preprocessing_config != batch.config
        ):
            raise ValueError("replay epoch contract does not match the decoder checkpoint")
        self.batch = batch
        self.decoder = decoder
        self._order = tuple(
            sorted(
                range(len(batch.metadata)),
                key=lambda index: (
                    batch.metadata[index].onset_seconds,
                    batch.metadata[index].onset_sample,
                    batch.metadata[index].epoch_id,
                ),
            )
        )
        source_times: list[float] = []
        for index in self._order:
            onset_seconds = batch.metadata[index].onset_seconds
            assert onset_seconds is not None
            source_times.append(onset_seconds)
        self._source_times = tuple(source_times)
        self._position = 0
        self._speed = 1.0
        self.set_speed(speed)
        self.state = ReplayState.READY

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def position(self) -> int:
        return self._position

    @property
    def frame_count(self) -> int:
        return len(self._order)

    def set_speed(self, speed: float) -> None:
        if not np.isfinite(speed) or speed <= 0.0 or speed > 32.0:
            raise ValueError("replay speed must be finite and in (0, 32]")
        self._speed = float(speed)

    def start(self) -> None:
        if self.state is ReplayState.FINISHED:
            raise RuntimeError("reset or seek before restarting a finished replay")
        self.state = ReplayState.RUNNING

    def pause(self) -> None:
        if self.state is not ReplayState.RUNNING:
            raise RuntimeError("only a running replay can be paused")
        self.state = ReplayState.PAUSED

    def reset(self) -> None:
        self._position = 0
        self.state = ReplayState.READY

    def seek_index(self, position: int) -> None:
        if position < 0 or position > self.frame_count:
            raise ValueError("replay position is outside the stream")
        self._position = position
        self.state = ReplayState.FINISHED if position == self.frame_count else ReplayState.PAUSED

    def seek_seconds(self, source_offset_seconds: float) -> None:
        if not np.isfinite(source_offset_seconds) or source_offset_seconds < 0.0:
            raise ValueError("source seek offset must be finite and non-negative")
        absolute_time = self._source_times[0] + source_offset_seconds
        self.seek_index(bisect_left(self._source_times, absolute_time))

    def next_frame(self) -> ReplayFrame | None:
        if self.state is ReplayState.FINISHED:
            return None
        if self.state is not ReplayState.RUNNING:
            raise RuntimeError("replay must be running before a frame can be emitted")
        source_index = self._order[self._position]
        metadata = self.batch.metadata[source_index]
        data = self.batch.data[source_index].copy()
        data.flags.writeable = False
        probability = None
        if self.decoder is not None:
            epoch = self.batch.data[source_index : source_index + 1]
            probability = float(self.decoder.predict_probabilities(epoch)[0])
        source_offset = self._source_times[self._position] - self._source_times[0]
        frame = ReplayFrame(
            sequence_index=self._position,
            source_offset_seconds=source_offset,
            replay_offset_seconds=source_offset / self.speed,
            metadata=metadata,
            label=metadata.label,
            data=data,
            target_probability=probability,
        )
        self._position += 1
        if self._position == self.frame_count:
            self.state = ReplayState.FINISHED
        return frame
