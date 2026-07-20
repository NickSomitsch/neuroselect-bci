"""BCI input adapters, flash aggregation, and evidence simulators."""

from neuroselect.bci.aggregation import (
    FlashLayout,
    FlashProbability,
    FlashProbabilityTrial,
    TileAggregationConfig,
    aggregate_flash_probabilities,
    remap_recorded_target,
)
from neuroselect.bci.replay import EpochReplay, ProbabilityDecoder, ReplayFrame, ReplayState
from neuroselect.bci.simulation import (
    SeededNeuralSimulator,
    SimulatedRound,
    SimulationConfig,
    SimulationRegime,
)

__all__ = [
    "EpochReplay",
    "FlashLayout",
    "FlashProbability",
    "FlashProbabilityTrial",
    "ProbabilityDecoder",
    "ReplayFrame",
    "ReplayState",
    "SeededNeuralSimulator",
    "SimulatedRound",
    "SimulationConfig",
    "SimulationRegime",
    "TileAggregationConfig",
    "aggregate_flash_probabilities",
    "remap_recorded_target",
]
