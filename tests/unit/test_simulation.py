from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from neuroselect.bci import SeededNeuralSimulator, SimulationConfig, SimulationRegime
from neuroselect.core.models import EvidenceMode

CANDIDATES = ("candidate-a", "candidate-b", "candidate-c", "candidate-d")


def test_simulation_is_reproducible_and_call_order_independent() -> None:
    simulator = SeededNeuralSimulator()
    first = simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-b",
        session_id="test-session",
        round_index=7,
    )
    simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-a",
        session_id="another-session",
        round_index=99,
    )
    second = simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-b",
        session_id="test-session",
        round_index=7,
    )

    assert first == second
    assert first.derived_seed == 4_402_022_173_831_256_763
    assert first.evidence.evidence_id == "sim-98ecc6ea9c4e3130bbb7"


def test_target_supported_regime_places_intended_candidate_first() -> None:
    simulator = SeededNeuralSimulator(
        SimulationConfig(lapse_probability=0.0, ambiguous_probability=0.0)
    )
    result = simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-b",
        session_id="target-session",
        round_index=3,
        subject_id="synthetic-person-1",
    )
    probabilities = result.evidence.candidate_probabilities

    assert result.regime is SimulationRegime.TARGET_SUPPORTED
    assert result.effective_target_id == result.intended_candidate_id == "candidate-b"
    assert max(probabilities, key=probabilities.__getitem__) == "candidate-b"
    assert result.evidence.mode is EvidenceMode.SIMULATION
    assert result.evidence.subject_id == "synthetic-person-1"
    assert result.evidence.session_id == "target-session"
    assert result.evidence.trial_id == "round-000003"
    assert result.evidence.recorded_at == datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC)


def test_lapse_regime_places_a_distractor_first() -> None:
    simulator = SeededNeuralSimulator(
        SimulationConfig(lapse_probability=1.0, ambiguous_probability=0.0)
    )
    result = simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-b",
        session_id="lapse-session",
        round_index=4,
    )
    probabilities = result.evidence.candidate_probabilities

    assert result.regime is SimulationRegime.DISTRACTOR_LAPSE
    assert result.effective_target_id in set(CANDIDATES) - {"candidate-b"}
    assert max(probabilities, key=probabilities.__getitem__) == result.effective_target_id


def test_ambiguous_regime_emits_normalized_uncertain_evidence() -> None:
    simulator = SeededNeuralSimulator(
        SimulationConfig(lapse_probability=0.0, ambiguous_probability=1.0)
    )
    result = simulator.simulate(
        candidate_ids=CANDIDATES,
        intended_candidate_id="candidate-c",
        session_id="ambiguous-session",
        round_index=11,
    )
    probabilities = result.evidence.candidate_probabilities
    ordered = sorted(probabilities.values(), reverse=True)

    assert result.regime is SimulationRegime.AMBIGUOUS
    assert result.effective_target_id is None
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert result.evidence.entropy == pytest.approx(
        -sum(probability * math.log(probability) for probability in probabilities.values())
    )
    assert result.evidence.top_margin == pytest.approx(ordered[0] - ordered[1])


@pytest.mark.parametrize(
    ("candidate_ids", "intended", "round_index", "message"),
    [
        (("only-one",), "only-one", 0, "at least two"),
        (("duplicate", "duplicate"), "duplicate", 0, "must be unique"),
        (("candidate-a", "candidate-b"), "absent", 0, "must be present"),
        (("candidate-a", "candidate-b"), "candidate-a", -1, "cannot be negative"),
    ],
)
def test_simulation_rejects_invalid_round_inputs(
    candidate_ids: tuple[str, ...], intended: str, round_index: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SeededNeuralSimulator().simulate(
            candidate_ids=candidate_ids,
            intended_candidate_id=intended,
            session_id="invalid-session",
            round_index=round_index,
        )


def test_simulation_config_rejects_overlapping_regime_probabilities() -> None:
    with pytest.raises(ValidationError, match="cannot sum above one"):
        SimulationConfig(lapse_probability=0.6, ambiguous_probability=0.5)
