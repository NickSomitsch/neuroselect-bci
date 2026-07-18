"""Call-order-independent simulation of noisy neural candidate probabilities."""

from __future__ import annotations

import hashlib
import math
import random
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import EvidenceMode, NeuralSelectionEvidence


class SimulationRegime(StrEnum):
    TARGET_SUPPORTED = "target_supported"
    DISTRACTOR_LAPSE = "distractor_lapse"
    AMBIGUOUS = "ambiguous"


class SimulationConfig(BaseModel):
    """Versioned controls for seeded Dirichlet-like probability draws."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=20260717, ge=0)
    target_concentration: float = Field(default=18.0, gt=0.0)
    distractor_concentration: float = Field(default=1.0, gt=0.0)
    ambiguous_concentration: float = Field(default=10.0, gt=0.0)
    lapse_probability: float = Field(default=0.08, ge=0.0, le=1.0)
    ambiguous_probability: float = Field(default=0.12, ge=0.0, le=1.0)
    timeline_origin: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def validate_regime_probabilities(self) -> SimulationConfig:
        if self.lapse_probability + self.ambiguous_probability > 1.0:
            raise ValueError("lapse and ambiguous probabilities cannot sum above one")
        return self


class SimulatedRound(BaseModel):
    """Neural evidence plus the known simulation ground truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: NeuralSelectionEvidence
    regime: SimulationRegime
    intended_candidate_id: str
    effective_target_id: str | None
    derived_seed: int


class SeededNeuralSimulator:
    """Generate reproducible neural evidence from stable round identifiers."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def _round_material(
        self,
        *,
        candidate_ids: tuple[str, ...],
        intended_candidate_id: str,
        session_id: str,
        round_index: int,
    ) -> bytes:
        return "\0".join(
            (
                str(self.config.seed),
                session_id,
                str(round_index),
                intended_candidate_id,
                *candidate_ids,
            )
        ).encode()

    def simulate(
        self,
        *,
        candidate_ids: tuple[str, ...],
        intended_candidate_id: str,
        session_id: str,
        round_index: int,
        subject_id: str = "synthetic-subject",
    ) -> SimulatedRound:
        """Simulate one round; identical identifiers always yield identical evidence."""

        if len(candidate_ids) < 2:
            raise ValueError("simulation requires at least two candidate IDs")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("simulation candidate IDs must be unique")
        if intended_candidate_id not in candidate_ids:
            raise ValueError("intended candidate must be present in candidate IDs")
        if round_index < 0:
            raise ValueError("round_index cannot be negative")

        material = self._round_material(
            candidate_ids=candidate_ids,
            intended_candidate_id=intended_candidate_id,
            session_id=session_id,
            round_index=round_index,
        )
        digest = hashlib.sha256(material).digest()
        derived_seed = int.from_bytes(digest[:8], "big")
        generator = random.Random(derived_seed)

        regime_draw = generator.random()
        if regime_draw < self.config.ambiguous_probability:
            regime = SimulationRegime.AMBIGUOUS
            effective_target_id = None
        elif regime_draw < self.config.ambiguous_probability + self.config.lapse_probability:
            regime = SimulationRegime.DISTRACTOR_LAPSE
            distractors = [value for value in candidate_ids if value != intended_candidate_id]
            effective_target_id = generator.choice(distractors)
        else:
            regime = SimulationRegime.TARGET_SUPPORTED
            effective_target_id = intended_candidate_id

        concentrations = {
            candidate_id: self._concentration(candidate_id, effective_target_id, regime)
            for candidate_id in candidate_ids
        }
        samples = {
            candidate_id: generator.gammavariate(concentration, 1.0)
            for candidate_id, concentration in concentrations.items()
        }
        total = sum(samples.values())
        probabilities = {candidate_id: sample / total for candidate_id, sample in samples.items()}
        if effective_target_id is not None:
            top_candidate = max(probabilities, key=probabilities.__getitem__)
            if top_candidate != effective_target_id:
                probabilities[top_candidate], probabilities[effective_target_id] = (
                    probabilities[effective_target_id],
                    probabilities[top_candidate],
                )

        ordered_probabilities = sorted(probabilities.values(), reverse=True)
        entropy = -sum(value * math.log(value) for value in probabilities.values())
        top_margin = ordered_probabilities[0] - ordered_probabilities[1]
        evidence_digest = hashlib.sha256(material + regime.value.encode()).hexdigest()
        evidence = NeuralSelectionEvidence(
            evidence_id=f"sim-{evidence_digest[:20]}",
            mode=EvidenceMode.SIMULATION,
            candidate_probabilities=probabilities,
            calibration_id="seeded-simulator-v1",
            entropy=entropy,
            top_margin=top_margin,
            subject_id=subject_id,
            session_id=session_id,
            trial_id=f"round-{round_index:06d}",
            recorded_at=self.config.timeline_origin + timedelta(seconds=round_index),
        )
        return SimulatedRound(
            evidence=evidence,
            regime=regime,
            intended_candidate_id=intended_candidate_id,
            effective_target_id=effective_target_id,
            derived_seed=derived_seed,
        )

    def _concentration(
        self,
        candidate_id: str,
        effective_target_id: str | None,
        regime: SimulationRegime,
    ) -> float:
        if regime is SimulationRegime.AMBIGUOUS:
            return self.config.ambiguous_concentration
        if candidate_id == effective_target_id:
            return self.config.target_concentration
        return self.config.distractor_concentration
