"""Aggregate calibrated P300 flash probabilities into fixed tile posteriors."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import EvidenceMode, NeuralSelectionEvidence


class FlashLayout(BaseModel):
    """Fixed tile-to-stimulus signatures for one complete P300 selection round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    layout_id: str = Field(min_length=1, max_length=128)
    candidate_ids: tuple[str, ...] = Field(min_length=2, max_length=64)
    stimulus_codes: tuple[int, ...] = Field(min_length=2, max_length=128)
    candidate_code_sets: dict[str, tuple[int, ...]]

    @model_validator(mode="after")
    def validate_signatures(self) -> FlashLayout:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("flash-layout candidate IDs must be unique")
        if len(self.stimulus_codes) != len(set(self.stimulus_codes)):
            raise ValueError("flash-layout stimulus codes must be unique")
        if set(self.candidate_code_sets) != set(self.candidate_ids):
            raise ValueError("flash-layout signatures must cover exactly the visible candidates")
        allowed = set(self.stimulus_codes)
        signatures: list[frozenset[int]] = []
        for candidate_id in self.candidate_ids:
            values = self.candidate_code_sets[candidate_id]
            signature = frozenset(values)
            if not values or len(values) != len(signature) or not signature.issubset(allowed):
                raise ValueError("each candidate requires a non-empty unique valid code signature")
            signatures.append(signature)
        if len(signatures) != len(set(signatures)):
            raise ValueError("candidate stimulus-code signatures must be unique")
        return self

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class FlashProbability(BaseModel):
    """One chronological decoder probability for an original recorded flash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_index: int = Field(ge=0)
    event_id: str = Field(min_length=1, max_length=200)
    stimulus_code: int = Field(ge=0)
    target_probability: float = Field(ge=0.0, le=1.0)
    onset_seconds: float = Field(ge=0.0)


class FlashProbabilityTrial(BaseModel):
    """Original event stream and target signature for one labeled Study P trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_trial_id: str = Field(min_length=1, max_length=200)
    subject_id: str = Field(min_length=1, max_length=32)
    session_id: str = Field(min_length=1, max_length=32)
    events: tuple[FlashProbability, ...] = Field(min_length=2)
    recorded_target_codes: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event_order(self) -> FlashProbabilityTrial:
        indices = tuple(event.sequence_index for event in self.events)
        if indices != tuple(range(len(self.events))):
            raise ValueError("flash events must use contiguous chronological sequence indices")
        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("flash event IDs must be unique within a selection trial")
        onsets = tuple(event.onset_seconds for event in self.events)
        if onsets != tuple(sorted(onsets)):
            raise ValueError("flash events must preserve chronological onset order")
        if len(self.recorded_target_codes) != len(set(self.recorded_target_codes)):
            raise ValueError("recorded target stimulus codes must be unique")
        if not set(self.recorded_target_codes).issubset(
            {event.stimulus_code for event in self.events}
        ):
            raise ValueError("recorded target codes must occur in the event stream")
        return self


class TileAggregationConfig(BaseModel):
    """Locked numerical and completeness rules for flash-to-tile aggregation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    aggregation_revision: Literal["flash-log-likelihood-softmax-v1"] = (
        "flash-log-likelihood-softmax-v1"
    )
    probability_clip: float = Field(default=1e-6, gt=0.0, lt=0.1)
    posterior_temperature: float = Field(default=1.0, gt=0.0, le=100.0)
    minimum_code_repetitions: int = Field(default=1, ge=1, le=1_000)

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def remap_recorded_target(
    layout: FlashLayout,
    *,
    target_candidate_id: str,
    recorded_target_codes: tuple[int, ...],
) -> FlashLayout:
    """Swap signatures so the intended tile occupies the original recorded target position."""

    if target_candidate_id not in layout.candidate_code_sets:
        raise ValueError("counterfactual target must be visible in the fixed layout")
    target_signature = frozenset(recorded_target_codes)
    source_candidate_id = next(
        (
            candidate_id
            for candidate_id, codes in layout.candidate_code_sets.items()
            if frozenset(codes) == target_signature
        ),
        None,
    )
    if source_candidate_id is None:
        raise ValueError("recorded target signature is absent from the supplied flash layout")
    if source_candidate_id == target_candidate_id:
        return layout
    signatures = dict(layout.candidate_code_sets)
    signatures[source_candidate_id], signatures[target_candidate_id] = (
        signatures[target_candidate_id],
        signatures[source_candidate_id],
    )
    material = (
        f"{layout.layout_id}:{target_candidate_id}:{','.join(map(str, recorded_target_codes))}"
    )
    return layout.model_copy(
        update={
            "layout_id": f"counterfactual-{hashlib.sha256(material.encode()).hexdigest()[:20]}",
            "candidate_code_sets": signatures,
        }
    )


def aggregate_flash_probabilities(
    trial: FlashProbabilityTrial,
    layout: FlashLayout,
    *,
    calibration_id: str,
    recorded_at: datetime,
    config: TileAggregationConfig | None = None,
) -> NeuralSelectionEvidence:
    """Compute a posterior using every flash's target/non-target log likelihood."""

    recipe = config or TileAggregationConfig()
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("aggregated neural evidence time must include a timezone")
    observed_codes = {event.stimulus_code for event in trial.events}
    if not observed_codes.issubset(set(layout.stimulus_codes)):
        raise ValueError("event stream contains stimulus codes outside the flash layout")
    counts = {
        code: sum(event.stimulus_code == code for event in trial.events)
        for code in layout.stimulus_codes
    }
    incomplete = {
        code: count for code, count in counts.items() if count < recipe.minimum_code_repetitions
    }
    if incomplete:
        raise ValueError(f"flash trial is incomplete for stimulus codes: {incomplete}")
    scores: list[float] = []
    for candidate_id in layout.candidate_ids:
        target_codes = set(layout.candidate_code_sets[candidate_id])
        score = 0.0
        for event in trial.events:
            probability = float(
                np.clip(
                    event.target_probability,
                    recipe.probability_clip,
                    1.0 - recipe.probability_clip,
                )
            )
            score += math.log(
                probability if event.stimulus_code in target_codes else 1.0 - probability
            )
        scores.append(score / recipe.posterior_temperature)
    maximum = max(scores)
    exponentials = np.exp(np.asarray(scores, dtype=np.float64) - maximum)
    probabilities = exponentials / np.sum(exponentials)
    posterior = {
        candidate_id: float(probabilities[index])
        for index, candidate_id in enumerate(layout.candidate_ids)
    }
    ordered = sorted(posterior.values(), reverse=True)
    entropy = -sum(value * math.log(value) for value in posterior.values() if value > 0.0)
    evidence_material = {
        "trial": trial.model_dump(mode="json"),
        "layout_sha256": layout.digest(),
        "config_sha256": recipe.digest(),
        "calibration_id": calibration_id,
    }
    evidence_sha = hashlib.sha256(
        json.dumps(evidence_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return NeuralSelectionEvidence(
        evidence_id=f"replay-{evidence_sha[:20]}",
        mode=EvidenceMode.REPLAY,
        candidate_probabilities=posterior,
        calibration_id=calibration_id,
        entropy=entropy,
        top_margin=ordered[0] - ordered[1],
        subject_id=trial.subject_id,
        session_id=trial.session_id,
        trial_id=trial.selection_trial_id,
        recorded_at=recorded_at,
    )
