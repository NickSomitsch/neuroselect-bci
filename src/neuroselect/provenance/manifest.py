"""Machine-readable experiment manifests with deterministic content digests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RunKind(StrEnum):
    SIMULATION = "simulation"
    EEG_ORIGINAL_TASK = "eeg_original_task"
    COUNTERFACTUAL_REPLAY = "counterfactual_replay"
    COMPONENT_EVALUATION = "component_evaluation"
    RESEARCH_REPORT = "research_report"
    PUBLICATION_ANALYSIS = "publication_analysis"
    EXPLORATORY_EVALUATION = "exploratory_evaluation"


class RunStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactRef(BaseModel):
    """Immutable identifier for an input or output artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1, max_length=160)
    uri: str = Field(min_length=1, max_length=2048)
    sha256: Sha256
    revision: str | None = Field(default=None, max_length=256)
    license: str | None = Field(default=None, max_length=128)


class RunManifest(BaseModel):
    """Complete provenance required for a reproducible NeuroSelect result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1, max_length=160)
    run_kind: RunKind
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    git_sha: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    config_sha256: Sha256
    random_seeds: dict[str, int] = Field(min_length=1)
    package_versions: dict[str, str] = Field(default_factory=dict)
    device: dict[str, str] = Field(default_factory=dict)
    datasets: tuple[ArtifactRef, ...] = ()
    models: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_completion(self) -> RunManifest:
        if self.status is RunStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed runs require completed_at")
        if self.status is RunStatus.COMPLETED and (not self.package_versions or not self.device):
            raise ValueError("completed runs require package versions and device provenance")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self

    def canonical_json(self) -> str:
        """Return stable JSON suitable for hashing and artifact comparison."""

        payload = self.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical manifest representation."""

        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
