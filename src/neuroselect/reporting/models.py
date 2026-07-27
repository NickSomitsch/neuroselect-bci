"""Typed contracts for release-grade, evidence-separated research reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation import EvaluationCondition
from neuroselect.provenance import RunKind

MetricValue = float | int | str | bool | None


class EvidenceKind(StrEnum):
    """Mutually exclusive evidence tiers that must never be pooled."""

    CONTROLLED_SIMULATION = "controlled_simulation"
    LANGUAGE_COMPONENT = "language_component"
    EEG_ORIGINAL_TASK = "eeg_original_task"
    COUNTERFACTUAL_REPLAY = "counterfactual_replay"


class ReportSourceSpec(BaseModel):
    """One expected local artifact directory in a research report recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=200)
    path: Path
    expected_run_kind: RunKind
    required: bool = False
    reference_condition: EvaluationCondition | None = None

    @model_validator(mode="after")
    def validate_source_kind(self) -> ReportSourceSpec:
        if self.expected_run_kind is RunKind.RESEARCH_REPORT:
            raise ValueError("a research report cannot consume another research report")
        if (
            self.reference_condition is not None
            and self.expected_run_kind is not RunKind.SIMULATION
        ):
            raise ValueError("reference conditions are supported only for paired simulation runs")
        return self


class ResearchReportSpec(BaseModel):
    """Tracked, deterministic recipe for a release report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    report_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    generated_at: datetime
    bootstrap_seed: int = Field(default=20260720, ge=0)
    bootstrap_resamples: int = Field(default=2_000, ge=100, le=100_000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    reject_dirty_sources: bool = True
    sources: tuple[ReportSourceSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_recipe(self) -> ResearchReportSpec:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("research report generation time must include a timezone")
        identifiers = [source.source_id for source in self.sources]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research report source IDs must be unique")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class ReportMetricRow(BaseModel):
    """One machine-readable point-estimate row within an evidence-specific table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=240)
    values: dict[str, MetricValue] = Field(min_length=1)


class ReportInterval(BaseModel):
    """One descriptive paired interval retained with its reference and sampling unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str = Field(min_length=1, max_length=160)
    reference_condition: str = Field(min_length=1, max_length=160)
    metric: str = Field(min_length=1, max_length=160)
    observed_delta: float = Field(ge=-1.0, le=1.0)
    lower_bound: float = Field(ge=-1.0, le=1.0)
    upper_bound: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    resamples: int = Field(ge=100)
    sampling_unit: Literal["profile_then_trial", "subject_then_trial"]

    @model_validator(mode="after")
    def validate_bounds(self) -> ReportInterval:
        if self.lower_bound > self.upper_bound:
            raise ValueError("report interval bounds must be ordered")
        return self


class EvidenceTable(BaseModel):
    """A statistical table whose evidence scope cannot be mistaken for another tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    evidence_kind: EvidenceKind
    scope_statement: str = Field(min_length=1, max_length=1_000)
    source_run_id: str = Field(min_length=1, max_length=160)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_git_sha: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_tree_dirty: bool
    claim_eligible: bool
    metric_rows: tuple[ReportMetricRow, ...] = Field(min_length=1)
    intervals: tuple[ReportInterval, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)


class MissingReportSource(BaseModel):
    """An explicitly absent optional or required input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    label: str
    path: str
    required: bool
    reason: str = Field(min_length=1, max_length=500)


class ResearchReport(BaseModel):
    """Canonical evidence-separated report before JSON and Markdown serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1, max_length=160)
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: ResearchReportSpec
    release_ready: bool
    tables: tuple[EvidenceTable, ...] = ()
    missing_sources: tuple[MissingReportSource, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> ResearchReport:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("research report time must include a timezone")
        if self.config_sha256 != self.spec.digest():
            raise ValueError("research report config hash must match the embedded spec")
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("research report table IDs must be unique")
        represented = set(table_ids).union(source.source_id for source in self.missing_sources)
        expected = {source.source_id for source in self.spec.sources}
        if represented != expected:
            raise ValueError("every configured report source must be present or explicitly missing")
        required_missing = any(source.required for source in self.missing_sources)
        disallowed_dirty = self.spec.reject_dirty_sources and any(
            table.source_tree_dirty for table in self.tables
        )
        if self.release_ready != (not required_missing and not disallowed_dirty):
            raise ValueError("release readiness must reflect missing and dirty source evidence")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
