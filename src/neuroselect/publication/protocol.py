"""Fail-closed protocol for the offline NeuroSelect journal paper."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.provenance import RunKind, RunManifest

DEFAULT_PUBLICATION_PROTOCOL = Path("configs/publication/offline_methods_v1.yaml")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class JournalRouteKind(StrEnum):
    CURRENT_PRIMARY = "current_primary"
    CURRENT_FALLBACK = "current_fallback"
    FUTURE_SEPARATE = "future_separate"


class EvidenceRole(StrEnum):
    PRIMARY = "primary"
    ENGINEERING_CONTEXT = "engineering_context"


class GateStatus(StrEnum):
    PENDING = "pending"
    SATISFIED = "satisfied"


class JournalRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    journal_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=1)
    route_kind: JournalRouteKind
    article_type: str = Field(min_length=1, max_length=120)
    official_url: str = Field(pattern=r"^https://", max_length=2_048)
    rationale: str = Field(min_length=1, max_length=1_000)
    submission_gate_ids: tuple[str, ...] = ()


class PublicationEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    expected_run_id: str = Field(min_length=1, max_length=160)
    expected_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    require_release_ready: bool = True
    require_clean_source: bool = True


class PublicationEvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1, max_length=100)
    path: Path
    evidence_kind: str = Field(min_length=1, max_length=100)
    evidence_role: EvidenceRole
    expected_run_id: str = Field(min_length=1, max_length=160)
    expected_run_kind: RunKind
    expected_git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_report_claim_eligible: bool
    require_clean_source: bool = True


class ResearchQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(pattern=r"^RQ[1-9][0-9]*$")
    question: str = Field(min_length=1, max_length=500)
    evidence_source_ids: tuple[str, ...] = Field(min_length=1)
    primary_estimands: tuple[str, ...] = Field(min_length=1)


class ClaimPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: tuple[str, ...] = Field(min_length=1)
    prohibited: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disjoint_claims(self) -> ClaimPolicy:
        if {claim.casefold() for claim in self.allowed} & {
            claim.casefold() for claim in self.prohibited
        }:
            raise ValueError("allowed and prohibited publication claims must be disjoint")
        return self


class AnalysisCommitments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_evidence_frozen: Literal[True]
    existing_evidence_timing: Literal["retrospective_publication_protocol"]
    future_analysis_timing: Literal["prospectively_locked_from_protocol_freeze"]
    candidate_v2_role: Literal["exploratory_supplement"]
    outcome_based_omission_forbidden: Literal[True]
    inference_scope: Literal["descriptive_fixed_samples"]
    bootstrap_resamples: int = Field(ge=2_000)
    bootstrap_seed: int = Field(ge=0)


class SubmissionGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: str = Field(min_length=1, max_length=100)
    status: GateStatus
    owner: str = Field(min_length=1, max_length=120)
    requirement: str = Field(min_length=1, max_length=1_000)
    required_for_primary_submission: bool = True


class PublicationProtocolSpec(BaseModel):
    """Decision-complete publication framing and immutable evidence inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal["offline-methods-v1"]
    frozen_at: datetime
    working_title: str = Field(min_length=1, max_length=300)
    article_kind: Literal["offline_computational_original_research"]
    study_design: Literal["secondary_public_eeg_and_synthetic_language"]
    human_participant_recruitment: Literal[False]
    primary_author: str = Field(min_length=1, max_length=160)
    journal_routes: tuple[JournalRoute, ...] = Field(min_length=1)
    evidence_report: PublicationEvidenceReport
    evidence_sources: tuple[PublicationEvidenceSource, ...] = Field(min_length=1)
    research_questions: tuple[ResearchQuestion, ...] = Field(min_length=1)
    claim_policy: ClaimPolicy
    analysis_commitments: AnalysisCommitments
    submission_gates: tuple[SubmissionGate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_protocol_graph(self) -> PublicationProtocolSpec:
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise ValueError("publication protocol freeze time must include a timezone")
        routes = {route.journal_id: route for route in self.journal_routes}
        if len(routes) != len(self.journal_routes):
            raise ValueError("journal route IDs must be unique")
        if len({route.priority for route in self.journal_routes}) != len(self.journal_routes):
            raise ValueError("journal route priorities must be unique")
        primary_routes = [
            route
            for route in self.journal_routes
            if route.route_kind is JournalRouteKind.CURRENT_PRIMARY
        ]
        if len(primary_routes) != 1 or primary_routes[0].priority != 1:
            raise ValueError("exactly one priority-one current primary journal is required")
        sources = {source.source_id for source in self.evidence_sources}
        if len(sources) != len(self.evidence_sources):
            raise ValueError("publication evidence source IDs must be unique")
        questions = {question.question_id for question in self.research_questions}
        if len(questions) != len(self.research_questions):
            raise ValueError("publication research question IDs must be unique")
        unknown_sources = {
            source_id
            for question in self.research_questions
            for source_id in question.evidence_source_ids
            if source_id not in sources
        }
        if unknown_sources:
            raise ValueError(f"research questions reference unknown sources: {unknown_sources}")
        gates = {gate.gate_id for gate in self.submission_gates}
        if len(gates) != len(self.submission_gates):
            raise ValueError("publication submission gate IDs must be unique")
        unknown_gates = {
            gate_id
            for route in self.journal_routes
            for gate_id in route.submission_gate_ids
            if gate_id not in gates
        }
        if unknown_gates:
            raise ValueError(f"journal routes reference unknown submission gates: {unknown_gates}")
        return self

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


class PublicationProtocolCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str
    ready: bool
    required_for_protocol: bool
    observed: str
    required: str
    detail: str


class PublicationProtocolAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_ready: bool
    submission_ready: bool
    checks: tuple[PublicationProtocolCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> PublicationProtocolAssessment:
        protocol_ready = all(check.ready for check in self.checks if check.required_for_protocol)
        submission_ready = protocol_ready and all(check.ready for check in self.checks)
        if self.protocol_ready != protocol_ready or self.submission_ready != submission_ready:
            raise ValueError("publication readiness must agree with its checks")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_publication_protocol(
    path: str | Path = DEFAULT_PUBLICATION_PROTOCOL,
) -> PublicationProtocolSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("publication protocol must contain a YAML mapping")
    return PublicationProtocolSpec.model_validate(payload)


def _check(
    check_id: str,
    ready: bool,
    *,
    required_for_protocol: bool,
    observed: object,
    required: object,
    detail: str,
) -> PublicationProtocolCheck:
    return PublicationProtocolCheck(
        check_id=check_id,
        ready=ready,
        required_for_protocol=required_for_protocol,
        observed=str(observed),
        required=str(required),
        detail=detail,
    )


def _read_manifest(path: Path) -> RunManifest:
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def assess_publication_protocol(
    spec: PublicationProtocolSpec,
) -> PublicationProtocolAssessment:
    """Verify the frozen evidence and expose external submission gates separately."""

    checks: list[PublicationProtocolCheck] = []
    report_directory = spec.evidence_report.path
    report_manifest_path = report_directory / "manifest.json"
    report_path = report_directory / "report.json"
    report_tables: dict[str, dict[str, Any]] = {}
    try:
        report_manifest = _read_manifest(report_manifest_path)
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_tables = {str(table["evidence_kind"]): table for table in report_payload["tables"]}
        report_ready = (
            report_manifest.run_kind is RunKind.RESEARCH_REPORT
            and report_manifest.run_id == spec.evidence_report.expected_run_id
            and report_manifest.digest() == spec.evidence_report.expected_manifest_sha256
            and (
                not spec.evidence_report.require_release_ready
                or report_payload.get("release_ready") is True
            )
            and (
                not spec.evidence_report.require_clean_source
                or report_manifest.metadata.get("working_tree_dirty") is False
            )
        )
        report_detail = "Research report identity, release state, and provenance verified."
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        report_ready = False
        report_detail = str(error)
    checks.append(
        _check(
            "frozen-research-report",
            report_ready,
            required_for_protocol=True,
            observed=(spec.evidence_report.expected_manifest_sha256 if report_ready else "invalid"),
            required=spec.evidence_report.expected_manifest_sha256,
            detail=report_detail,
        )
    )

    for source in spec.evidence_sources:
        try:
            manifest = _read_manifest(source.path / "manifest.json")
            table = report_tables[source.evidence_kind]
            source_ready = (
                manifest.run_id == source.expected_run_id
                and manifest.run_kind is source.expected_run_kind
                and manifest.git_sha == source.expected_git_sha
                and manifest.digest() == source.expected_manifest_sha256
                and (
                    not source.require_clean_source
                    or manifest.metadata.get("working_tree_dirty") is False
                )
                and table.get("source_run_id") == source.expected_run_id
                and table.get("source_manifest_sha256") == source.expected_manifest_sha256
                and table.get("claim_eligible") is source.expected_report_claim_eligible
            )
            source_detail = (
                "Source manifest and its evidence-table identity are verified."
                if source_ready
                else "Source manifest or report-table identity differs from the protocol."
            )
            observed = manifest.digest()
        except (KeyError, OSError, TypeError, ValueError) as error:
            source_ready = False
            source_detail = str(error)
            observed = "invalid"
        checks.append(
            _check(
                f"evidence-{source.source_id}",
                source_ready,
                required_for_protocol=True,
                observed=observed,
                required=source.expected_manifest_sha256,
                detail=source_detail,
            )
        )

    checks.append(
        _check(
            "publication-framing",
            (
                spec.human_participant_recruitment is False
                and spec.analysis_commitments.primary_evidence_frozen
                and spec.analysis_commitments.outcome_based_omission_forbidden
            ),
            required_for_protocol=True,
            observed=spec.study_design,
            required="offline frozen primary evidence with no participant recruitment",
            detail=(
                "The paper is locked as an offline computational study; future v2 work is "
                "exploratory and cannot replace the primary evidence."
            ),
        )
    )

    for gate in spec.submission_gates:
        if not gate.required_for_primary_submission:
            continue
        checks.append(
            _check(
                f"submission-{gate.gate_id}",
                gate.status is GateStatus.SATISFIED,
                required_for_protocol=False,
                observed=gate.status.value,
                required=GateStatus.SATISFIED.value,
                detail=gate.requirement,
            )
        )

    protocol_ready = all(check.ready for check in checks if check.required_for_protocol)
    submission_ready = protocol_ready and all(check.ready for check in checks)
    return PublicationProtocolAssessment(
        protocol_id=spec.protocol_id,
        protocol_sha256=spec.digest(),
        protocol_ready=protocol_ready,
        submission_ready=submission_ready,
        checks=tuple(checks),
    )
