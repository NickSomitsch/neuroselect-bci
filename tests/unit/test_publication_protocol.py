from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.provenance import RunKind, RunManifest, RunStatus
from neuroselect.publication import (
    PublicationProtocolSpec,
    assess_publication_protocol,
    load_publication_protocol,
)

ROOT = Path(__file__).parents[2]


def test_tracked_publication_protocol_locks_strategy_and_sources() -> None:
    spec = load_publication_protocol(ROOT / "configs/publication/offline_methods_v1.yaml")

    assert spec.human_participant_recruitment is False
    assert spec.journal_routes[0].journal_id == "taylor-francis-rbet"
    assert spec.journal_routes[0].article_type == "Original Research"
    assert [question.question_id for question in spec.research_questions] == [
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
        "RQ5",
    ]
    assert {source.source_id for source in spec.evidence_sources} == {
        "controlled-simulation",
        "held-out-language",
        "xdawn-original-task",
        "counterfactual-research",
    }
    assert spec.analysis_commitments.primary_evidence_frozen
    assert spec.analysis_commitments.candidate_v2_role == "exploratory_supplement"
    assert len(spec.digest()) == 64


def test_publication_protocol_rejects_unknown_sources_and_unzoned_time() -> None:
    spec = load_publication_protocol(ROOT / "configs/publication/offline_methods_v1.yaml")
    invalid_question = spec.research_questions[0].model_copy(
        update={"evidence_source_ids": ("missing",)}
    )
    payload = spec.model_dump(mode="json")
    payload["research_questions"][0] = invalid_question.model_dump(mode="json")
    with pytest.raises(ValidationError, match="unknown sources"):
        PublicationProtocolSpec.model_validate(payload)
    payload = spec.model_dump(mode="json")
    payload["frozen_at"] = "2026-07-27T16:42:08"
    with pytest.raises(ValidationError, match="include a timezone"):
        type(spec).model_validate(payload)


def _copy_evidence_fixture(
    spec: PublicationProtocolSpec, tmp_path: Path
) -> PublicationProtocolSpec:
    updated_sources = []
    report_tables = []
    for source in spec.evidence_sources:
        source_directory = tmp_path / source.source_id
        source_directory.mkdir()
        manifest = RunManifest(
            run_id=source.expected_run_id,
            run_kind=source.expected_run_kind,
            status=RunStatus.COMPLETED,
            started_at=datetime(2026, 7, 27, tzinfo=UTC),
            completed_at=datetime(2026, 7, 27, tzinfo=UTC),
            git_sha=source.expected_git_sha,
            config_sha256="a" * 64,
            random_seeds={"global": 1},
            package_versions={"python": "3.12"},
            device={"system": "test"},
            metadata={"working_tree_dirty": False},
        )
        (source_directory / "manifest.json").write_text(
            manifest.canonical_json() + "\n", encoding="utf-8"
        )
        updated_sources.append(
            source.model_copy(
                update={
                    "path": source_directory,
                    "expected_manifest_sha256": manifest.digest(),
                }
            )
        )
        report_tables.append(
            {
                "evidence_kind": source.evidence_kind,
                "source_run_id": source.expected_run_id,
                "source_manifest_sha256": manifest.digest(),
                "claim_eligible": source.expected_report_claim_eligible,
            }
        )
    report_directory = tmp_path / "report"
    report_directory.mkdir()
    report_manifest = RunManifest(
        run_id=spec.evidence_report.expected_run_id,
        run_kind=RunKind.RESEARCH_REPORT,
        status=RunStatus.COMPLETED,
        started_at=datetime(2026, 7, 27, tzinfo=UTC),
        completed_at=datetime(2026, 7, 27, tzinfo=UTC),
        git_sha="b239179",
        config_sha256="b" * 64,
        random_seeds={"global": 1},
        package_versions={"python": "3.12"},
        device={"system": "test"},
        metadata={"working_tree_dirty": False},
    )
    (report_directory / "manifest.json").write_text(
        report_manifest.canonical_json() + "\n", encoding="utf-8"
    )
    (report_directory / "report.json").write_text(
        json.dumps({"release_ready": True, "tables": report_tables}),
        encoding="utf-8",
    )
    return spec.model_copy(
        update={
            "evidence_sources": tuple(updated_sources),
            "evidence_report": spec.evidence_report.model_copy(
                update={
                    "path": report_directory,
                    "expected_manifest_sha256": report_manifest.digest(),
                }
            ),
        }
    )


def test_publication_assessment_verifies_sources_and_separates_submission_gates(
    tmp_path: Path,
) -> None:
    tracked = load_publication_protocol(ROOT / "configs/publication/offline_methods_v1.yaml")
    spec = _copy_evidence_fixture(tracked, tmp_path)

    assessment = assess_publication_protocol(spec)

    assert assessment.protocol_ready is True
    assert assessment.submission_ready is False
    assert all(check.ready for check in assessment.checks if check.required_for_protocol)
    assert {check.check_id for check in assessment.checks if not check.ready} == {
        "submission-uibk-open-access",
        "submission-secondary-use-ethics",
        "submission-domain-review",
        "submission-author-metadata",
    }
    assert '"protocol_ready":true' in assessment.canonical_json()

    first_source = spec.evidence_sources[0]
    (first_source.path / "manifest.json").write_text("{}\n", encoding="utf-8")
    invalid = assess_publication_protocol(spec)
    assert invalid.protocol_ready is False
    assert invalid.submission_ready is False
