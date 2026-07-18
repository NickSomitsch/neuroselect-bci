from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus

SHA256 = "a" * 64
STARTED = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def make_manifest(**updates: object) -> RunManifest:
    values: dict[str, object] = {
        "run_id": "smoke-001",
        "run_kind": RunKind.SIMULATION,
        "status": RunStatus.COMPLETED,
        "started_at": STARTED,
        "completed_at": STARTED + timedelta(seconds=2),
        "git_sha": "48f8314",
        "config_sha256": SHA256,
        "random_seeds": {"global": 20260717},
        "datasets": (
            ArtifactRef(
                artifact_id="fixture-data",
                uri="synthetic://fixture-data",
                sha256=SHA256,
                license="CC0-1.0",
            ),
        ),
    }
    values.update(updates)
    return RunManifest.model_validate(values)


def test_manifest_digest_is_stable() -> None:
    first = make_manifest(package_versions={"pydantic": "2", "python": "3.12"})
    second = make_manifest(package_versions={"python": "3.12", "pydantic": "2"})

    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64


def test_completed_manifest_requires_completion_time() -> None:
    with pytest.raises(ValidationError, match="require completed_at"):
        make_manifest(completed_at=None)


def test_manifest_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        make_manifest(completed_at=STARTED - timedelta(seconds=1))


def test_started_manifest_may_be_incomplete() -> None:
    manifest = make_manifest(status=RunStatus.STARTED, completed_at=None)
    assert manifest.completed_at is None
