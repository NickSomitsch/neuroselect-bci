from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

import neuroselect.publication.display as display_module
from neuroselect.publication.display import (
    PublicationDisplaySource,
    PublicationDisplaySpec,
    PublicationTable,
    RenderedPublicationFigure,
    read_publication_display,
    write_publication_display,
)


def _spec(tmp_path: Path) -> PublicationDisplaySpec:
    roles = (
        "primary_synthesis",
        "exploratory_test_exposed",
        "exploratory_locked",
        "exploratory_locked",
    )
    source_ids = (
        "primary-analysis",
        "candidate-generation-v2",
        "candidate-generation-step4",
        "opening-generalization",
    )
    return PublicationDisplaySpec(
        display_id="test-display",
        display_revision="paper-display-v1",
        generated_at=datetime(2026, 7, 29, tzinfo=UTC),
        publication_protocol=tmp_path / "protocol.yaml",
        expected_protocol_sha256="a" * 64,
        sources=tuple(
            PublicationDisplaySource(
                source_id=cast(Any, source_id),
                path=tmp_path / source_id,
                expected_manifest_sha256=str(index + 1) * 64,
                evidence_role=cast(Any, role),
            )
            for index, (source_id, role) in enumerate(zip(source_ids, roles, strict=True))
        ),
        style_revision="neuroselect-publication-style-v1",
        evidence_separation_policy=("primary_secondary_and_exploratory_results_are_never_pooled"),
    )


def _table(item_id: str = "table-1-test") -> PublicationTable:
    return PublicationTable(
        item_id=item_id,
        title="Test table",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        columns=("Metric", "Value"),
        rows=(("A|B", "0.500"),),
        caption="A test caption.",
    )


def _figure(item_id: str = "figure-1-test") -> RenderedPublicationFigure:
    return RenderedPublicationFigure(
        item_id=item_id,
        title="Test figure",
        evidence_roles=("exploratory",),
        source_ids=("opening-generalization",),
        caption="A figure caption.",
        files={
            "svg": b"<svg/>",
            "png": b"\x89PNG\r\n",
            "pdf": b"%PDF-1.4\n",
        },
    )


def test_display_spec_locks_source_order_and_formats(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert spec.digest() == spec.digest()
    payload = spec.model_dump()
    payload["sources"] = tuple(reversed(payload["sources"]))
    with pytest.raises(ValidationError, match="sources and their order"):
        PublicationDisplaySpec.model_validate(payload)
    payload = spec.model_dump()
    payload["figure_formats"] = ("png",)
    with pytest.raises(ValidationError, match="SVG, PNG, and PDF"):
        PublicationDisplaySpec.model_validate(payload)


def test_publication_table_renders_csv_and_escaped_markdown() -> None:
    table = _table()
    assert table.csv_content() == "Metric,Value\nA|B,0.500\n"
    assert "A\\|B" in table.markdown_content()
    with pytest.raises(ValidationError, match="row must match"):
        PublicationTable(
            item_id="table-2-invalid",
            title="Invalid",
            evidence_roles=("primary",),
            source_ids=("primary-analysis",),
            columns=("A", "B"),
            rows=(("only-one",),),
            caption="Invalid row.",
        )


def test_publication_display_round_trip_and_checksum_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        display_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    destination = tmp_path / "display"
    inventory, manifest = write_publication_display(
        _spec(tmp_path),
        (_table(),),
        (_figure(),),
        destination,
        git_sha="a1b2c3d",
        source_tree_sha256=None,
    )
    restored, restored_manifest = read_publication_display(
        destination,
        require_publication_ready=True,
    )
    assert (restored, restored_manifest) == (inventory, manifest)
    assert inventory.publication_ready
    assert len({item.artifact_id for item in manifest.outputs}) == len(manifest.outputs)
    assert (destination / "tables/table-1-test.csv").exists()
    assert (destination / "figures/figure-1-test.svg").read_bytes() == b"<svg/>"
    with pytest.raises(FileExistsError):
        write_publication_display(
            _spec(tmp_path),
            (_table(),),
            (_figure(),),
            destination,
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )
    (destination / "figures/figure-1-test.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_publication_display(destination)


def test_publication_display_marks_dirty_render_non_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        display_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    destination = tmp_path / "dirty"
    inventory, _ = write_publication_display(
        _spec(tmp_path),
        (_table(),),
        (_figure(),),
        destination,
        git_sha="a1b2c3d",
        source_tree_sha256="f" * 64,
    )
    assert not inventory.publication_ready
    with pytest.raises(ValueError, match="uncommitted source tree"):
        read_publication_display(destination, require_publication_ready=True)


@pytest.mark.parametrize(
    ("tables", "figures", "message"),
    (
        ((), (_figure(),), "at least one table"),
        ((_table(),), (), "at least one table"),
        ((_table(),), (_figure("table-1-test"),), "item IDs must be unique"),
    ),
)
def test_publication_display_rejects_incomplete_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tables: tuple[PublicationTable, ...],
    figures: tuple[RenderedPublicationFigure, ...],
    message: str,
) -> None:
    monkeypatch.setattr(
        display_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    with pytest.raises(ValueError, match=message):
        write_publication_display(
            _spec(tmp_path),
            tables,
            figures,
            tmp_path / "invalid",
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )


def test_publication_display_requires_every_figure_format(tmp_path: Path) -> None:
    figure = _figure()
    incomplete = RenderedPublicationFigure(
        item_id=figure.item_id,
        title=figure.title,
        evidence_roles=figure.evidence_roles,
        source_ids=figure.source_ids,
        caption=figure.caption,
        files=cast(Any, {"png": b"png"}),
    )
    with pytest.raises(ValueError, match="all locked output formats"):
        write_publication_display(
            _spec(tmp_path),
            (_table(),),
            (incomplete,),
            tmp_path / "invalid-formats",
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )
