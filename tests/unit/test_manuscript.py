from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

import neuroselect.publication.display as display_module
import neuroselect.publication.latex as latex_module
import neuroselect.publication.manuscript as manuscript_module
from neuroselect.provenance import RunManifest
from neuroselect.publication import load_publication_protocol
from neuroselect.publication.display import (
    PublicationDisplayInventory,
    PublicationDisplayItem,
    PublicationDisplaySource,
    PublicationDisplaySpec,
    PublicationTable,
    RenderedPublicationFigure,
    write_publication_display,
)
from neuroselect.publication.latex import (
    render_bibliography,
    render_latex_manuscript,
)
from neuroselect.publication.manuscript import (
    ManuscriptClaim,
    ManuscriptDisplaySource,
    ManuscriptSpec,
    audit_claims,
    citation_order,
    load_claim_ledger,
    load_manuscript_spec,
    load_references,
    read_manuscript_bundle,
    replace_citations,
    validate_manuscript_markers,
    verify_manuscript_inputs,
    write_manuscript_bundle,
)

ROOT = Path(__file__).parents[2]
PROTOCOL_PATH = ROOT / "configs/publication/offline_methods_v1.yaml"
PROTOCOL_SHA = load_publication_protocol(PROTOCOL_PATH).digest()


def _display_spec(tmp_path: Path) -> PublicationDisplaySpec:
    source_ids = (
        "primary-analysis",
        "candidate-generation-v2",
        "candidate-generation-step4",
        "opening-generalization",
    )
    roles = (
        "primary_synthesis",
        "exploratory_test_exposed",
        "exploratory_locked",
        "exploratory_locked",
    )
    return PublicationDisplaySpec(
        display_id="manuscript-test-display",
        display_revision="paper-display-v1",
        generated_at=datetime(2026, 7, 29, tzinfo=UTC),
        publication_protocol=PROTOCOL_PATH,
        expected_protocol_sha256=PROTOCOL_SHA,
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


def _write_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, PublicationDisplayInventory, RunManifest]:
    monkeypatch.setattr(
        display_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    table = PublicationTable(
        item_id="table-1-test",
        title="Test table",
        evidence_roles=("primary",),
        source_ids=("primary-analysis",),
        columns=("Name", "Value"),
        rows=(("A", "0.500"),),
        caption="Table caption.",
    )
    figure = RenderedPublicationFigure(
        item_id="figure-1-test",
        title="Test figure",
        evidence_roles=("exploratory",),
        source_ids=("opening-generalization",),
        caption="Figure caption.",
        files={"svg": b"<svg/>", "png": b"\x89PNG\r\n", "pdf": b"%PDF-1.4\n"},
    )
    path = tmp_path / "display"
    inventory, manifest = write_publication_display(
        _display_spec(tmp_path),
        (table,),
        (figure,),
        path,
        git_sha="a1b2c3d",
        source_tree_sha256=None,
    )
    return path, inventory, manifest


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "manuscript.md"
    source.write_text(
        "# Results\n\nVerified value 0.500 [@test-ref].\n\n"
        "{{table:table-1-test}}\n\n{{figure:figure-1-test}}\n\n"
        "# References\n\n{{references}}\n",
        encoding="utf-8",
    )
    references = tmp_path / "references.yaml"
    references.write_text(
        "- reference_id: test-ref\n"
        "  formatted: Test A. Verified source. 2026.\n"
        "  persistent_url: https://example.org/source\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims.yaml"
    claims.write_text(
        "- claim_id: result-value\n"
        f"  source_path: {tmp_path / 'values.csv'}\n"
        "  source_kind: csv\n"
        "  row_match:\n"
        "    Name: A\n"
        "  field: Value\n"
        '  expected: "0.500"\n'
        "  required_text: Verified value 0.500\n"
        "  evidence_role: primary\n",
        encoding="utf-8",
    )
    (tmp_path / "values.csv").write_text("Name,Value\nA,0.500\n", encoding="utf-8")
    return source, references, claims


def _manuscript_spec(
    tmp_path: Path,
    display_path: Path,
    display_manifest: RunManifest,
) -> ManuscriptSpec:
    source, references, claims = _write_sources(tmp_path)
    display_inventory = json.loads((display_path / "inventory.json").read_text())
    return ManuscriptSpec(
        manuscript_id="test-manuscript",
        manuscript_revision="journal-manuscript-v1",
        assembled_at=datetime(2026, 7, 29, tzinfo=UTC),
        title="Test manuscript",
        article_type="Original Research",
        author="Test Author",
        source_markdown=source,
        references=references,
        latex_source=tmp_path / "test-manuscript.tex",
        latex_bibliography=tmp_path / "references.bib",
        claim_ledger=claims,
        publication_protocol=PROTOCOL_PATH,
        expected_protocol_sha256=PROTOCOL_SHA,
        display=ManuscriptDisplaySource(
            path=display_path,
            expected_manifest_sha256=display_manifest.digest(),
            expected_config_sha256=display_inventory["config_sha256"],
        ),
        included_tables=("table-1-test",),
        included_figures=("figure-1-test",),
        style_revision="narrative-proposal-journal-manuscript-v1",
        output_filename="test-manuscript.docx",
    )


def test_tracked_manuscript_recipe_is_complete_without_generated_artifacts() -> None:
    spec = load_manuscript_spec(ROOT / "configs/publication/manuscript_v1.yaml")
    source = spec.source_markdown.read_text(encoding="utf-8")
    references = load_references(spec.references)
    claims = load_claim_ledger(spec.claim_ledger)
    ordered_references = citation_order(source, references)

    assert source.startswith("# Abstract")
    assert len(ordered_references) == 18
    assert len(claims) == 49
    assert len(spec.included_tables) == 10
    assert len(spec.included_figures) == 5
    assert all(claim.required_text in source for claim in claims)
    assert spec.latex_source.read_text(encoding="utf-8").startswith(r"\documentclass")
    bibliography = spec.latex_bibliography.read_text(encoding="utf-8")
    assert all(
        f"@{reference.bibtex_type}{{{reference.reference_id}," in bibliography
        for reference in references
    )


def test_citations_are_ordered_replaced_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "references.yaml"
    path.write_text(
        "- reference_id: beta\n"
        "  formatted: Beta reference.\n"
        "  persistent_url: https://example.org/beta\n"
        "- reference_id: alpha\n"
        "  formatted: Alpha reference.\n"
        "  persistent_url: https://example.org/alpha\n",
        encoding="utf-8",
    )
    references = load_references(path)
    source = "First [@alpha], then [@beta; @alpha]."
    ordered = citation_order(source, references)
    assert [item.reference_id for item in ordered] == ["alpha", "beta"]
    assert replace_citations(source, ordered) == "First [1], then [2, 1]."
    with pytest.raises(ValueError, match="unknown manuscript reference"):
        citation_order("Unknown [@missing].", references)
    with pytest.raises(ValueError, match="uncited manuscript references"):
        citation_order("Only [@alpha].", references)
    with pytest.raises(ValueError, match="must contain citations"):
        citation_order("No citation.", ())


def test_structured_bibtex_is_rendered_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "references.yaml"
    path.write_text(
        "- reference_id: article\n"
        "  formatted: Test A. Structured article. 2026.\n"
        "  persistent_url: https://example.org/article\n"
        "  bibtex_type: article\n"
        "  bibtex_fields:\n"
        "    author: Test, Alice\n"
        "    title: Structured article\n"
        '    year: "2026"\n'
        "    journal: Test Journal\n",
        encoding="utf-8",
    )
    references = load_references(path)
    bibliography = render_bibliography(references)
    assert "@article{article," in bibliography
    assert "author = {Test, Alice}," in bibliography
    assert "title = {{Structured article}}," in bibliography
    assert "url = {https://example.org/article}," in bibliography
    assert "note =" not in bibliography

    path.write_text(
        "- reference_id: invalid\n"
        "  formatted: Invalid structured article.\n"
        "  persistent_url: https://example.org/invalid\n"
        "  bibtex_type: article\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="require structured BibTeX fields"):
        load_references(path)

    path.write_text(
        "- reference_id: incomplete\n"
        "  formatted: Incomplete structured article.\n"
        "  persistent_url: https://example.org/incomplete\n"
        "  bibtex_type: article\n"
        "  bibtex_fields:\n"
        "    title: Incomplete\n"
        '    year: "2026"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing fields"):
        load_references(path)


def test_claim_audit_supports_csv_yaml_json_and_detects_failures(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("Name,Value\nA,0.500\n", encoding="utf-8")
    yaml_path = tmp_path / "values.yaml"
    yaml_path.write_text("items: [a, b]\n", encoding="utf-8")
    json_path = tmp_path / "values.json"
    json_path.write_text('{"nested":{"value":3}}', encoding="utf-8")
    claims = (
        ManuscriptClaim(
            claim_id="csv-value",
            source_path=csv_path,
            source_kind="csv",
            row_match={"Name": "A"},
            field="Value",
            expected="0.500",
            required_text="CSV 0.500",
            evidence_role="primary",
        ),
        ManuscriptClaim(
            claim_id="yaml-length",
            source_path=yaml_path,
            source_kind="yaml",
            field="items",
            transform="length",
            expected=2,
            required_text="two YAML items",
            evidence_role="method",
        ),
        ManuscriptClaim(
            claim_id="json-value",
            source_path=json_path,
            source_kind="json",
            field="nested.value",
            expected=3,
            required_text="JSON value 3",
            evidence_role="secondary",
        ),
    )
    audit = audit_claims("CSV 0.500, two YAML items, JSON value 3.", claims)
    assert audit.claim_count == 3
    assert audit.entries[1].resolved_value == 2
    with pytest.raises(ValueError, match="phrase is absent"):
        audit_claims("CSV 0.500 only.", claims)
    wrong = claims[0].model_copy(update={"expected": "0.600"})
    with pytest.raises(ValueError, match="expected"):
        audit_claims("CSV 0.500", (wrong,))
    duplicate = csv_path.read_text() + "A,0.500\n"
    csv_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="expected one CSV row"):
        audit_claims("CSV 0.500", (claims[0],))


def test_claim_and_reference_loaders_reject_invalid_documents(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not-a-list: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="references must contain a YAML list"):
        load_references(invalid)
    with pytest.raises(ValueError, match="claim ledger must contain a YAML list"):
        load_claim_ledger(invalid)
    duplicates = tmp_path / "duplicates.yaml"
    duplicates.write_text(
        "- reference_id: same\n"
        "  formatted: One.\n"
        "  persistent_url: https://example.org/one\n"
        "- reference_id: same\n"
        "  formatted: Two.\n"
        "  persistent_url: https://example.org/two\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference IDs must be unique"):
        load_references(duplicates)


def test_marker_validation_checks_exact_ids_kinds_and_references() -> None:
    inventory = PublicationDisplayInventory(
        display_id="test",
        display_revision="paper-display-v1",
        generated_at=datetime(2026, 7, 29, tzinfo=UTC),
        config_sha256="a" * 64,
        protocol_sha256="b" * 64,
        source_manifest_sha256={"primary-analysis": "c" * 64},
        source_tree_clean=True,
        publication_ready=True,
        evidence_separation_policy="separate",
        items=(
            PublicationDisplayItem(
                item_id="table-1-test",
                item_kind="table",
                title="Table",
                evidence_roles=("primary",),
                source_ids=("primary-analysis",),
                files=("tables/table-1-test.csv",),
                caption="Caption",
            ),
            PublicationDisplayItem(
                item_id="figure-1-test",
                item_kind="figure",
                title="Figure",
                evidence_roles=("exploratory",),
                source_ids=("opening-generalization",),
                files=("figures/figure-1-test.png",),
                caption="Caption",
            ),
        ),
        limitations=("Offline only.",),
    )
    spec = cast(
        ManuscriptSpec,
        cast(
            Any,
            type(
                "Spec",
                (),
                {
                    "included_tables": ("table-1-test",),
                    "included_figures": ("figure-1-test",),
                },
            )(),
        ),
    )
    valid = "{{table:table-1-test}}\n{{figure:figure-1-test}}\n{{references}}"
    validate_manuscript_markers(valid, spec, inventory)
    with pytest.raises(ValueError, match="exactly once"):
        validate_manuscript_markers(
            valid + "\n{{table:table-1-test}}",
            spec,
            inventory,
        )
    with pytest.raises(ValueError, match="differ from the locked recipe"):
        validate_manuscript_markers(
            "{{table:table-1-test}}\n{{references}}",
            spec,
            inventory,
        )
    with pytest.raises(ValueError, match="exactly one references marker"):
        validate_manuscript_markers(
            "{{table:table-1-test}}\n{{figure:figure-1-test}}",
            spec,
            inventory,
        )


def test_verify_inputs_with_synthetic_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display_path, inventory, display_manifest = _write_display(tmp_path, monkeypatch)
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    source, references, audit, restored_inventory, restored_manifest = verify_manuscript_inputs(
        spec
    )
    assert "Verified value 0.500" in source
    assert references[0].reference_id == "test-ref"
    assert audit.claim_count == 1
    assert restored_inventory == inventory
    assert restored_manifest == display_manifest
    bad = spec.model_copy(
        update={"display": spec.display.model_copy(update={"expected_manifest_sha256": "f" * 64})}
    )
    with pytest.raises(ValueError, match="manifest differs"):
        verify_manuscript_inputs(bad)


def test_latex_render_contains_citations_tables_figures_and_bibliography(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display_path, display_inventory, display_manifest = _write_display(tmp_path, monkeypatch)
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    source, references, _, _, _ = verify_manuscript_inputs(spec)
    package = render_latex_manuscript(spec, source, references, display_inventory)

    assert package.source.startswith(r"\documentclass")
    assert r"Verified value 0.500 \cite{test-ref}." in package.source
    assert r"\begin{tabular}" in package.source
    assert r"\includegraphics" in package.source
    assert set(package.figures) == {"figures/figure-1-test.pdf"}
    assert package.bibliography == render_bibliography(references)
    assert "@misc{test-ref," in package.bibliography
    assert "Affiliation, institutional email, and ORCID to be confirmed" not in package.source
    assert r"\date{}" in package.source


def test_latex_inline_and_structural_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = latex_module._inline_latex(
        r"**A&B** costs 5% ± 1, uses x_y, \(y \in C(x)\), "
        "cites [@test-ref], and links https://example.org/a_b."
    )
    assert r"\textbf{A\&B}" in rendered
    assert r"5\% \ensuremath{\pm} 1" in rendered
    assert r"x\_y" in rendered
    assert r"$y \in C(x)$" in rendered
    assert r"\cite{test-ref}" in rendered
    assert r"\url{https://example.org/a_b}." in rendered
    with pytest.raises(ValueError, match="invalid manuscript citation"):
        latex_module._citation_command("[@INVALID]")
    with pytest.raises(ValueError, match="invalid table ID"):
        latex_module._table_label("invalid")
    with pytest.raises(ValueError, match="invalid figure ID"):
        latex_module._figure_label("invalid")

    display_path, display_inventory, display_manifest = _write_display(tmp_path, monkeypatch)
    (display_path / "primary-analysis").mkdir(exist_ok=True)
    csv_path = display_path / "primary-analysis" / "wide.csv"
    csv_path.write_text(
        "A,B,C,D,E,F,G\n1,2,3,4,5,6,7\n",
        encoding="utf-8",
    )
    table = display_inventory.items[0].model_copy(update={"files": ("primary-analysis/wide.csv",)})
    wide_inventory = display_inventory.model_copy(
        update={"items": (table, display_inventory.items[1])}
    )
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    source = (
        "# Abstract\n\nSummary.\n\n# 1. Introduction\n\n- First item\n\n"
        "{{table:table-1-test}}\n\n{{figure:figure-1-test}}\n\n"
        "# Declarations\n\n## Data availability\n\nAvailable.\n\n# References\n\n"
        "{{references}}\n"
    )
    references = load_references(spec.references)
    package = render_latex_manuscript(spec, source, references, wide_inventory)
    assert r"\begin{abstract}" in package.source
    assert r"\end{abstract}" in package.source
    assert r"\begin{itemize}" in package.source
    assert r"\begin{landscape}" in package.source
    assert r"\section*{Declarations}" in package.source


def test_manuscript_bundle_round_trip_and_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display_path, display_inventory, display_manifest = _write_display(tmp_path, monkeypatch)
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    source, references, audit, _, _ = verify_manuscript_inputs(spec)
    monkeypatch.setattr(
        manuscript_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    destination = tmp_path / "manuscript"
    inventory, manifest = write_manuscript_bundle(
        spec,
        rendered_markdown=replace_citations(source, references),
        document_bytes=b"PK-test-docx",
        additional_files={
            "test-manuscript.tex": b"\\documentclass{article}",
            "references.bib": b"@misc{test-ref}",
            "test-manuscript.pdf": b"%PDF-test",
            "figures/figure-1-test.pdf": b"%PDF-figure",
        },
        claim_audit=audit,
        ordered_references=references,
        display_inventory=display_inventory,
        display_manifest=display_manifest,
        output_dir=destination,
        git_sha="a1b2c3d",
        source_tree_sha256=None,
    )
    restored_inventory, restored_audit, restored_manifest = read_manuscript_bundle(
        destination,
        require_assembly_ready=True,
    )
    assert (restored_inventory, restored_audit, restored_manifest) == (
        inventory,
        audit,
        manifest,
    )
    assert inventory.assembly_ready
    assert not inventory.submission_ready
    assert inventory.latex_compiled
    assert inventory.document_formats == ("docx", "markdown", "latex", "bibtex", "pdf")
    assert set(inventory.pending_submission_gates) == {
        "uibk-open-access",
        "secondary-use-ethics",
        "domain-review",
        "author-metadata",
    }
    with pytest.raises(FileExistsError):
        write_manuscript_bundle(
            spec,
            rendered_markdown=source,
            document_bytes=b"PK-test-docx",
            claim_audit=audit,
            ordered_references=references,
            display_inventory=display_inventory,
            display_manifest=display_manifest,
            output_dir=destination,
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )
    (destination / "manuscript.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_manuscript_bundle(destination)


def test_manuscript_bundle_marks_dirty_source_and_rejects_non_docx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display_path, display_inventory, display_manifest = _write_display(tmp_path, monkeypatch)
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    source, references, audit, _, _ = verify_manuscript_inputs(spec)
    monkeypatch.setattr(
        manuscript_module,
        "capture_runtime_environment",
        lambda: ({"python": "3.12"}, {"platform": "test"}),
    )
    with pytest.raises(ValueError, match="not a DOCX"):
        write_manuscript_bundle(
            spec,
            rendered_markdown=source,
            document_bytes=b"invalid",
            claim_audit=audit,
            ordered_references=references,
            display_inventory=display_inventory,
            display_manifest=display_manifest,
            output_dir=tmp_path / "invalid",
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )
    with pytest.raises(ValueError, match="not a PDF"):
        write_manuscript_bundle(
            spec,
            rendered_markdown=source,
            document_bytes=b"PK-test-docx",
            additional_files={"test-manuscript.pdf": b"invalid"},
            claim_audit=audit,
            ordered_references=references,
            display_inventory=display_inventory,
            display_manifest=display_manifest,
            output_dir=tmp_path / "invalid-pdf",
            git_sha="a1b2c3d",
            source_tree_sha256=None,
        )
    destination = tmp_path / "dirty"
    inventory, _ = write_manuscript_bundle(
        spec,
        rendered_markdown=source,
        document_bytes=b"PK-test-docx",
        claim_audit=audit,
        ordered_references=references,
        display_inventory=display_inventory,
        display_manifest=display_manifest,
        output_dir=destination,
        git_sha="a1b2c3d",
        source_tree_sha256="d" * 64,
    )
    assert not inventory.assembly_ready
    with pytest.raises(ValueError, match="uncommitted source tree"):
        read_manuscript_bundle(destination, require_assembly_ready=True)


def test_manuscript_spec_rejects_duplicates_and_unzoned_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display_path, _, display_manifest = _write_display(tmp_path, monkeypatch)
    spec = _manuscript_spec(tmp_path, display_path, display_manifest)
    payload = spec.model_dump(mode="json")
    payload["included_tables"] = ["table-1-test", "table-1-test"]
    with pytest.raises(ValidationError, match="table IDs must be unique"):
        ManuscriptSpec.model_validate(payload)
    payload = spec.model_dump(mode="json")
    payload["assembled_at"] = "2026-07-29T12:00:00"
    with pytest.raises(ValidationError, match="include a timezone"):
        ManuscriptSpec.model_validate(payload)
