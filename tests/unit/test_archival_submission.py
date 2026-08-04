from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from docx import Document

import neuroselect.publication.archival as archival
import neuroselect.publication.submission as submission
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.publication.archival import (
    ExcludedArtifactClass,
    ReleaseArtifactSource,
    ReleaseSpec,
    build_publication_release,
    load_release_spec,
    verify_publication_release,
)
from neuroselect.publication.latex import LatexManuscript
from neuroselect.publication.submission import (
    SubmissionAuthor,
    SubmissionDeclarations,
    SubmissionFile,
    SubmissionGate,
    SubmissionInventory,
    SubmissionSpec,
    build_journal_submission,
    load_submission_spec,
    selected_journal,
    verify_journal_submission,
)

ROOT = Path(__file__).parents[2]


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def _release_repository(tmp_path: Path) -> tuple[Path, ReleaseSpec]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.org")
    _git(repository, "config", "user.name", "Test Author")
    (repository / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    (repository / "ui").mkdir()
    (repository / "ui/package.json").write_text('{"version": "0.1.0"}\n', encoding="utf-8")
    (repository / "CITATION.cff").write_text("version: 0.1.0\n", encoding="utf-8")
    (repository / "uv.lock").write_text('version = "0.1.0"\n', encoding="utf-8")
    artifact = repository / "artifacts/result"
    artifact.mkdir(parents=True)
    result = b'{"result":0}\n'
    checkpoint = b"checkpoint"
    (artifact / "result.json").write_bytes(result)
    (artifact / "decoder.joblib").write_bytes(checkpoint)
    now = datetime(2026, 7, 31, tzinfo=UTC)
    manifest = RunManifest(
        run_id="release-test",
        run_kind=RunKind.PUBLICATION_ANALYSIS,
        status=RunStatus.COMPLETED,
        started_at=now,
        completed_at=now,
        git_sha="abcdef0",
        config_sha256="1" * 64,
        random_seeds={"global": 1},
        package_versions={"python": "3.12"},
        device={"system": "test"},
        outputs=(
            ArtifactRef(
                artifact_id="result",
                uri="artifact://result.json",
                sha256=archival.sha256_bytes(result),
            ),
            ArtifactRef(
                artifact_id="checkpoint",
                uri="artifact://decoder.joblib",
                sha256=archival.sha256_bytes(checkpoint),
            ),
        ),
    )
    (artifact / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    _git(repository, "add", "pyproject.toml", "ui/package.json", "CITATION.cff", "uv.lock")
    _git(repository, "commit", "-qm", "fixture")
    spec = ReleaseSpec(
        release_id="neuroselect-v0.1.0",
        version="0.1.0",
        tag="v0.1.0",
        repository_url="https://github.com/example/neuroselect",
        artifact_sources=(
            ReleaseArtifactSource(
                source_id="result",
                path=Path("artifacts/result"),
                expected_manifest_sha256=manifest.digest(),
                purpose="Test result",
            ),
        ),
        excluded_artifact_classes=(
            ExcludedArtifactClass(artifact_class="raw-eeg", reason="Not redistributed"),
        ),
    )
    return repository, spec


def test_tracked_release_and_submission_specs_encode_zero_cost_route() -> None:
    release = load_release_spec(ROOT / "configs/publication/release_v1.yaml")
    spec = load_submission_spec(ROOT / "configs/publication/submission_v1.yaml")

    assert release.version == "0.1.0"
    assert release.tag == "v0.1.0"
    assert release.zenodo_doi == "10.5281/zenodo.21793545"
    assert spec.author.orcid == "https://orcid.org/0009-0005-5436-4445"
    assert spec.gates["orcid_validated"].status == "satisfied"
    assert spec.gates["rbet_apc_fully_covered"].status == "not_applicable"
    assert len(release.artifact_sources) == 11
    assert selected_journal(spec) == "neuroinformatics"
    assert 150 <= len(submission._WORD.findall(spec.neuroinformatics_abstract.read_text())) <= 250


def test_release_build_is_deterministic_and_excludes_checkpoint(tmp_path: Path) -> None:
    repository, spec = _release_repository(tmp_path)
    first = build_publication_release(
        spec,
        repository=repository,
        output=tmp_path / "first",
        allow_pending=True,
    )
    second = build_publication_release(
        spec,
        repository=repository,
        output=tmp_path / "second",
        allow_pending=True,
    )

    assert not first.inventory.release_ready
    assert [item.sha256 for item in first.inventory.archives] == [
        item.sha256 for item in second.inventory.archives
    ]
    assert any(
        item.path and item.path.endswith("decoder.joblib") for item in first.inventory.exclusions
    )
    evidence_archive = next(
        archive for archive in first.inventory.archives if "research-outputs" in archive.filename
    )
    assert not any(item.path.endswith("decoder.joblib") for item in evidence_archive.files)
    verified = verify_publication_release(tmp_path / "first", require_ready=False)
    assert verified.git_revision == first.inventory.git_revision

    archive = tmp_path / "first" / first.inventory.archives[0].filename
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        verify_publication_release(tmp_path / "first", require_ready=False)


def test_release_helpers_reject_unsafe_and_unclassified_paths(tmp_path: Path) -> None:
    assert archival._tar_gz({"safe/file.txt": b"value"}) == archival._tar_gz(
        {"safe/file.txt": b"value"}
    )
    with pytest.raises(ValueError, match="unsafe archive path"):
        archival._tar_gz({"../private": b"value"})
    with pytest.raises(ValueError, match="not local"):
        archival._manifest_output_path("https://example.org/result.json")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        path = tmp_path / "invalid.yaml"
        path.write_text("[]\n", encoding="utf-8")
        load_release_spec(path)


def _submission_spec(tmp_path: Path, *, rbet_ready: bool) -> SubmissionSpec:
    abstract = tmp_path / "abstract.md"
    abstract.write_text(" ".join(f"word{index}" for index in range(170)), encoding="utf-8")
    checked = SubmissionGate(
        status="satisfied", checked_on=date(2026, 7, 31), public_note="Checked"
    )
    pending = SubmissionGate(status="pending", public_note="Awaiting institution")
    return SubmissionSpec(
        submission_id="neuroselect-journal-submission-v1",
        manuscript_config=tmp_path / "manuscript.yaml",
        manuscript_artifacts=tmp_path / "manuscript-artifacts",
        release_config=tmp_path / "release.yaml",
        reviewer_archive_url="https://anonymous.example/review" if rbet_ready else None,
        author=SubmissionAuthor(
            name="Nick Somitsch",
            orcid="https://orcid.org/0000-0002-1825-0097",
            affiliation="Independent Researcher",
            email="author@example.org",
            contributions="Conceptualization, Software, Analysis, Writing",
        ),
        declarations=SubmissionDeclarations(
            funding="No external funding.",
            competing_interests="The author declares no competing interests.",
            ethics="Approved secondary-use wording.",
        ),
        keywords=("BCI", "P300", "language", "personalization"),
        neuroinformatics_abstract=abstract,
        gates={
            "common": checked,
            "affiliation": checked if rbet_ready else pending,
            "apc": checked if rbet_ready else pending,
        },
        common_required_gates=("common",),
        rbet_required_gates=("affiliation", "apc"),
    )


def _mock_submission_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: SubmissionSpec
) -> None:
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda executable: None if executable == "pdffonts" else real_which(executable),
    )
    release = ReleaseSpec(
        release_id="neuroselect-v0.1.0",
        version="0.1.0",
        tag="v0.1.0",
        repository_url="https://github.com/example/neuroselect",
        zenodo_doi="10.5281/zenodo.12345",
        artifact_sources=(
            ReleaseArtifactSource(
                source_id="result",
                path=Path("result"),
                expected_manifest_sha256="1" * 64,
                purpose="Result",
            ),
        ),
        excluded_artifact_classes=(
            ExcludedArtifactClass(artifact_class="raw-eeg", reason="Excluded"),
        ),
    )
    monkeypatch.setattr(submission, "load_release_spec", lambda _: release)
    manuscript_spec = cast(
        Any,
        type(
            "Spec",
            (),
            {"output_filename": "manuscript.docx", "title": "Test NeuroSelect manuscript"},
        )(),
    )
    monkeypatch.setattr(submission, "load_manuscript_spec", lambda _: manuscript_spec)
    manuscript = (
        "# Abstract\n\nOriginal abstract.\n\n# 1. Introduction\n\nText.\n\n"
        "# 2. Materials and methods\n\nMethods.\n\n# 3. Results\n\nResults.\n\n"
        "# Declarations\n\nOld declarations.\n\n# References\n\n{{references}}\n"
    )
    monkeypatch.setattr(
        submission,
        "verify_manuscript_inputs",
        lambda _: (manuscript, (), cast(Any, None), cast(Any, None), cast(Any, None)),
    )
    latex = LatexManuscript(
        source=(
            "\\documentclass{article}\n\\begin{document}\n\\maketitle\n"
            "\\begin{abstract}Verified abstract.\\end{abstract}\n"
            "\\subsection{2.9 Use of AI-assisted tools}\n"
            "\\section{3. Results}\n"
            "\\section{Information Sharing Statement}\n"
            "\\section{Acknowledgments}\n"
            "\\section{Statements and Declarations}\n"
            "\\bibliographystyle{unsrt}\n\\bibliography{references}\n\\end{document}\n"
        ),
        bibliography="@misc{test, title={Test}, year={2026}}\n",
        figures={"figures/figure.pdf": b"%PDF-1.4\n"},
    )
    monkeypatch.setattr(submission, "render_latex_manuscript", lambda *args: latex)
    monkeypatch.setattr(
        submission,
        "_shared_supplements",
        lambda _: {
            "figures/Figure1.pdf": b"%PDF-1.4\n",
            "tables/Table1.csv": b"value\n",
            "supplement/inventory.json": b"{}\n",
        },
    )
    monkeypatch.setattr(
        submission,
        "_springer_template",
        lambda _: {"sn-jnl.cls": b"class", "sn-basic.bst": b"style"},
    )
    monkeypatch.setattr(submission, "_compile_latex", lambda *args: b"%PDF-1.4\n")
    spec.manuscript_artifacts.mkdir(parents=True)
    document = Document()
    document.add_paragraph("Nick Somitsch https://github.com/example/neuroselect v0.1.0")
    document.save(str(spec.manuscript_artifacts / "manuscript.docx"))


@pytest.mark.parametrize("journal", ["neuroinformatics", "rbet"])
def test_submission_builders_and_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.org")
    _git(repository, "config", "user.name", "Test Author")
    (repository / "README.md").write_text("test\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    (repository / "ui").mkdir()
    (repository / "ui/package.json").write_text('{"version": "0.1.0"}\n', encoding="utf-8")
    (repository / "CITATION.cff").write_text(
        "version: 0.1.0\ndoi: 10.5281/zenodo.12345\n", encoding="utf-8"
    )
    (repository / "uv.lock").write_text('version = "0.1.0"\n', encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    _git(repository, "tag", "v0.1.0")
    spec = _submission_spec(tmp_path, rbet_ready=journal == "rbet")
    _mock_submission_dependencies(tmp_path, monkeypatch, spec)
    output = tmp_path / f"output-{journal}"
    inventory = build_journal_submission(
        spec,
        journal=cast(Any, journal),
        repository=repository,
        output=output,
    )

    assert inventory.submission_ready
    assert verify_journal_submission(output, repository=repository).journal == journal
    if journal == "neuroinformatics":
        source = (output / "manuscript/manuscript.tex").read_text(encoding="utf-8")
        assert r"\documentclass[pdflatex,sn-basic]{sn-jnl}" in source
        assert r"\author*[1]{\fnm{Nick} \sur{Somitsch}}" in source
        class_record = next(item for item in inventory.files if item.filename.endswith(".cls"))
        assert class_record.license == "LPPL-1.0-or-later"
    else:
        anonymous = submission._anonymous_payload_text(output / "review/anonymous-manuscript.docx")
        assert not submission._DIRECT_IDENTIFIER.search(anonymous)

    first = inventory.files[0]
    (output / first.filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        verify_journal_submission(output)


def test_submission_gates_anonymity_and_validation(tmp_path: Path) -> None:
    spec = _submission_spec(tmp_path, rbet_ready=False)
    assert selected_journal(spec) == "neuroinformatics"
    with pytest.raises(ValueError, match="RBET is blocked"):
        build_journal_submission(
            spec,
            journal="rbet",
            repository=tmp_path,
            output=tmp_path / "blocked",
        )
    redacted = submission._anonymous_text(
        "Nick Somitsch GitHub v0.1.0 https://orcid.org/0000-0002-1825-0097"
    )
    assert not submission._DIRECT_IDENTIFIER.search(redacted)
    assert submission._deterministic_zip({"safe.txt": b"safe"}) == submission._deterministic_zip(
        {"safe.txt": b"safe"}
    )
    with pytest.raises(ValueError, match="unsafe submission archive path"):
        submission._deterministic_zip({"../unsafe": b"unsafe"})
    with pytest.raises(ValueError, match="require.*checked_on date"):
        SubmissionGate(status="satisfied", public_note="Missing date")


def test_submission_contracts_fail_closed_on_invalid_metadata_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="full https://orcid.org URL"):
        SubmissionAuthor(name="Author", orcid="0000-0000")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_submission_spec(invalid)
    with pytest.raises(ValueError, match="section is missing"):
        submission._replace_section("source", "# Start\n", "# End", "replacement")
    with pytest.raises(ValueError, match="terminator is missing"):
        submission._replace_section("# Start\nsource", "# Start\n", "# End", "replacement")

    spec = _submission_spec(tmp_path, rbet_ready=False)
    release = load_release_spec(ROOT / "configs/publication/release_v1.yaml").model_copy(
        update={"zenodo_doi": None}
    )
    monkeypatch.setattr(submission, "load_release_spec", lambda _: release)
    pending_spec = spec.model_copy(
        update={
            "author": SubmissionAuthor(name="Nick Somitsch"),
            "declarations": SubmissionDeclarations(),
        }
    )
    pending = submission._pending_gates(pending_spec, "rbet")
    assert {
        "orcid_metadata",
        "author_affiliation",
        "corresponding_author_email",
        "credit_contributions",
        "funding_declaration",
        "competing_interests_declaration",
        "secondary_use_ethics_wording",
        "reserved_zenodo_doi",
        "anonymous_reviewer_archive",
    }.issubset(pending)

    short_abstract = tmp_path / "short.md"
    short_abstract.write_text("too short", encoding="utf-8")
    short_spec = spec.model_copy(update={"neuroinformatics_abstract": short_abstract})
    manuscript = (
        "# Abstract\n\nAbstract.\n\n# 1. Introduction\n\nText.\n\n"
        "# 2. Materials and methods\n\nMethods.\n\n# 3. Results\n\nResults.\n\n"
        "# Declarations\n\nDeclarations.\n\n# References\n"
    )
    with pytest.raises(ValueError, match="150-250 words"):
        submission._submission_source(short_spec, "neuroinformatics", manuscript)

    with pytest.raises(ValueError, match="no more than three heading levels"):
        submission._submission_source(spec, "rbet", manuscript.replace("Text.", "#### Too deep"))
    prepared = submission._submission_source(spec, "rbet", manuscript)
    assert prepared.index("## 2.9 Use of AI-assisted tools") < prepared.index("# 3. Results")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Tectonic is required"):
        submission._compile_latex("source", "references", {}, {})
    base = LatexManuscript(
        source="\\begin{document}\\maketitle\\end{document}", bibliography="", figures={}
    )
    manuscript_spec = cast(Any, type("Spec", (), {"title": "Test"})())
    with pytest.raises(ValueError, match="missing its abstract"):
        submission._springer_latex(base, spec, manuscript_spec, "abstract")


def test_submission_pdf_font_check_rejects_unembedded_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "manuscript.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    inventory = SubmissionInventory(
        submission_id="neuroselect-journal-submission-v1",
        journal="neuroinformatics",
        article_type="Original Article",
        route="subscription-no-apc",
        git_revision="1" * 40,
        release_tag="v0.1.0",
        zenodo_doi="10.5281/zenodo.12345",
        submission_ready=True,
        development_preview=False,
        pending_gates=(),
        abstract_word_count=200,
        keyword_count=4,
        files=(
            SubmissionFile(
                filename=pdf.name,
                portal_designation="Main manuscript",
                size=pdf.stat().st_size,
                sha256=archival.sha256_bytes(pdf.read_bytes()),
                license="CC-BY-4.0",
                public_status="public",
            ),
        ),
        archive_filename="submission.zip",
        archive_sha256="2" * 64,
    )
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/pdffonts")

    def font_result(embedded: str) -> subprocess.CompletedProcess[str]:
        output = (
            "name type encoding emb sub uni object ID\n"
            "---- ---- -------- --- --- --- ------ --\n"
            f"Font Type 1C Custom {embedded} yes yes 1 0\n"
        )
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: font_result("yes"))
    submission._verify_pdf_fonts(tmp_path, inventory)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: font_result("no"))
    with pytest.raises(ValueError, match="unembedded font"):
        submission._verify_pdf_fonts(tmp_path, inventory)
