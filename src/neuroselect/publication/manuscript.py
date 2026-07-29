"""Verified source, citation, claim, and artifact contracts for the manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation import capture_runtime_environment
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.publication.display import (
    PublicationDisplayInventory,
    read_publication_display,
)
from neuroselect.publication.protocol import load_publication_protocol

DEFAULT_MANUSCRIPT_CONFIG = Path("configs/publication/manuscript_v1.yaml")
_CITATION_PATTERN = re.compile(r"\[@([a-z0-9-]+(?:;\s*@[a-z0-9-]+)*)\]")
_ITEM_PATTERN = re.compile(r"\{\{(figure|table):([a-z0-9-]+)\}\}")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ManuscriptDisplaySource(BaseModel):
    """The immutable publication display assembled into the manuscript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    expected_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    require_publication_ready: Literal[True] = True


class ManuscriptSpec(BaseModel):
    """Journal-neutral manuscript recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    manuscript_id: str = Field(min_length=1, max_length=160)
    manuscript_revision: Literal["journal-manuscript-v1"]
    assembled_at: datetime
    title: str = Field(min_length=1, max_length=300)
    article_type: str = Field(min_length=1, max_length=100)
    author: str = Field(min_length=1, max_length=160)
    source_markdown: Path
    references: Path
    latex_source: Path
    latex_bibliography: Path
    claim_ledger: Path
    publication_protocol: Path
    expected_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    display: ManuscriptDisplaySource
    included_tables: tuple[str, ...] = Field(min_length=1)
    included_figures: tuple[str, ...] = Field(min_length=1)
    style_revision: Literal["narrative-proposal-journal-manuscript-v1"]
    output_filename: str = Field(pattern=r"^[a-z0-9-]+\.docx$")
    outcome_based_omission_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_recipe(self) -> ManuscriptSpec:
        if self.assembled_at.tzinfo is None or self.assembled_at.utcoffset() is None:
            raise ValueError("manuscript assembly time must include a timezone")
        if len(set(self.included_tables)) != len(self.included_tables):
            raise ValueError("manuscript table IDs must be unique")
        if len(set(self.included_figures)) != len(self.included_figures):
            raise ValueError("manuscript figure IDs must be unique")
        if self.latex_source.suffix != ".tex":
            raise ValueError("manuscript LaTeX source must use a .tex suffix")
        if self.latex_bibliography.suffix != ".bib":
            raise ValueError("manuscript bibliography must use a .bib suffix")
        return self

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.model_dump(mode="json")).encode())


class ManuscriptReference(BaseModel):
    """One verified bibliography entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(pattern=r"^[a-z0-9-]+$")
    formatted: str = Field(min_length=1)
    persistent_url: str = Field(pattern=r"^https://")


class ManuscriptClaim(BaseModel):
    """One quantitative statement tied to a tracked source value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^[a-z0-9-]+$")
    source_path: Path
    source_kind: Literal["csv", "yaml", "json"]
    row_match: dict[str, str] = Field(default_factory=dict)
    field: str = Field(min_length=1)
    transform: Literal["identity", "length"] = "identity"
    expected: str | int | float | bool | tuple[str, ...]
    required_text: str = Field(min_length=1)
    evidence_role: Literal["method", "primary", "secondary", "exploratory"]

    @model_validator(mode="after")
    def validate_locator(self) -> ManuscriptClaim:
        if self.source_kind == "csv" and not self.row_match:
            raise ValueError("CSV manuscript claims require a row selector")
        if self.source_kind != "csv" and self.row_match:
            raise ValueError("only CSV manuscript claims may use a row selector")
        return self


class ClaimAuditEntry(BaseModel):
    """Resolved claim value and manuscript phrase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    source_path: str
    field: str
    resolved_value: str | int | float | bool | tuple[str, ...]
    required_text: str
    evidence_role: str


class ManuscriptClaimAudit(BaseModel):
    """Fail-closed audit for every registered quantitative manuscript claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    claim_count: int = Field(ge=1)
    entries: tuple[ClaimAuditEntry, ...] = Field(min_length=1)
    all_source_values_verified: Literal[True] = True
    all_required_phrases_present: Literal[True] = True

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


class ManuscriptInventory(BaseModel):
    """Assembly inventory that deliberately differs from submission readiness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    manuscript_id: str
    manuscript_revision: str
    assembled_at: datetime
    title: str
    author: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    display_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    display_publication_ready: bool
    source_tree_clean: bool
    assembly_ready: bool
    submission_ready: bool
    pending_submission_gates: tuple[str, ...]
    included_tables: tuple[str, ...]
    included_figures: tuple[str, ...]
    citation_count: int = Field(ge=1)
    quantitative_claim_count: int = Field(ge=1)
    document_formats: tuple[str, ...] = Field(min_length=2)
    latex_compiled: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_manuscript_spec(path: str | Path = DEFAULT_MANUSCRIPT_CONFIG) -> ManuscriptSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: object = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("manuscript config must contain a YAML mapping")
    return ManuscriptSpec.model_validate(payload)


def load_references(path: str | Path) -> tuple[ManuscriptReference, ...]:
    with Path(path).open(encoding="utf-8") as reference_file:
        payload: object = yaml.safe_load(reference_file)
    if not isinstance(payload, list):
        raise ValueError("manuscript references must contain a YAML list")
    references = tuple(ManuscriptReference.model_validate(item) for item in payload)
    ids = [item.reference_id for item in references]
    if len(ids) != len(set(ids)):
        raise ValueError("manuscript reference IDs must be unique")
    return references


def load_claim_ledger(path: str | Path) -> tuple[ManuscriptClaim, ...]:
    with Path(path).open(encoding="utf-8") as ledger_file:
        payload: object = yaml.safe_load(ledger_file)
    if not isinstance(payload, list):
        raise ValueError("manuscript claim ledger must contain a YAML list")
    claims = tuple(ManuscriptClaim.model_validate(item) for item in payload)
    ids = [item.claim_id for item in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("manuscript claim IDs must be unique")
    return claims


def citation_order(
    manuscript_source: str,
    references: tuple[ManuscriptReference, ...],
) -> tuple[ManuscriptReference, ...]:
    """Return references in first-citation order and reject missing or unused entries."""

    reference_index = {item.reference_id: item for item in references}
    ordered_ids: list[str] = []
    for match in _CITATION_PATTERN.finditer(manuscript_source):
        for raw_id in match.group(1).split(";"):
            reference_id = raw_id.strip().removeprefix("@")
            if reference_id not in reference_index:
                raise ValueError(f"unknown manuscript reference: {reference_id}")
            if reference_id not in ordered_ids:
                ordered_ids.append(reference_id)
    if not ordered_ids:
        raise ValueError("manuscript must contain citations")
    unused = set(reference_index) - set(ordered_ids)
    if unused:
        raise ValueError(f"uncited manuscript references: {sorted(unused)}")
    return tuple(reference_index[reference_id] for reference_id in ordered_ids)


def replace_citations(
    text: str,
    ordered_references: tuple[ManuscriptReference, ...],
) -> str:
    """Replace stable citation keys with numbered journal-style references."""

    positions = {item.reference_id: index for index, item in enumerate(ordered_references, start=1)}

    def replacement(match: re.Match[str]) -> str:
        ids = [item.strip().removeprefix("@") for item in match.group(1).split(";")]
        return "[" + ", ".join(str(positions[reference_id]) for reference_id in ids) + "]"

    return _CITATION_PATTERN.sub(replacement, text)


def validate_manuscript_markers(
    manuscript_source: str,
    spec: ManuscriptSpec,
    inventory: PublicationDisplayInventory,
) -> None:
    """Ensure the configured evidence is included exactly once and by role."""

    marker_pairs = _ITEM_PATTERN.findall(manuscript_source)
    marker_ids = [item_id for _, item_id in marker_pairs]
    if len(marker_ids) != len(set(marker_ids)):
        raise ValueError("each manuscript table or figure marker must appear exactly once")
    expected_pairs = {
        *(("table", item_id) for item_id in spec.included_tables),
        *(("figure", item_id) for item_id in spec.included_figures),
    }
    if set(marker_pairs) != expected_pairs:
        raise ValueError("manuscript evidence markers differ from the locked recipe")
    inventory_items = {item.item_id: item for item in inventory.items}
    missing = set(marker_ids) - set(inventory_items)
    if missing:
        raise ValueError(f"manuscript references unknown display items: {sorted(missing)}")
    for item_kind, item_id in marker_pairs:
        if inventory_items[item_id].item_kind != item_kind:
            raise ValueError(f"manuscript marker kind disagrees with inventory: {item_id}")
    if manuscript_source.count("{{references}}") != 1:
        raise ValueError("manuscript requires exactly one references marker")


def _nested_value(payload: object, field: str) -> object:
    current = payload
    for segment in field.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise ValueError(f"claim field does not exist: {field}")
        current = current[segment]
    return current


def _resolved_claim_value(claim: ManuscriptClaim) -> object:
    if claim.source_kind == "csv":
        with claim.source_path.open(encoding="utf-8", newline="") as source_file:
            rows = list(csv.DictReader(source_file))
        matches = [
            row
            for row in rows
            if all(row.get(column) == value for column, value in claim.row_match.items())
        ]
        if len(matches) != 1:
            raise ValueError(f"claim {claim.claim_id} expected one CSV row, found {len(matches)}")
        if claim.field not in matches[0]:
            raise ValueError(f"claim field does not exist: {claim.field}")
        value: object = matches[0][claim.field]
    else:
        content = claim.source_path.read_text(encoding="utf-8")
        payload = json.loads(content) if claim.source_kind == "json" else yaml.safe_load(content)
        value = _nested_value(payload, claim.field)
    if claim.transform == "length":
        if not isinstance(value, list | tuple | dict | str):
            raise ValueError(f"claim {claim.claim_id} cannot take length of its source value")
        value = len(value)
    return value


def audit_claims(
    manuscript_source: str,
    claims: tuple[ManuscriptClaim, ...],
) -> ManuscriptClaimAudit:
    """Verify source values and the exact human-readable phrases used in prose."""

    entries: list[ClaimAuditEntry] = []
    for claim in claims:
        resolved = _resolved_claim_value(claim)
        expected: object = (
            list(claim.expected) if isinstance(claim.expected, tuple) else claim.expected
        )
        normalized = list(resolved) if isinstance(resolved, tuple) else resolved
        if normalized != expected:
            raise ValueError(f"claim {claim.claim_id} expected {expected!r}, found {normalized!r}")
        if claim.required_text not in manuscript_source:
            raise ValueError(f"claim phrase is absent from manuscript: {claim.claim_id}")
        audit_value: str | int | float | bool | tuple[str, ...]
        if isinstance(resolved, list):
            audit_value = tuple(str(item) for item in resolved)
        elif isinstance(resolved, str | int | float | bool):
            audit_value = resolved
        else:
            raise ValueError(f"claim {claim.claim_id} resolved to an unsupported value")
        entries.append(
            ClaimAuditEntry(
                claim_id=claim.claim_id,
                source_path=str(claim.source_path),
                field=claim.field,
                resolved_value=audit_value,
                required_text=claim.required_text,
                evidence_role=claim.evidence_role,
            )
        )
    return ManuscriptClaimAudit(claim_count=len(entries), entries=tuple(entries))


def verify_manuscript_inputs(
    spec: ManuscriptSpec,
) -> tuple[
    str,
    tuple[ManuscriptReference, ...],
    ManuscriptClaimAudit,
    PublicationDisplayInventory,
    RunManifest,
]:
    """Verify all manuscript inputs without rendering a document."""

    protocol = load_publication_protocol(spec.publication_protocol)
    if protocol.digest() != spec.expected_protocol_sha256:
        raise ValueError("publication protocol differs from the manuscript pin")
    inventory, display_manifest = read_publication_display(
        spec.display.path,
        require_publication_ready=spec.display.require_publication_ready,
    )
    if display_manifest.digest() != spec.display.expected_manifest_sha256:
        raise ValueError("publication display manifest differs from the manuscript pin")
    if inventory.config_sha256 != spec.display.expected_config_sha256:
        raise ValueError("publication display config differs from the manuscript pin")
    manuscript_source = spec.source_markdown.read_text(encoding="utf-8")
    validate_manuscript_markers(manuscript_source, spec, inventory)
    ordered_references = citation_order(manuscript_source, load_references(spec.references))
    claim_audit = audit_claims(manuscript_source, load_claim_ledger(spec.claim_ledger))
    return (
        manuscript_source,
        ordered_references,
        claim_audit,
        inventory,
        display_manifest,
    )


def write_manuscript_bundle(
    spec: ManuscriptSpec,
    *,
    rendered_markdown: str,
    document_bytes: bytes,
    additional_files: dict[str, bytes] | None = None,
    claim_audit: ManuscriptClaimAudit,
    ordered_references: tuple[ManuscriptReference, ...],
    display_inventory: PublicationDisplayInventory,
    display_manifest: RunManifest,
    output_dir: str | Path,
    git_sha: str,
    source_tree_sha256: str | None,
    overwrite: bool = False,
) -> tuple[ManuscriptInventory, RunManifest]:
    """Write the assembled manuscript and an auditable provenance bundle."""

    if not document_bytes.startswith(b"PK"):
        raise ValueError("assembled manuscript is not a DOCX archive")
    extra_payloads = additional_files or {}
    for name in extra_payloads:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manuscript artifact path must be relative: {name}")
    latex_pdf_name = Path(spec.output_filename).with_suffix(".pdf").name
    latex_pdf = extra_payloads.get(latex_pdf_name)
    latex_compiled = latex_pdf is not None and latex_pdf.startswith(b"%PDF")
    if latex_pdf is not None and not latex_compiled:
        raise ValueError("assembled LaTeX PDF is not a PDF document")
    protocol = load_publication_protocol(spec.publication_protocol)
    pending_gates = tuple(
        gate.gate_id for gate in protocol.submission_gates if gate.status == "pending"
    )
    source_tree_clean = source_tree_sha256 is None
    assembly_ready = display_inventory.publication_ready and source_tree_clean
    inventory = ManuscriptInventory(
        manuscript_id=spec.manuscript_id,
        manuscript_revision=spec.manuscript_revision,
        assembled_at=spec.assembled_at,
        title=spec.title,
        author=spec.author,
        config_sha256=spec.digest(),
        protocol_sha256=spec.expected_protocol_sha256,
        display_manifest_sha256=display_manifest.digest(),
        display_publication_ready=display_inventory.publication_ready,
        source_tree_clean=source_tree_clean,
        assembly_ready=assembly_ready,
        submission_ready=assembly_ready and not pending_gates,
        pending_submission_gates=pending_gates,
        included_tables=spec.included_tables,
        included_figures=spec.included_figures,
        citation_count=len(ordered_references),
        quantitative_claim_count=claim_audit.claim_count,
        document_formats=tuple(
            format_name
            for format_name, present in (
                ("docx", True),
                ("markdown", True),
                ("latex", any(name.endswith(".tex") for name in extra_payloads)),
                ("bibtex", any(name.endswith(".bib") for name in extra_payloads)),
                ("pdf", latex_compiled),
            )
            if present
        ),
        latex_compiled=latex_compiled,
        limitations=(
            "This is an offline computational manuscript, not a live NeuroSelect user study.",
            "Original-task EEG, synthetic language, counterfactual replay, and exploratory "
            "candidate-generation evidence remain separate.",
            "Assembly readiness records clean, verified files; it does not satisfy external "
            "ethics, authorship, funding, open-access, or domain-review gates.",
        ),
    )
    payloads = {
        spec.output_filename: document_bytes,
        "manuscript.md": rendered_markdown.encode(),
        "claim-audit.json": (claim_audit.canonical_json() + "\n").encode(),
        "inventory.json": (inventory.canonical_json() + "\n").encode(),
        **extra_payloads,
    }
    destination = Path(output_dir)
    existing = [
        str(destination / name)
        for name in (*payloads, "manifest.json")
        if (destination / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite manuscript artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    package_versions, device = capture_runtime_environment()
    manifest = RunManifest(
        run_id=f"manuscript-{_sha256_bytes(inventory.canonical_json().encode())[:20]}",
        run_kind=RunKind.PUBLICATION_ANALYSIS,
        status=RunStatus.COMPLETED,
        started_at=spec.assembled_at,
        completed_at=spec.assembled_at,
        git_sha=git_sha,
        config_sha256=spec.digest(),
        random_seeds={"document_layout": 0},
        package_versions=package_versions,
        device=device,
        datasets=(
            ArtifactRef(
                artifact_id="paper-display",
                uri="artifact://source-manifest/paper-display",
                sha256=display_manifest.digest(),
                revision=display_inventory.display_revision,
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id="manuscript-" + re.sub(r"[^a-z0-9]+", "-", name).strip("-"),
                uri=f"artifact://{name}",
                sha256=_sha256_bytes(content),
                revision=spec.manuscript_revision,
            )
            for name, content in sorted(payloads.items())
        ),
        metadata={
            "assembly_ready": inventory.assembly_ready,
            "submission_ready": inventory.submission_ready,
            "pending_submission_gates": list(inventory.pending_submission_gates),
            "citation_count": inventory.citation_count,
            "quantitative_claim_count": inventory.quantitative_claim_count,
            "document_formats": list(inventory.document_formats),
            "latex_compiled": inventory.latex_compiled,
            "working_tree_dirty": not source_tree_clean,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    (destination / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    return inventory, manifest


def read_manuscript_bundle(
    directory: str | Path,
    *,
    require_assembly_ready: bool = False,
) -> tuple[ManuscriptInventory, ManuscriptClaimAudit, RunManifest]:
    """Verify every assembled manuscript checksum."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json((source / "manifest.json").read_text())
    for output in manifest.outputs:
        relative = output.uri.removeprefix("artifact://")
        if _sha256_bytes((source / relative).read_bytes()) != output.sha256:
            raise ValueError(f"manuscript SHA-256 mismatch: {relative}")
    inventory = ManuscriptInventory.model_validate_json((source / "inventory.json").read_text())
    claim_audit = ManuscriptClaimAudit.model_validate_json(
        (source / "claim-audit.json").read_text()
    )
    if (
        manifest.run_kind is not RunKind.PUBLICATION_ANALYSIS
        or manifest.config_sha256 != inventory.config_sha256
        or manifest.metadata.get("assembly_ready") != inventory.assembly_ready
        or manifest.metadata.get("submission_ready") != inventory.submission_ready
        or manifest.metadata.get("quantitative_claim_count") != claim_audit.claim_count
        or manifest.metadata.get("document_formats") != list(inventory.document_formats)
        or manifest.metadata.get("latex_compiled") != inventory.latex_compiled
    ):
        raise ValueError("manuscript manifest disagrees with its inventory")
    if require_assembly_ready and not inventory.assembly_ready:
        raise ValueError("manuscript was assembled from an uncommitted source tree")
    return inventory, claim_audit, manifest
