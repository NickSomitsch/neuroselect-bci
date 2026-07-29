"""Checksum-pinned tables and figures for the offline NeuroSelect manuscript."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation import capture_runtime_environment
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus

DEFAULT_PUBLICATION_DISPLAY_CONFIG = Path("configs/publication/paper_display_v1.yaml")
SourceId = Literal[
    "primary-analysis",
    "candidate-generation-v2",
    "candidate-generation-step4",
    "opening-generalization",
]
EvidenceRole = Literal[
    "evidence-map",
    "primary",
    "primary-with-secondary-comparator",
    "exploratory",
]
FigureFormat = Literal["svg", "png", "pdf"]


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class PublicationDisplaySource(BaseModel):
    """One immutable evidence source used only for display transformation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    path: Path
    expected_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_role: Literal[
        "primary_synthesis",
        "exploratory_test_exposed",
        "exploratory_locked",
    ]


class PublicationDisplaySpec(BaseModel):
    """Locked, outcome-independent visual and tabular display recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    display_id: str = Field(min_length=1, max_length=160)
    display_revision: Literal["paper-display-v1"]
    generated_at: datetime
    publication_protocol: Path
    expected_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: tuple[PublicationDisplaySource, ...]
    figure_formats: tuple[FigureFormat, ...] = ("svg", "png", "pdf")
    raster_dpi: Literal[300] = 300
    style_revision: Literal["neuroselect-publication-style-v1"]
    evidence_separation_policy: Literal[
        "primary_secondary_and_exploratory_results_are_never_pooled"
    ]
    outcome_based_omission_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_display_recipe(self) -> PublicationDisplaySpec:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("publication display time must include a timezone")
        expected_sources = (
            "primary-analysis",
            "candidate-generation-v2",
            "candidate-generation-step4",
            "opening-generalization",
        )
        if tuple(source.source_id for source in self.sources) != expected_sources:
            raise ValueError("publication display sources and their order are locked")
        if self.figure_formats != ("svg", "png", "pdf"):
            raise ValueError("publication figures require SVG, PNG, and PDF outputs")
        return self

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.model_dump(mode="json")).encode())


class PublicationTable(BaseModel):
    """A table represented once and emitted as machine-readable CSV and Markdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(pattern=r"^table-[0-9]+[a-z]?-[a-z0-9-]+$")
    title: str = Field(min_length=1, max_length=240)
    evidence_roles: tuple[EvidenceRole, ...] = Field(min_length=1)
    source_ids: tuple[SourceId, ...] = Field(min_length=1)
    columns: tuple[str, ...] = Field(min_length=2)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)
    caption: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rows(self) -> PublicationTable:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("publication table columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("every publication table row must match its columns")
        return self

    def csv_content(self) -> str:
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return output.getvalue()

    def markdown_content(self) -> str:
        def escaped(value: str) -> str:
            return value.replace("|", r"\|").replace("\n", "<br>")

        header = "| " + " | ".join(escaped(column) for column in self.columns) + " |"
        separator = "| " + " | ".join("---" for _ in self.columns) + " |"
        rows = ["| " + " | ".join(escaped(value) for value in row) + " |" for row in self.rows]
        return "\n".join((f"# {self.title}", "", self.caption, "", header, separator, *rows)) + "\n"


@dataclass(frozen=True)
class RenderedPublicationFigure:
    """In-memory figure files and the metadata shared across all formats."""

    item_id: str
    title: str
    evidence_roles: tuple[EvidenceRole, ...]
    source_ids: tuple[SourceId, ...]
    caption: str
    files: dict[FigureFormat, bytes]


class PublicationDisplayItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    item_kind: Literal["table", "figure"]
    title: str
    evidence_roles: tuple[EvidenceRole, ...]
    source_ids: tuple[SourceId, ...]
    files: tuple[str, ...] = Field(min_length=1)
    caption: str


class PublicationDisplayInventory(BaseModel):
    """Auditable inventory consumed by manuscript drafting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    display_id: str
    display_revision: Literal["paper-display-v1"]
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: dict[str, str]
    source_tree_clean: bool
    publication_ready: bool
    evidence_separation_policy: str
    items: tuple[PublicationDisplayItem, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_publication_display_spec(
    path: str | Path = DEFAULT_PUBLICATION_DISPLAY_CONFIG,
) -> PublicationDisplaySpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: object = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("publication display config must contain a YAML mapping")
    return PublicationDisplaySpec.model_validate(payload)


def _captions_content(items: tuple[PublicationDisplayItem, ...]) -> str:
    lines = ["# Publication table and figure captions", ""]
    for item in items:
        roles = ", ".join(item.evidence_roles)
        lines.extend(
            (
                f"## {item.item_id}: {item.title}",
                "",
                item.caption,
                "",
                f"Evidence role: {roles}.",
                "",
            )
        )
    return "\n".join(lines)


def write_publication_display(
    spec: PublicationDisplaySpec,
    tables: tuple[PublicationTable, ...],
    figures: tuple[RenderedPublicationFigure, ...],
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None,
    overwrite: bool = False,
) -> tuple[PublicationDisplayInventory, RunManifest]:
    """Write all display files and a checksum-addressed inventory and manifest."""

    if not tables or not figures:
        raise ValueError("publication display requires at least one table and one figure")
    item_ids = [table.item_id for table in tables] + [figure.item_id for figure in figures]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("publication display item IDs must be unique")
    expected_formats = set(spec.figure_formats)
    if any(set(figure.files) != expected_formats for figure in figures):
        raise ValueError("every publication figure must provide all locked output formats")
    if any(not content for figure in figures for content in figure.files.values()):
        raise ValueError("publication figure files cannot be empty")

    destination = Path(output_dir)
    payloads: dict[str, bytes] = {}
    items: list[PublicationDisplayItem] = []
    for table in tables:
        table_files = (
            f"tables/{table.item_id}.csv",
            f"tables/{table.item_id}.md",
        )
        payloads[table_files[0]] = table.csv_content().encode()
        payloads[table_files[1]] = table.markdown_content().encode()
        items.append(
            PublicationDisplayItem(
                item_id=table.item_id,
                item_kind="table",
                title=table.title,
                evidence_roles=table.evidence_roles,
                source_ids=table.source_ids,
                files=table_files,
                caption=table.caption,
            )
        )
    for figure in figures:
        figure_files = tuple(f"figures/{figure.item_id}.{suffix}" for suffix in spec.figure_formats)
        for suffix, relative_path in zip(spec.figure_formats, figure_files, strict=True):
            payloads[relative_path] = figure.files[suffix]
        items.append(
            PublicationDisplayItem(
                item_id=figure.item_id,
                item_kind="figure",
                title=figure.title,
                evidence_roles=figure.evidence_roles,
                source_ids=figure.source_ids,
                files=figure_files,
                caption=figure.caption,
            )
        )

    source_manifest_sha256: dict[str, str] = {
        source.source_id: source.expected_manifest_sha256 for source in spec.sources
    }
    source_tree_clean = source_tree_sha256 is None
    inventory = PublicationDisplayInventory(
        display_id=spec.display_id,
        display_revision=spec.display_revision,
        generated_at=spec.generated_at,
        config_sha256=spec.digest(),
        protocol_sha256=spec.expected_protocol_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_clean=source_tree_clean,
        publication_ready=source_tree_clean,
        evidence_separation_policy=spec.evidence_separation_policy,
        items=tuple(items),
        limitations=(
            "Figures and tables transform frozen estimates; they do not add observations or "
            "rerun models.",
            "Primary, secondary-comparator, and exploratory results remain explicitly labeled "
            "and are never pooled into a single performance score.",
            "Synthetic language and counterfactual replay are not participant-use evidence.",
            "A clean source tree is required before this display bundle is publication-ready.",
        ),
    )
    payloads["inventory.json"] = (inventory.canonical_json() + "\n").encode()
    payloads["captions.md"] = _captions_content(inventory.items).encode()

    existing = [
        str(destination / path)
        for path in (*payloads, "manifest.json")
        if (destination / path).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite publication display artifacts: {existing}")
    for relative, content in payloads.items():
        output_path = destination / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)

    package_versions, device = capture_runtime_environment()
    manifest = RunManifest(
        run_id=f"paper-display-{_sha256_bytes(inventory.canonical_json().encode())[:20]}",
        run_kind=RunKind.PUBLICATION_ANALYSIS,
        status=RunStatus.COMPLETED,
        started_at=spec.generated_at,
        completed_at=spec.generated_at,
        git_sha=git_sha,
        config_sha256=spec.digest(),
        random_seeds={"display_layout": 0},
        package_versions=package_versions,
        device=device,
        datasets=tuple(
            ArtifactRef(
                artifact_id=source.source_id,
                uri=f"artifact://source-manifest/{source.source_id}",
                sha256=source.expected_manifest_sha256,
                revision=source.evidence_role,
            )
            for source in spec.sources
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=("paper-display-" + relative.replace("/", "-").replace(".", "-")),
                uri=f"artifact://{relative}",
                sha256=_sha256_bytes(content),
                revision=spec.display_revision,
            )
            for relative, content in sorted(payloads.items())
        ),
        metadata={
            "protocol_sha256": spec.expected_protocol_sha256,
            "table_count": len(tables),
            "figure_count": len(figures),
            "publication_ready": inventory.publication_ready,
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


def read_publication_display(
    directory: str | Path,
    *,
    require_publication_ready: bool = False,
) -> tuple[PublicationDisplayInventory, RunManifest]:
    """Verify every table and figure checksum before returning the inventory."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for output in manifest.outputs:
        relative = output.uri.removeprefix("artifact://")
        if _sha256_bytes((source / relative).read_bytes()) != output.sha256:
            raise ValueError(f"publication display SHA-256 mismatch: {relative}")
    inventory = PublicationDisplayInventory.model_validate_json(
        (source / "inventory.json").read_text(encoding="utf-8")
    )
    inventory_files = {path for item in inventory.items for path in item.files}
    manifest_files = {
        output.uri.removeprefix("artifact://")
        for output in manifest.outputs
        if output.uri not in {"artifact://inventory.json", "artifact://captions.md"}
    }
    if inventory_files != manifest_files:
        raise ValueError("publication display inventory and manifest file lists differ")
    manifest_sources = {item.artifact_id: item.sha256 for item in manifest.datasets}
    if (
        manifest.run_kind is not RunKind.PUBLICATION_ANALYSIS
        or manifest.config_sha256 != inventory.config_sha256
        or manifest_sources != inventory.source_manifest_sha256
        or manifest.metadata.get("protocol_sha256") != inventory.protocol_sha256
        or manifest.metadata.get("publication_ready") != inventory.publication_ready
    ):
        raise ValueError("publication display manifest disagrees with its inventory")
    if require_publication_ready and not inventory.publication_ready:
        raise ValueError("publication display was generated from an uncommitted source tree")
    return inventory, manifest
