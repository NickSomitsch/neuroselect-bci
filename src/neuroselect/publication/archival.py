"""Fail-closed, deterministic archival release assembly."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.provenance import RunManifest, RunStatus

DEFAULT_RELEASE_CONFIG = Path("configs/publication/release_v1.yaml")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOI = re.compile(r"^10\.5281/zenodo\.[0-9]+$")
_ALLOWED_OUTPUT_SUFFIXES = {
    ".bib",
    ".csv",
    ".docx",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
_DENIED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".joblib",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_DENIED_PARTS = {
    "checkpoints",
    "data",
    "huggingface-cache",
    "models",
    "raw",
}
_DENIED_SOURCE_PREFIXES = ("artifacts/", "data/", "models/")


def sha256_bytes(content: bytes) -> str:
    """Return the lower-case SHA-256 digest of *content*."""

    return hashlib.sha256(content).hexdigest()


def canonical_json(payload: object) -> bytes:
    """Serialize an object as deterministic UTF-8 JSON with a trailing newline."""

    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


class ReleaseArtifactSource(BaseModel):
    """One frozen manifest whose public-safe outputs may enter the archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9-]+$")
    path: Path
    expected_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    purpose: str = Field(min_length=1, max_length=240)
    license: Literal["CC-BY-4.0"] = "CC-BY-4.0"


class ExcludedArtifactClass(BaseModel):
    """A prohibited artifact category documented in the release inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_class: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=300)


class ReleaseSpec(BaseModel):
    """Typed recipe for the stable software and research-output archives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    release_id: Literal["neuroselect-v0.1.0"]
    version: Literal["0.1.0"]
    tag: Literal["v0.1.0"]
    repository_url: str = Field(pattern=r"^https://github\.com/")
    zenodo_doi: str | None = None
    source_license: Literal["MIT"] = "MIT"
    publication_license: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    artifact_sources: tuple[ReleaseArtifactSource, ...] = Field(min_length=1)
    excluded_artifact_classes: tuple[ExcludedArtifactClass, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_release(self) -> ReleaseSpec:
        ids = [source.source_id for source in self.artifact_sources]
        if len(ids) != len(set(ids)):
            raise ValueError("release artifact source IDs must be unique")
        if self.zenodo_doi is not None and not _DOI.fullmatch(self.zenodo_doi):
            raise ValueError("Zenodo DOI must use the form 10.5281/zenodo.<record>")
        return self


class ReleaseFile(BaseModel):
    """One byte-exact archive member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    purpose: str
    source_manifest: str | None = None
    license: str


class ReleaseExclusion(BaseModel):
    """One excluded manifest output or declared artifact class."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_class: str
    path: str | None = None
    reason: str
    source_manifest: str | None = None


class ArchiveInventory(BaseModel):
    """Inventory for one deterministic archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1)
    files: tuple[ReleaseFile, ...]


class PublicationReleaseInventory(BaseModel):
    """Top-level, machine-readable release provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    release_id: str
    version: str
    tag: str
    git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_url: str
    zenodo_doi: str | None
    release_ready: bool
    pending_gates: tuple[str, ...]
    archives: tuple[ArchiveInventory, ArchiveInventory]
    exclusions: tuple[ReleaseExclusion, ...]


class PublicationReleaseResult(BaseModel):
    """Paths and validated inventory produced by the release builder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: Path
    inventory: PublicationReleaseInventory


def load_release_spec(path: str | Path = DEFAULT_RELEASE_CONFIG) -> ReleaseSpec:
    """Load and validate the archival release recipe."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release config must contain a YAML mapping")
    return ReleaseSpec.model_validate(payload)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {value}")
    return path


def _git(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_bytes(*arguments: str, cwd: Path) -> bytes:
    return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True).stdout


def _tracked_source_files(repository: Path, revision: str) -> dict[str, bytes]:
    paths = _git("ls-tree", "-r", "--name-only", revision, cwd=repository).splitlines()
    return {
        path: _git_bytes("show", f"{revision}:{path}", cwd=repository) for path in paths if path
    }


def _tar_gz(files: dict[str, bytes]) -> bytes:
    """Create a deterministic gzip-compressed POSIX tar archive."""

    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path, content in sorted(files.items()):
            _safe_relative_path(path)
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _license_for_source(path: str, spec: ReleaseSpec) -> str:
    return spec.publication_license if path.startswith("paper/") else spec.source_license


def _source_archive(
    spec: ReleaseSpec, repository: Path, revision: str
) -> tuple[bytes, tuple[ReleaseFile, ...]]:
    root = f"neuroselect-bci-{spec.version}"
    files = _tracked_source_files(repository, revision)
    denied = [
        path
        for path in files
        if path.startswith(_DENIED_SOURCE_PREFIXES)
        or PurePosixPath(path).suffix.lower() in _DENIED_SUFFIXES
    ]
    if denied:
        raise ValueError(f"restricted files are tracked at the release commit: {denied}")
    inventory = tuple(
        ReleaseFile(
            path=f"{root}/{path}",
            sha256=sha256_bytes(content),
            size=len(content),
            purpose="Exact tagged repository source",
            license=_license_for_source(path, spec),
        )
        for path, content in sorted(files.items())
    )
    archive = _tar_gz({f"{root}/{path}": content for path, content in files.items()})
    return archive, inventory


def _manifest_output_path(uri: str) -> PurePosixPath:
    prefix = "artifact://"
    if not uri.startswith(prefix):
        raise ValueError(f"release output URI is not local: {uri}")
    return _safe_relative_path(uri.removeprefix(prefix))


def _research_archive(
    spec: ReleaseSpec, repository: Path
) -> tuple[bytes, tuple[ReleaseFile, ...], tuple[ReleaseExclusion, ...]]:
    root = f"neuroselect-research-outputs-{spec.version}"
    archive_files: dict[str, bytes] = {}
    included: list[ReleaseFile] = []
    excluded = [
        ReleaseExclusion(artifact_class=item.artifact_class, reason=item.reason)
        for item in spec.excluded_artifact_classes
    ]
    for source in spec.artifact_sources:
        source_path = repository / source.path
        manifest_path = source_path / "manifest.json"
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.status is not RunStatus.COMPLETED:
            raise ValueError(f"release source is not completed: {source.source_id}")
        if manifest.digest() != source.expected_manifest_sha256:
            raise ValueError(f"release source manifest changed: {source.source_id}")
        manifest_bytes = canonical_json(manifest.model_dump(mode="json", exclude_none=True))
        manifest_target = f"{root}/{source.source_id}/manifest.json"
        archive_files[manifest_target] = manifest_bytes
        included.append(
            ReleaseFile(
                path=manifest_target,
                sha256=sha256_bytes(manifest_bytes),
                size=len(manifest_bytes),
                purpose=f"Source manifest: {source.purpose}",
                source_manifest=source.expected_manifest_sha256,
                license=source.license,
            )
        )
        for output in manifest.outputs:
            relative = _manifest_output_path(output.uri)
            file_path = source_path.joinpath(*relative.parts)
            suffix = file_path.suffix.lower()
            denied_part = next((part for part in relative.parts if part in _DENIED_PARTS), None)
            if suffix in _DENIED_SUFFIXES or denied_part is not None:
                excluded.append(
                    ReleaseExclusion(
                        artifact_class="executable-or-restricted-artifact",
                        path=str(source.path / Path(*relative.parts)),
                        reason=(
                            "Model, checkpoint, raw-data, or executable payloads are not "
                            "redistributed."
                        ),
                        source_manifest=source.expected_manifest_sha256,
                    )
                )
                continue
            if suffix not in _ALLOWED_OUTPUT_SUFFIXES:
                raise ValueError(f"unclassified release output type: {file_path}")
            content = file_path.read_bytes()
            if sha256_bytes(content) != output.sha256:
                raise ValueError(f"release output checksum changed: {file_path}")
            target = f"{root}/{source.source_id}/{relative.as_posix()}"
            archive_files[target] = content
            included.append(
                ReleaseFile(
                    path=target,
                    sha256=output.sha256,
                    size=len(content),
                    purpose=f"{source.purpose}: {output.artifact_id}",
                    source_manifest=source.expected_manifest_sha256,
                    license=source.license,
                )
            )
    return _tar_gz(archive_files), tuple(included), tuple(excluded)


def publication_release_gates(
    spec: ReleaseSpec, repository: Path, revision: str | None = None
) -> list[str]:
    """Return every condition that blocks an exact archival release."""

    revision = revision or _git("rev-parse", "HEAD", cwd=repository)
    expected = spec.version
    checks = {
        "pyproject.toml": f'version = "{expected}"',
        "ui/package.json": f'"version": "{expected}"',
        "CITATION.cff": f"version: {expected}",
        "uv.lock": f'version = "{expected}"',
    }
    pending = [
        f"version metadata is not {expected} in {path}"
        for path, marker in checks.items()
        if marker not in (repository / path).read_text(encoding="utf-8")
    ]
    citation = (repository / "CITATION.cff").read_text(encoding="utf-8")
    if spec.zenodo_doi is None:
        pending.append("Zenodo DOI has not been reserved")
    elif spec.zenodo_doi not in citation:
        pending.append("reserved Zenodo DOI is not embedded in CITATION.cff")
    tags = _git("tag", "--points-at", revision, cwd=repository).splitlines()
    if spec.tag not in tags:
        pending.append(f"tag {spec.tag} does not point to the release commit")
    if _git("status", "--porcelain", cwd=repository):
        pending.append("working tree is not clean")
    return pending


def build_publication_release(
    spec: ReleaseSpec,
    *,
    repository: Path,
    output: Path,
    allow_pending: bool = False,
    overwrite: bool = False,
) -> PublicationReleaseResult:
    """Build deterministic source/evidence archives and their verification metadata."""

    repository = repository.resolve()
    revision = _git("rev-parse", "HEAD", cwd=repository)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("release requires a full Git commit SHA")
    pending = publication_release_gates(spec, repository, revision)
    if pending and not allow_pending:
        raise ValueError("publication release is blocked: " + "; ".join(pending))
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"release output already exists: {output}")
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_bytes, source_files = _source_archive(spec, repository, revision)
    evidence_bytes, evidence_files, exclusions = _research_archive(spec, repository)
    source_name = f"neuroselect-bci-{spec.version}-source.tar.gz"
    evidence_name = f"neuroselect-bci-{spec.version}-research-outputs.tar.gz"
    (output / source_name).write_bytes(source_bytes)
    (output / evidence_name).write_bytes(evidence_bytes)
    archives = (
        ArchiveInventory(
            filename=source_name,
            sha256=sha256_bytes(source_bytes),
            size=len(source_bytes),
            files=source_files,
        ),
        ArchiveInventory(
            filename=evidence_name,
            sha256=sha256_bytes(evidence_bytes),
            size=len(evidence_bytes),
            files=evidence_files,
        ),
    )
    inventory = PublicationReleaseInventory(
        release_id=spec.release_id,
        version=spec.version,
        tag=spec.tag,
        git_revision=revision,
        repository_url=spec.repository_url,
        zenodo_doi=spec.zenodo_doi,
        release_ready=not pending,
        pending_gates=tuple(pending),
        archives=archives,
        exclusions=exclusions,
    )
    (output / "release-inventory.json").write_bytes(
        canonical_json(inventory.model_dump(mode="json"))
    )
    sums = "".join(f"{item.sha256}  {item.filename}\n" for item in archives)
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    doi = spec.zenodo_doi or "PENDING—reserve a version-specific Zenodo DOI"
    readiness = "READY" if not pending else "DEVELOPMENT PREVIEW—NOT FOR PUBLICATION"
    notes = (
        f"# NeuroSelect {spec.tag}\n\nStatus: {readiness}\n\n"
        f"Source commit: `{revision}`\n\nZenodo DOI: `{doi}`\n\n"
        "This release preserves all weak, null, and unfavorable findings. The source is "
        "MIT-licensed; "
        "original publication displays and distributable research summaries are CC BY 4.0. "
        "Study P data, Qwen weights, adapters, and checkpoints are not redistributed.\n"
    )
    (output / "RELEASE-NOTES.md").write_text(notes, encoding="utf-8")
    verification = {
        "schema_version": "1.0",
        "status": "passed" if not pending else "blocked-pending-external-gates",
        "git_revision": revision,
        "archives_recomputed": True,
        "manifest_outputs_verified": True,
        "denylist_enforced": True,
        "pending_gates": pending,
    }
    (output / "verification-report.json").write_bytes(canonical_json(verification))
    verify_publication_release(output, require_ready=False)
    return PublicationReleaseResult(output=output, inventory=inventory)


def verify_publication_release(
    output: Path, *, require_ready: bool = True
) -> PublicationReleaseInventory:
    """Independently verify inventory schema, archive members, checksums, and denylist."""

    inventory = PublicationReleaseInventory.model_validate_json(
        (output / "release-inventory.json").read_text(encoding="utf-8")
    )
    if require_ready and not inventory.release_ready:
        raise ValueError("publication release is not ready: " + "; ".join(inventory.pending_gates))
    expected_sums: list[str] = []
    for archive in inventory.archives:
        content = (output / archive.filename).read_bytes()
        if len(content) != archive.size or sha256_bytes(content) != archive.sha256:
            raise ValueError(f"release archive checksum mismatch: {archive.filename}")
        expected_sums.append(f"{archive.sha256}  {archive.filename}\n")
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            members = {member.name: member for member in tar.getmembers() if member.isfile()}
            if set(members) != {item.path for item in archive.files}:
                raise ValueError(f"release archive inventory mismatch: {archive.filename}")
            for item in archive.files:
                member = members[item.path]
                extracted = tar.extractfile(member)
                data = b"" if extracted is None else extracted.read()
                if len(data) != item.size or sha256_bytes(data) != item.sha256:
                    raise ValueError(f"release member checksum mismatch: {item.path}")
                path = PurePosixPath(item.path)
                if path.suffix.lower() in _DENIED_SUFFIXES:
                    raise ValueError(f"denied artifact entered release: {item.path}")
    if (output / "SHA256SUMS").read_text(encoding="utf-8") != "".join(expected_sums):
        raise ValueError("SHA256SUMS does not match the release inventory")
    return inventory
