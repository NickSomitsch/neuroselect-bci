"""Verified portable inputs for cloud held-out language evaluation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.language import (
    LocalModelConfig,
    PersonalizationAdapterManifest,
    PersonalizationCorpusManifest,
    load_personalization_adapter,
    load_personalization_corpus_manifest,
)

CLOUD_BUNDLE_MANIFEST = "neuroselect-language-cloud-bundle.json"
CLOUD_BUNDLE_REVISION: Literal["step11-language-inputs-v1"] = "step11-language-inputs-v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


class LanguageCloudBundleFile(BaseModel):
    """One byte-exact member of the portable input bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=500)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_safe_relative_path(self) -> LanguageCloudBundleFile:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path:
            raise ValueError("cloud bundle paths must be normalized and relative")
        return self


class LanguageCloudBundleManifest(BaseModel):
    """Model and profile provenance for a Step 11 input bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    bundle_revision: Literal["step11-language-inputs-v1"]
    model_id: str = Field(min_length=1, max_length=500)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_suffix: str = Field(pattern=r"^-[a-z0-9-]+$")
    profile_ids: tuple[str, ...] = Field(min_length=1)
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[LanguageCloudBundleFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_sorted_members(self) -> LanguageCloudBundleManifest:
        if self.profile_ids != tuple(sorted(set(self.profile_ids))):
            raise ValueError("cloud bundle profiles must be unique and sorted")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("cloud bundle files must be unique and sorted")
        return self

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.model_dump(mode="json")).encode())


def _input_paths(
    *,
    profile_ids: tuple[str, ...],
    adapter_root: Path,
    adapter_suffix: str,
    corpus_root: Path,
    model_config: LocalModelConfig,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    benchmark_sha256: str | None = None
    for profile_id in profile_ids:
        adapter_dir = adapter_root / f"{profile_id}{adapter_suffix}"
        adapter = load_personalization_adapter(
            adapter_dir,
            expected_profile_id=profile_id,
            expected_model_id=model_config.model_id,
            expected_model_revision=model_config.model_revision,
        )
        corpus_dir = corpus_root / profile_id
        corpus = load_personalization_corpus_manifest(corpus_dir)
        if adapter.manifest.trainer_revision != "neuroselect-mlx-lora-v1":
            raise ValueError(f"{profile_id} does not contain a research-trained adapter")
        if not adapter.manifest.validation_evaluated or not adapter.manifest.test_evaluated:
            raise ValueError(f"{profile_id} adapter lacks validation/test evaluation")
        if adapter.manifest.source_corpus_manifest_sha256 != corpus.digest():
            raise ValueError(f"{profile_id} adapter and corpus manifests do not agree")
        if benchmark_sha256 is None:
            benchmark_sha256 = corpus.source_benchmark_sha256
        elif benchmark_sha256 != corpus.source_benchmark_sha256:
            raise ValueError("all cloud corpora must reference one benchmark")
        paths.extend(
            (
                adapter_dir / "adapter_config.json",
                adapter_dir / adapter.manifest.adapter_file,
                adapter_dir / "manifest.json",
                corpus_dir / "manifest.json",
                *(corpus_dir / artifact.path for artifact in corpus.artifacts),
            )
        )
    return tuple(paths)


def create_language_cloud_bundle(
    output_path: str | Path,
    *,
    repository_root: str | Path,
    profile_ids: tuple[str, ...],
    adapter_root: str | Path,
    adapter_suffix: str,
    corpus_root: str | Path,
    model_config: LocalModelConfig,
    overwrite: bool = False,
) -> LanguageCloudBundleManifest:
    """Create a deterministic tar.gz containing only final adapters and their corpora."""

    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite cloud bundle: {destination}")
    root = Path(repository_root).resolve()
    profiles = tuple(sorted(set(profile_ids)))
    paths = _input_paths(
        profile_ids=profiles,
        adapter_root=Path(adapter_root),
        adapter_suffix=adapter_suffix,
        corpus_root=Path(corpus_root),
        model_config=model_config,
    )
    contents: dict[str, bytes] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"cloud input is outside the repository: {path}")
        relative = resolved.relative_to(root).as_posix()
        if relative in contents:
            raise ValueError(f"duplicate cloud bundle input: {relative}")
        contents[relative] = resolved.read_bytes()

    first_corpus = load_personalization_corpus_manifest(Path(corpus_root) / profiles[0])
    manifest = LanguageCloudBundleManifest(
        schema_version="1.0",
        bundle_revision=CLOUD_BUNDLE_REVISION,
        model_id=model_config.model_id,
        model_revision=model_config.model_revision,
        adapter_suffix=adapter_suffix,
        profile_ids=profiles,
        benchmark_source_sha256=first_corpus.source_benchmark_sha256,
        files=tuple(
            LanguageCloudBundleFile(
                path=relative,
                size_bytes=len(content),
                sha256=_sha256_bytes(content),
            )
            for relative, content in sorted(contents.items())
        ),
    )
    manifest_content = (_canonical_json(manifest.model_dump(mode="json")) + "\n").encode()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=temporary, mtime=0) as compressed,
            tarfile.open(mode="w", fileobj=compressed) as archive,
        ):
            for relative, content in (
                (CLOUD_BUNDLE_MANIFEST, manifest_content),
                *sorted(contents.items()),
            ):
                info = tarfile.TarInfo(relative)
                info.size = len(content)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(content))
        temporary.flush()
        os.fsync(temporary.fileno())
    temporary_path.replace(destination)
    return manifest


def _read_language_cloud_bundle(
    bundle_path: str | Path,
) -> tuple[LanguageCloudBundleManifest, dict[str, bytes]]:
    source = Path(bundle_path)
    members: dict[str, bytes] = {}
    with tarfile.open(source, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or member_path.is_absolute()
                or ".." in member_path.parts
                or str(member_path) != member.name
                or member.name in members
            ):
                raise ValueError(f"unsafe or duplicate cloud bundle member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read cloud bundle member: {member.name}")
            members[member.name] = extracted.read()
    try:
        manifest_content = members.pop(CLOUD_BUNDLE_MANIFEST)
    except KeyError as error:
        raise ValueError("cloud bundle manifest is missing") from error
    manifest = LanguageCloudBundleManifest.model_validate_json(manifest_content)
    expected = {item.path: item for item in manifest.files}
    if set(members) != set(expected):
        raise ValueError("cloud bundle members do not match its manifest")
    for path, content in members.items():
        item = expected[path]
        if len(content) != item.size_bytes or _sha256_bytes(content) != item.sha256:
            raise ValueError(f"cloud bundle checksum mismatch: {path}")
    _validate_bundle_semantics(manifest, members)
    return manifest, members


def _validate_bundle_semantics(
    manifest: LanguageCloudBundleManifest,
    members: dict[str, bytes],
) -> None:
    expected_paths: set[str] = set()
    for profile_id in manifest.profile_ids:
        adapter_prefix = f"artifacts/models/language-lora/{profile_id}{manifest.adapter_suffix}"
        corpus_prefix = f"artifacts/language/personalization-v1/{profile_id}"
        adapter_manifest_path = f"{adapter_prefix}/manifest.json"
        corpus_manifest_path = f"{corpus_prefix}/manifest.json"
        try:
            adapter = PersonalizationAdapterManifest.model_validate_json(
                members[adapter_manifest_path]
            )
            corpus = PersonalizationCorpusManifest.model_validate_json(
                members[corpus_manifest_path]
            )
        except KeyError as error:
            raise ValueError(
                f"cloud bundle is missing profile metadata for {profile_id}"
            ) from error
        if (
            adapter.profile_id != profile_id
            or corpus.profile_id != profile_id
            or adapter.base_model_id != manifest.model_id
            or adapter.base_model_revision != manifest.model_revision
            or adapter.trainer_revision != "neuroselect-mlx-lora-v1"
            or not adapter.validation_evaluated
            or not adapter.test_evaluated
            or adapter.source_corpus_manifest_sha256 != corpus.digest()
            or corpus.source_benchmark_sha256 != manifest.benchmark_source_sha256
        ):
            raise ValueError(f"cloud bundle provenance mismatch for {profile_id}")
        adapter_path = f"{adapter_prefix}/{adapter.adapter_file}"
        if _sha256_bytes(members.get(adapter_path, b"")) != adapter.adapter_sha256:
            raise ValueError(f"cloud adapter checksum mismatch for {profile_id}")
        expected_paths.update(
            {
                f"{adapter_prefix}/adapter_config.json",
                adapter_path,
                adapter_manifest_path,
                corpus_manifest_path,
            }
        )
        for artifact in corpus.artifacts:
            path = f"{corpus_prefix}/{artifact.path}"
            content = members.get(path, b"")
            if (
                _sha256_bytes(content) != artifact.sha256
                or sum(1 for line in content.splitlines() if line) != artifact.example_count
            ):
                raise ValueError(f"cloud corpus checksum/count mismatch: {path}")
            expected_paths.add(path)
    if set(members) != expected_paths:
        raise ValueError("cloud bundle contains unexpected adapter or corpus files")


def verify_language_cloud_bundle(
    bundle_path: str | Path,
) -> LanguageCloudBundleManifest:
    """Verify archive safety, checksums, and cross-file research provenance."""

    manifest, _ = _read_language_cloud_bundle(bundle_path)
    return manifest


def extract_language_cloud_bundle(
    bundle_path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> LanguageCloudBundleManifest:
    """Safely extract a fully verified bundle below one repository root."""

    manifest, members = _read_language_cloud_bundle(bundle_path)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    targets: dict[Path, bytes] = {}
    for relative, content in members.items():
        target = root / PurePosixPath(relative)
        resolved_target = target.resolve(strict=False)
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(f"cloud bundle target escapes its destination: {relative}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite extracted cloud input: {target}")
        targets[target] = content
    for target, content in targets.items():
        _atomic_write(target, content)
    return manifest
