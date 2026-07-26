from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.evaluation.language_cloud import (
    LanguageCloudBundleFile,
    LanguageCloudBundleManifest,
    create_language_cloud_bundle,
    extract_language_cloud_bundle,
    verify_language_cloud_bundle,
)
from neuroselect.language import (
    PersonalizationAdapterManifest,
    PersonalizationCorpusArtifact,
    PersonalizationCorpusManifest,
    load_local_model_config,
    load_personalization_adapter,
    load_personalization_corpus_manifest,
)
from neuroselect.synthetic import BenchmarkSplit

ROOT = Path(__file__).parents[2]
PROFILE_ID = "synthetic-concise"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_research_inputs(repository: Path) -> None:
    model = load_local_model_config(ROOT / "configs/models/qwen3_4b_mlx.yaml")
    corpus_dir = repository / "artifacts/language/personalization-v1" / PROFILE_ID
    corpus_dir.mkdir(parents=True)
    artifacts: list[PersonalizationCorpusArtifact] = []
    for split, filename in (
        (BenchmarkSplit.TRAIN, "train.jsonl"),
        (BenchmarkSplit.VALIDATION, "valid.jsonl"),
        (BenchmarkSplit.TEST, "test.jsonl"),
    ):
        content = b'{"completion":" alpha","prompt":"test"}\n'
        (corpus_dir / filename).write_bytes(content)
        artifacts.append(
            PersonalizationCorpusArtifact(
                split=split,
                path=filename,
                source_message_count=1,
                example_count=1,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    corpus = PersonalizationCorpusManifest(
        schema_version="1.0",
        profile_id=PROFILE_ID,
        source_benchmark_sha256="a" * 64,
        profile_style_sha256="b" * 64,
        prompt_revision="personal-next-span-completion-v1",
        artifacts=tuple(artifacts),
    )
    (corpus_dir / "manifest.json").write_text(
        _canonical_json(corpus.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    adapter_dir = repository / "artifacts/models/language-lora" / f"{PROFILE_ID}-research-v1"
    adapter_dir.mkdir(parents=True)
    weights = b"test adapter weights"
    (adapter_dir / "adapters.safetensors").write_bytes(weights)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    adapter = PersonalizationAdapterManifest(
        schema_version="1.0",
        adapter_id="lora-synthetic-concise-research-v1",
        profile_id=PROFILE_ID,
        base_model_id=model.model_id,
        base_model_revision=model.model_revision,
        adapter_file="adapters.safetensors",
        adapter_sha256=hashlib.sha256(weights).hexdigest(),
        source_corpus_manifest_sha256=corpus.digest(),
        training_config_sha256="c" * 64,
        trainer_revision="neuroselect-mlx-lora-v1",
        mlx_lm_version="0.31.3",
        trained_at=datetime(2026, 7, 26, tzinfo=UTC),
        validation_evaluated=True,
        test_evaluated=True,
    )
    (adapter_dir / "manifest.json").write_text(
        _canonical_json(adapter.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )


def test_cloud_bundle_is_deterministic_verified_and_safely_extracted(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    write_research_inputs(repository)
    model = load_local_model_config(ROOT / "configs/models/qwen3_4b_mlx.yaml")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    def create(path: Path) -> LanguageCloudBundleManifest:
        return create_language_cloud_bundle(
            path,
            repository_root=repository,
            profile_ids=(PROFILE_ID,),
            adapter_root=repository / "artifacts/models/language-lora",
            adapter_suffix="-research-v1",
            corpus_root=repository / "artifacts/language/personalization-v1",
            model_config=model,
        )

    created = create(first)
    create(second)

    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite cloud bundle"):
        create(first)
    assert verify_language_cloud_bundle(first) == created
    assert {item.path for item in created.files} == {
        f"artifacts/models/language-lora/{PROFILE_ID}-research-v1/adapter_config.json",
        f"artifacts/models/language-lora/{PROFILE_ID}-research-v1/adapters.safetensors",
        f"artifacts/models/language-lora/{PROFILE_ID}-research-v1/manifest.json",
        f"artifacts/language/personalization-v1/{PROFILE_ID}/manifest.json",
        f"artifacts/language/personalization-v1/{PROFILE_ID}/train.jsonl",
        f"artifacts/language/personalization-v1/{PROFILE_ID}/valid.jsonl",
        f"artifacts/language/personalization-v1/{PROFILE_ID}/test.jsonl",
    }

    extracted = tmp_path / "extracted"
    assert extract_language_cloud_bundle(first, extracted) == created
    load_personalization_adapter(
        extracted / "artifacts/models/language-lora" / f"{PROFILE_ID}-research-v1",
        expected_profile_id=PROFILE_ID,
        expected_model_id=model.model_id,
        expected_model_revision=model.model_revision,
    )
    load_personalization_corpus_manifest(
        extracted / "artifacts/language/personalization-v1" / PROFILE_ID
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_language_cloud_bundle(first, extracted)
    assert extract_language_cloud_bundle(first, extracted, overwrite=True) == created

    with pytest.raises(ValueError, match="outside the repository"):
        create_language_cloud_bundle(
            tmp_path / "outside.tar.gz",
            repository_root=repository / "artifacts/language",
            profile_ids=(PROFILE_ID,),
            adapter_root=repository / "artifacts/models/language-lora",
            adapter_suffix="-research-v1",
            corpus_root=repository / "artifacts/language/personalization-v1",
            model_config=model,
        )


def test_cloud_bundle_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.tar.gz"
    content = b"escape"
    with tarfile.open(unsafe, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with pytest.raises(ValueError, match="unsafe or duplicate"):
        verify_language_cloud_bundle(unsafe)


def test_cloud_bundle_models_and_archive_checksums_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="normalized and relative"):
        LanguageCloudBundleFile(path="../escape", size_bytes=1, sha256="a" * 64)

    file = LanguageCloudBundleFile(path="safe", size_bytes=1, sha256="a" * 64)
    payload = {
        "schema_version": "1.0",
        "bundle_revision": "step11-language-inputs-v1",
        "model_id": "test/model",
        "model_revision": "b" * 40,
        "adapter_suffix": "-research-v1",
        "profile_ids": [PROFILE_ID, PROFILE_ID],
        "benchmark_source_sha256": "c" * 64,
        "files": [file.model_dump()],
    }
    with pytest.raises(ValidationError, match="profiles must be unique"):
        LanguageCloudBundleManifest.model_validate(payload)
    payload["profile_ids"] = [PROFILE_ID]
    payload["files"] = [file.model_dump(), file.model_dump()]
    with pytest.raises(ValidationError, match="files must be unique"):
        LanguageCloudBundleManifest.model_validate(payload)

    missing_manifest = tmp_path / "missing-manifest.tar.gz"
    content = b"data"
    with tarfile.open(missing_manifest, mode="w:gz") as archive:
        info = tarfile.TarInfo("data")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="manifest is missing"):
        verify_language_cloud_bundle(missing_manifest)


def test_cloud_bundle_detects_member_tampering(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    write_research_inputs(repository)
    model = load_local_model_config(ROOT / "configs/models/qwen3_4b_mlx.yaml")
    valid = tmp_path / "valid.tar.gz"
    create_language_cloud_bundle(
        valid,
        repository_root=repository,
        profile_ids=(PROFILE_ID,),
        adapter_root=repository / "artifacts/models/language-lora",
        adapter_suffix="-research-v1",
        corpus_root=repository / "artifacts/language/personalization-v1",
        model_config=model,
    )
    members: list[tuple[str, bytes]] = []
    with tarfile.open(valid, mode="r:gz") as archive:
        for member in archive.getmembers():
            extracted = archive.extractfile(member)
            assert extracted is not None
            value = extracted.read()
            if member.name.endswith("adapter_config.json"):
                value = b'{"tampered":true}\n'
            members.append((member.name, value))
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, mode="w:gz") as archive:
        for name, value in members:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_language_cloud_bundle(tampered)
