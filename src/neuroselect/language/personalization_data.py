"""Leakage-resistant MLX completion corpora for synthetic style adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.synthetic.models import (
    BenchmarkMessage,
    BenchmarkSplit,
    GeneratedBenchmark,
    SyntheticProfile,
)

PERSONALIZATION_PROMPT_REVISION: Literal["personal-next-span-completion-v1"] = (
    "personal-next-span-completion-v1"
)
SPLIT_FILENAMES = {
    BenchmarkSplit.TRAIN: "train.jsonl",
    BenchmarkSplit.VALIDATION: "valid.jsonl",
    BenchmarkSplit.TEST: "test.jsonl",
}


class PersonalizationCorpusArtifact(BaseModel):
    """One MLX-LM split file with stable counts and checksum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: BenchmarkSplit
    path: str = Field(pattern=r"^(train|valid|test)\.jsonl$")
    source_message_count: int = Field(ge=1)
    example_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PersonalizationCorpusManifest(BaseModel):
    """Per-profile train/validation/test provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    profile_id: str = Field(pattern=r"^synthetic-[a-z0-9-]+$")
    synthetic: Literal[True] = True
    source_benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_style_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_revision: Literal["personal-next-span-completion-v1"]
    artifacts: tuple[PersonalizationCorpusArtifact, ...]

    @model_validator(mode="after")
    def require_all_splits(self) -> PersonalizationCorpusManifest:
        splits = [artifact.split for artifact in self.artifacts]
        if len(splits) != len(set(splits)) or set(splits) != set(BenchmarkSplit):
            raise ValueError("personalization corpus must contain each split exactly once")
        for artifact in self.artifacts:
            if artifact.path != SPLIT_FILENAMES[artifact.split]:
                raise ValueError("personalization split uses the wrong MLX filename")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def personalization_prompt(confirmed_text: str) -> str:
    """Stable completion prompt shared by corpus generation and scoring design."""

    context = confirmed_text or "(empty message)"
    return (
        "Continue the confirmed assistive-communication message with exactly one next word or "
        f"short phrase.\nConfirmed message: {context!r}\nNext phrase:"
    )


def _examples(message: BenchmarkMessage) -> tuple[dict[str, str], ...]:
    examples: list[dict[str, str]] = []
    confirmed: list[str] = []
    for span in message.target_spans:
        examples.append(
            {
                "prompt": personalization_prompt(" ".join(confirmed)),
                "completion": f" {span}",
            }
        )
        confirmed.append(span)
    return tuple(examples)


def _profile_style_sha256(profile: SyntheticProfile) -> str:
    payload = json.dumps(
        profile.style.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_personalization_corpus(
    benchmark: GeneratedBenchmark,
    profile: SyntheticProfile,
    output_dir: str | Path,
) -> PersonalizationCorpusManifest:
    """Write one profile's MLX train/valid/test corpus without crossing splits."""

    if profile.profile_id not in benchmark.profile_ids:
        raise ValueError("profile is not present in the generated benchmark")
    destination = Path(output_dir) / profile.profile_id
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[PersonalizationCorpusArtifact] = []
    message_ids_by_split: dict[BenchmarkSplit, set[str]] = {}

    for split in BenchmarkSplit:
        messages = tuple(
            message
            for message in benchmark.messages[split]
            if message.profile_id == profile.profile_id
        )
        if not messages:
            raise ValueError(f"profile has no {split.value} messages")
        message_ids_by_split[split] = {message.message_id for message in messages}
        examples = tuple(example for message in messages for example in _examples(message))
        content = "".join(
            json.dumps(
                example,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for example in examples
        )
        path = destination / SPLIT_FILENAMES[split]
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PersonalizationCorpusArtifact(
                split=split,
                path=path.name,
                source_message_count=len(messages),
                example_count=len(examples),
                sha256=hashlib.sha256(content.encode()).hexdigest(),
            )
        )

    split_pairs = (
        (BenchmarkSplit.TRAIN, BenchmarkSplit.VALIDATION),
        (BenchmarkSplit.TRAIN, BenchmarkSplit.TEST),
        (BenchmarkSplit.VALIDATION, BenchmarkSplit.TEST),
    )
    if any(
        message_ids_by_split[left].intersection(message_ids_by_split[right])
        for left, right in split_pairs
    ):
        raise ValueError("message IDs must be disjoint across personalization splits")

    manifest = PersonalizationCorpusManifest(
        schema_version="1.0",
        profile_id=profile.profile_id,
        source_benchmark_sha256=benchmark.source_sha256,
        profile_style_sha256=_profile_style_sha256(profile),
        prompt_revision=PERSONALIZATION_PROMPT_REVISION,
        artifacts=tuple(artifacts),
    )
    manifest_content = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    (destination / "manifest.json").write_text(manifest_content + "\n", encoding="utf-8")
    return manifest


def write_all_personalization_corpora(
    benchmark: GeneratedBenchmark,
    profiles: tuple[SyntheticProfile, ...],
    output_dir: str | Path,
) -> tuple[PersonalizationCorpusManifest, ...]:
    """Write stable per-profile corpora in profile-ID order."""

    profile_by_id = {profile.profile_id: profile for profile in profiles}
    if set(profile_by_id) != set(benchmark.profile_ids):
        raise ValueError("profiles must match the generated benchmark exactly")
    return tuple(
        write_personalization_corpus(benchmark, profile_by_id[profile_id], output_dir)
        for profile_id in sorted(profile_by_id)
    )


def load_personalization_corpus_manifest(
    path: str | Path,
) -> PersonalizationCorpusManifest:
    """Load and checksum every split before training."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = PersonalizationCorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    directory = manifest_path.parent
    for artifact in manifest.artifacts:
        artifact_path = directory / artifact.path
        content = artifact_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise ValueError(f"personalization corpus checksum mismatch: {artifact.path}")
        line_count = sum(1 for line in content.splitlines() if line)
        if line_count != artifact.example_count:
            raise ValueError(f"personalization corpus count mismatch: {artifact.path}")
    return manifest
