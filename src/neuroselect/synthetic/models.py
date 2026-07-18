"""Typed source and output models for the public synthetic benchmark."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import KnowledgeKind, RecordPermission


class BenchmarkSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class KnowledgeRecord(BaseModel):
    """An explicitly synthetic, permissioned fact kept separate from style data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1, max_length=128)
    kind: KnowledgeKind
    content: str = Field(min_length=1, max_length=500)
    source: str = Field(pattern=r"^synthetic:", max_length=256)
    permissions: frozenset[RecordPermission] = Field(min_length=1)
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_validity_window(self) -> KnowledgeRecord:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("valid_until must be later than valid_from")
        if self.kind is KnowledgeKind.CURRENT_EVENT and self.valid_until is None:
            raise ValueError("current-event records require valid_until")
        return self


class StyleSpec(BaseModel):
    """Stable communication traits suitable for later LoRA training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tone: str = Field(min_length=1, max_length=120)
    formality: Literal["low", "medium", "high"]
    sentence_pattern: str = Field(min_length=1, max_length=200)
    preferred_vocabulary: tuple[str, ...] = Field(min_length=3)


class SyntheticProfile(BaseModel):
    """A public persona containing stable style slots and changeable knowledge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=r"^synthetic-[a-z0-9-]+$", max_length=128)
    display_name: str = Field(min_length=1, max_length=120)
    synthetic: Literal[True]
    style_summary: str = Field(min_length=1, max_length=500)
    style: StyleSpec
    slots: dict[str, tuple[str, ...]] = Field(min_length=1)
    knowledge: tuple[KnowledgeRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile_collections(self) -> SyntheticProfile:
        empty_slots = [name for name, values in self.slots.items() if not values]
        if empty_slots:
            raise ValueError(f"profile slots cannot be empty: {', '.join(sorted(empty_slots))}")
        record_ids = [record.record_id for record in self.knowledge]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("knowledge record IDs must be unique within a profile")
        return self


class TemplateSpec(BaseModel):
    """A split-specific message template composed of selectable short spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str = Field(min_length=1, max_length=128)
    topic: str = Field(min_length=1, max_length=128)
    segments: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_template_fields(self) -> TemplateSpec:
        for segment in self.segments:
            for _, field_name, format_spec, conversion in Formatter().parse(segment):
                if field_name is None:
                    continue
                if not field_name.isidentifier() or format_spec or conversion:
                    raise ValueError("template slots must be simple identifiers without formatting")
        return self

    @property
    def slot_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for segment in self.segments:
            for _, field_name, _, _ in Formatter().parse(segment):
                if field_name is not None and field_name not in names:
                    names.append(field_name)
        return tuple(names)


class SplitSpec(BaseModel):
    """Requested size and source templates for one benchmark split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count_per_profile: int = Field(ge=1, le=10_000)
    templates: tuple[TemplateSpec, ...] = Field(min_length=1)


class BenchmarkSpec(BaseModel):
    """Versioned benchmark recipe; generated messages are derived artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    seed: int = Field(ge=0)
    maximum_span_tokens: int = Field(default=4, ge=1, le=8)
    global_slots: dict[str, tuple[str, ...]] = Field(min_length=1)
    splits: dict[BenchmarkSplit, SplitSpec]

    @model_validator(mode="after")
    def validate_split_isolation(self) -> BenchmarkSpec:
        if set(self.splits) != set(BenchmarkSplit):
            raise ValueError("benchmark must define train, validation, and test splits")
        empty_slots = [name for name, values in self.global_slots.items() if not values]
        if empty_slots:
            raise ValueError(f"global slots cannot be empty: {', '.join(sorted(empty_slots))}")

        seen_template_ids: set[str] = set()
        seen_topics: set[str] = set()
        for split in BenchmarkSplit:
            split_spec = self.splits[split]
            template_ids = {template.template_id for template in split_spec.templates}
            topics = {template.topic for template in split_spec.templates}
            if seen_template_ids.intersection(template_ids):
                raise ValueError("template IDs must be disjoint across splits")
            if seen_topics.intersection(topics):
                raise ValueError("topics must be disjoint across splits")
            seen_template_ids.update(template_ids)
            seen_topics.update(topics)
        return self


class BenchmarkMessage(BaseModel):
    """One intended synthetic message with deterministic selection spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(pattern=r"^msg-[0-9a-f]{20}$")
    profile_id: str
    split: BenchmarkSplit
    template_id: str
    topic: str
    text: str = Field(min_length=1, max_length=500)
    target_spans: tuple[str, ...] = Field(min_length=1)
    synthetic: Literal[True] = True


class GeneratedBenchmark(BaseModel):
    """In-memory deterministic benchmark ready for artifact serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_ids: tuple[str, ...]
    messages: dict[BenchmarkSplit, tuple[BenchmarkMessage, ...]]


class SplitArtifact(BaseModel):
    """Digest and size of one emitted JSONL split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: BenchmarkSplit
    path: str
    message_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkManifest(BaseModel):
    """Machine-readable description of a generated benchmark artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_ids: tuple[str, ...]
    counts: dict[str, dict[BenchmarkSplit, int]]
    artifacts: tuple[SplitArtifact, ...]
