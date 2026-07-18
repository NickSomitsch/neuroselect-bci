"""Generate leak-resistant synthetic corpora from compact versioned sources."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from neuroselect.synthetic.models import (
    BenchmarkManifest,
    BenchmarkMessage,
    BenchmarkSpec,
    BenchmarkSplit,
    GeneratedBenchmark,
    SplitArtifact,
    SyntheticProfile,
    TemplateSpec,
)

DEFAULT_SPEC_PATH = Path("synthetic_data/benchmark.yaml")
DEFAULT_PROFILES_DIR = Path("synthetic_data/profiles")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source_file:
        payload = yaml.safe_load(source_file)
    if not isinstance(payload, dict):
        raise ValueError(f"synthetic source must contain a YAML mapping: {path}")
    return payload


def load_benchmark_spec(path: str | Path = DEFAULT_SPEC_PATH) -> BenchmarkSpec:
    """Load the versioned benchmark recipe."""

    return BenchmarkSpec.model_validate(_load_yaml_mapping(Path(path)))


def load_profiles(directory: str | Path = DEFAULT_PROFILES_DIR) -> tuple[SyntheticProfile, ...]:
    """Load public synthetic profiles in stable profile-ID order."""

    profile_paths = sorted(Path(directory).glob("*.yaml"))
    if not profile_paths:
        raise ValueError(f"no synthetic profile YAML files found in {directory}")
    profiles = tuple(
        SyntheticProfile.model_validate(_load_yaml_mapping(profile_path))
        for profile_path in profile_paths
    )
    profile_ids = [profile.profile_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("synthetic profile IDs must be unique")
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def _source_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        # Source locations differ between checkouts, so only stable filenames and
        # file contents participate in the reproducibility digest.
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _seed_for(base_seed: int, profile_id: str, split: BenchmarkSplit) -> int:
    material = f"{base_seed}:{profile_id}:{split.value}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _normalize_segment(segment: str) -> str:
    normalized = re.sub(r"\s+", " ", segment).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", normalized)


def _candidate_messages(
    *,
    profile: SyntheticProfile,
    template: TemplateSpec,
    global_slots: dict[str, tuple[str, ...]],
    maximum_span_tokens: int,
) -> list[tuple[str, str, tuple[str, ...]]]:
    slots = {**global_slots, **profile.slots}
    missing_slots = [name for name in template.slot_names if name not in slots]
    if missing_slots:
        raise ValueError(
            f"profile {profile.profile_id} is missing slots for {template.template_id}: "
            f"{', '.join(missing_slots)}"
        )

    values = [slots[name] for name in template.slot_names]
    candidates: list[tuple[str, str, tuple[str, ...]]] = []
    for combination in itertools.product(*values):
        context = dict(zip(template.slot_names, combination, strict=True))
        spans = tuple(
            normalized
            for segment in template.segments
            if (normalized := _normalize_segment(segment.format_map(context)))
        )
        oversized = [span for span in spans if len(span.split()) > maximum_span_tokens]
        if oversized:
            raise ValueError(
                f"template {template.template_id} produced spans longer than "
                f"{maximum_span_tokens} tokens: {oversized[0]!r}"
            )
        text = _normalize_segment(" ".join(spans))
        candidates.append((template.template_id, text, spans))
    return candidates


def _generate_profile_split(
    *, profile: SyntheticProfile, split: BenchmarkSplit, spec: BenchmarkSpec
) -> tuple[BenchmarkMessage, ...]:
    split_spec = spec.splits[split]
    unique: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for template in split_spec.templates:
        for template_id, text, spans in _candidate_messages(
            profile=profile,
            template=template,
            global_slots=spec.global_slots,
            maximum_span_tokens=spec.maximum_span_tokens,
        ):
            unique.setdefault(text.casefold(), (template_id, text, spans))

    candidates = sorted(unique.values(), key=lambda value: (value[0], value[1]))
    if len(candidates) < split_spec.count_per_profile:
        raise ValueError(
            f"split {split.value} for {profile.profile_id} produced {len(candidates)} unique "
            f"messages but requires {split_spec.count_per_profile}"
        )
    random.Random(_seed_for(spec.seed, profile.profile_id, split)).shuffle(candidates)

    topic_by_template = {template.template_id: template.topic for template in split_spec.templates}
    messages: list[BenchmarkMessage] = []
    for template_id, text, spans in candidates[: split_spec.count_per_profile]:
        content_digest = hashlib.sha256(
            f"{profile.profile_id}\0{split.value}\0{template_id}\0{text}".encode()
        ).hexdigest()
        messages.append(
            BenchmarkMessage(
                message_id=f"msg-{content_digest[:20]}",
                profile_id=profile.profile_id,
                split=split,
                template_id=template_id,
                topic=topic_by_template[template_id],
                text=text,
                target_spans=spans,
            )
        )
    return tuple(messages)


def generate_benchmark(
    *, spec: BenchmarkSpec, profiles: tuple[SyntheticProfile, ...], source_sha256: str
) -> GeneratedBenchmark:
    """Generate exact split sizes without reading or writing private data."""

    if not profiles:
        raise ValueError("at least one synthetic profile is required")
    messages = {
        split: tuple(
            message
            for profile in profiles
            for message in _generate_profile_split(profile=profile, split=split, spec=spec)
        )
        for split in BenchmarkSplit
    }
    return GeneratedBenchmark(
        schema_version=spec.schema_version,
        source_sha256=source_sha256,
        profile_ids=tuple(profile.profile_id for profile in profiles),
        messages=messages,
    )


def generate_from_sources(
    spec_path: str | Path = DEFAULT_SPEC_PATH,
    profiles_dir: str | Path = DEFAULT_PROFILES_DIR,
) -> GeneratedBenchmark:
    """Load tracked sources and generate the complete deterministic benchmark."""

    spec_file = Path(spec_path)
    profile_directory = Path(profiles_dir)
    profile_paths = sorted(profile_directory.glob("*.yaml"))
    source_sha256 = _source_digest((spec_file, *profile_paths))
    return generate_benchmark(
        spec=load_benchmark_spec(spec_file),
        profiles=load_profiles(profile_directory),
        source_sha256=source_sha256,
    )


def _canonical_message_line(message: BenchmarkMessage) -> str:
    payload = message.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_benchmark(benchmark: GeneratedBenchmark, output_dir: str | Path) -> BenchmarkManifest:
    """Write JSONL splits and a checksum manifest to an artifact directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[SplitArtifact] = []
    counts = {profile_id: dict.fromkeys(BenchmarkSplit, 0) for profile_id in benchmark.profile_ids}

    for split in BenchmarkSplit:
        messages = benchmark.messages[split]
        content = "\n".join(_canonical_message_line(message) for message in messages) + "\n"
        output_path = destination / f"{split.value}.jsonl"
        output_path.write_text(content, encoding="utf-8")
        for message in messages:
            counts[message.profile_id][split] += 1
        artifacts.append(
            SplitArtifact(
                split=split,
                path=output_path.name,
                message_count=len(messages),
                sha256=hashlib.sha256(content.encode()).hexdigest(),
            )
        )

    manifest = BenchmarkManifest(
        schema_version=benchmark.schema_version,
        source_sha256=benchmark.source_sha256,
        profile_ids=benchmark.profile_ids,
        counts=counts,
        artifacts=tuple(artifacts),
    )
    manifest_content = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    (destination / "manifest.json").write_text(manifest_content + "\n", encoding="utf-8")
    return manifest
