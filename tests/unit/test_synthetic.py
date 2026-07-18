from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.synthetic import (
    BenchmarkSplit,
    GeneratedBenchmark,
    KnowledgeKind,
    RecordPermission,
    generate_benchmark,
    generate_from_sources,
    load_benchmark_spec,
    load_profiles,
    write_benchmark,
)
from neuroselect.synthetic.models import KnowledgeRecord, SyntheticProfile, TemplateSpec

ROOT = Path(__file__).parents[2]
SPEC_PATH = ROOT / "synthetic_data" / "benchmark.yaml"
PROFILES_PATH = ROOT / "synthetic_data" / "profiles"
EXPECTED_SOURCE_SHA256 = "9b7d61f2b2024edd40008fa6b6dd69cd97e06096d579f9ce61d23193cd86748f"
EXPECTED_PER_PROFILE = {
    BenchmarkSplit.TRAIN: 1_000,
    BenchmarkSplit.VALIDATION: 150,
    BenchmarkSplit.TEST: 250,
}


@pytest.fixture(scope="module")
def profiles() -> tuple[SyntheticProfile, ...]:
    return load_profiles(PROFILES_PATH)


@pytest.fixture(scope="module")
def benchmark() -> GeneratedBenchmark:
    return generate_from_sources(SPEC_PATH, PROFILES_PATH)


def test_profiles_are_explicitly_synthetic_and_permissioned(
    profiles: tuple[SyntheticProfile, ...],
) -> None:
    assert len(profiles) == 4
    assert tuple(profile.profile_id for profile in profiles) == tuple(
        sorted(profile.profile_id for profile in profiles)
    )
    required_kinds = set(KnowledgeKind)

    for profile in profiles:
        assert profile.synthetic is True
        assert {record.kind for record in profile.knowledge} == required_kinds
        assert all(record.source.startswith("synthetic:") for record in profile.knowledge)
        assert all(record.enabled and record.permissions for record in profile.knowledge)
        current_event = next(
            record for record in profile.knowledge if record.kind is KnowledgeKind.CURRENT_EVENT
        )
        assert current_event.valid_from is not None
        assert current_event.valid_until is not None
        assert current_event.valid_from.utcoffset() is not None
        assert current_event.valid_until > current_event.valid_from.astimezone(UTC)


def test_benchmark_has_exact_isolated_splits(benchmark: GeneratedBenchmark) -> None:
    assert benchmark.source_sha256 == EXPECTED_SOURCE_SHA256
    assert len(benchmark.profile_ids) == 4
    all_message_ids: list[str] = []

    for split, expected_per_profile in EXPECTED_PER_PROFILE.items():
        messages = benchmark.messages[split]
        assert len(messages) == expected_per_profile * 4
        assert Counter(message.profile_id for message in messages) == dict.fromkeys(
            benchmark.profile_ids, expected_per_profile
        )
        assert all(message.split is split and message.synthetic is True for message in messages)
        all_message_ids.extend(message.message_id for message in messages)

    assert len(all_message_ids) == len(set(all_message_ids)) == 5_600

    for profile_id in benchmark.profile_ids:
        texts_by_split = {
            split: {
                message.text
                for message in benchmark.messages[split]
                if message.profile_id == profile_id
            }
            for split in BenchmarkSplit
        }
        template_ids_by_split = {
            split: {message.template_id for message in benchmark.messages[split]}
            for split in BenchmarkSplit
        }
        topics_by_split = {
            split: {message.topic for message in benchmark.messages[split]}
            for split in BenchmarkSplit
        }
        for left, right in (
            (BenchmarkSplit.TRAIN, BenchmarkSplit.VALIDATION),
            (BenchmarkSplit.TRAIN, BenchmarkSplit.TEST),
            (BenchmarkSplit.VALIDATION, BenchmarkSplit.TEST),
        ):
            assert texts_by_split[left].isdisjoint(texts_by_split[right])
            assert template_ids_by_split[left].isdisjoint(template_ids_by_split[right])
            assert topics_by_split[left].isdisjoint(topics_by_split[right])


def test_every_target_span_respects_the_selection_budget(benchmark: GeneratedBenchmark) -> None:
    for messages in benchmark.messages.values():
        for message in messages:
            assert all(1 <= len(span.split()) <= 4 for span in message.target_spans)
            assert message.text == " ".join(message.target_spans)


def test_generation_is_deterministic_and_location_independent(
    benchmark: GeneratedBenchmark, tmp_path: Path
) -> None:
    copied_sources = tmp_path / "copied-sources"
    copied_profiles = copied_sources / "profiles"
    copied_profiles.mkdir(parents=True)
    shutil.copy2(SPEC_PATH, copied_sources / "benchmark.yaml")
    for source in PROFILES_PATH.glob("*.yaml"):
        shutil.copy2(source, copied_profiles / source.name)

    regenerated = generate_from_sources(copied_sources / "benchmark.yaml", copied_profiles)

    assert regenerated == benchmark


def test_writer_emits_reproducible_jsonl_and_manifest(
    benchmark: GeneratedBenchmark, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    manifest = write_benchmark(benchmark, first_dir)
    second_manifest = write_benchmark(benchmark, second_dir)

    assert manifest == second_manifest
    assert manifest.counts == dict.fromkeys(benchmark.profile_ids, EXPECTED_PER_PROFILE)
    assert json.loads((first_dir / "manifest.json").read_text()) == manifest.model_dump(mode="json")

    for artifact in manifest.artifacts:
        first_content = (first_dir / artifact.path).read_bytes()
        assert first_content == (second_dir / artifact.path).read_bytes()
        assert hashlib.sha256(first_content).hexdigest() == artifact.sha256
        assert first_content.count(b"\n") == artifact.message_count


def test_loader_rejects_non_mapping_and_missing_profiles(tmp_path: Path) -> None:
    invalid_spec = tmp_path / "invalid.yaml"
    invalid_spec.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_benchmark_spec(invalid_spec)
    with pytest.raises(ValueError, match="no synthetic profile"):
        load_profiles(tmp_path / "missing")


def test_loader_rejects_duplicate_profile_ids(tmp_path: Path) -> None:
    duplicate_dir = tmp_path / "profiles"
    duplicate_dir.mkdir()
    source = next(PROFILES_PATH.glob("*.yaml"))
    shutil.copy2(source, duplicate_dir / "first.yaml")
    shutil.copy2(source, duplicate_dir / "second.yaml")

    with pytest.raises(ValueError, match="profile IDs must be unique"):
        load_profiles(duplicate_dir)


def test_source_models_reject_unsafe_or_ambiguous_records() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        KnowledgeRecord(
            record_id="external-fact",
            kind=KnowledgeKind.ROUTINE,
            content="A private imported fact.",
            source="private:message-log",
            permissions=frozenset({RecordPermission.SUGGEST}),
        )
    with pytest.raises(ValidationError, match="current-event records require valid_until"):
        KnowledgeRecord(
            record_id="event-without-expiry",
            kind=KnowledgeKind.CURRENT_EVENT,
            content="A temporary synthetic event.",
            source="synthetic:test",
            permissions=frozenset({RecordPermission.SUGGEST}),
        )
    with pytest.raises(ValidationError, match="valid_until must be later"):
        KnowledgeRecord(
            record_id="reversed-window",
            kind=KnowledgeKind.PREFERENCE,
            content="A synthetic preference.",
            source="synthetic:test",
            permissions=frozenset({RecordPermission.EXPLAIN}),
            valid_from=datetime(2026, 7, 18, tzinfo=UTC),
            valid_until=datetime(2026, 7, 17, tzinfo=UTC),
        )


def test_template_model_allows_only_simple_slot_names() -> None:
    template = TemplateSpec(
        template_id="valid-template",
        topic="valid-topic",
        segments=("Please {action}", "{object}"),
    )
    assert template.slot_names == ("action", "object")

    with pytest.raises(ValidationError, match="simple identifiers"):
        TemplateSpec(
            template_id="invalid-template",
            topic="invalid-topic",
            segments=("Please {action!r}",),
        )


def test_generation_rejects_missing_spans_and_insufficient_variation(
    profiles: tuple[SyntheticProfile, ...],
) -> None:
    spec = load_benchmark_spec(SPEC_PATH)
    missing_style_slots = profiles[0].model_copy(update={"slots": {}})
    with pytest.raises(ValueError, match="is missing slots"):
        generate_benchmark(
            spec=spec,
            profiles=(missing_style_slots,),
            source_sha256="a" * 64,
        )

    one_value_spec = spec.model_copy(
        update={"global_slots": {name: (values[0],) for name, values in spec.global_slots.items()}}
    )
    one_value_profile = profiles[0].model_copy(
        update={"slots": {name: (values[0],) for name, values in profiles[0].slots.items()}}
    )
    with pytest.raises(ValueError, match="unique messages but requires"):
        generate_benchmark(
            spec=one_value_spec,
            profiles=(one_value_profile,),
            source_sha256="b" * 64,
        )


def test_generation_rejects_empty_profiles_and_oversized_spans(
    profiles: tuple[SyntheticProfile, ...],
) -> None:
    spec = load_benchmark_spec(SPEC_PATH)
    with pytest.raises(ValueError, match="at least one synthetic profile"):
        generate_benchmark(spec=spec, profiles=(), source_sha256="a" * 64)

    one_token_spec = spec.model_copy(update={"maximum_span_tokens": 1})
    with pytest.raises(ValueError, match="spans longer than 1 tokens"):
        generate_benchmark(
            spec=one_token_spec,
            profiles=(profiles[0],),
            source_sha256="b" * 64,
        )
