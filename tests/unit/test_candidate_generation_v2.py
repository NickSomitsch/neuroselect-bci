from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from neuroselect.evaluation.candidate_generation_v2 import (
    CandidateGenerationV2Spec,
    TargetBlindContextualGeneratorV2,
    build_candidate_bank_v2,
    evaluate_candidate_generation_v2,
    load_candidate_generation_v2_spec,
)
from neuroselect.evaluation.candidate_generation_v2_artifacts import (
    read_candidate_generation_v2_artifacts,
    write_candidate_generation_v2_artifacts,
)
from neuroselect.synthetic import BenchmarkMessage, BenchmarkSplit, GeneratedBenchmark

ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 7, 27, 23, 30, tzinfo=UTC)


def spec(**updates: object) -> CandidateGenerationV2Spec:
    values: dict[str, object] = {
        "experiment_id": "candidate-v2-test",
        "protocol_revision": "candidate-generation-v2-exploratory-v1",
        "generated_at": NOW,
        "publication_protocol": Path("protocol.yaml"),
        "expected_protocol_sha256": "a" * 64,
        "primary_language_artifacts": Path("language"),
        "expected_primary_language_manifest_sha256": "b" * 64,
        "benchmark_spec": Path("benchmark.yaml"),
        "profiles_directory": Path("profiles"),
        "expected_benchmark_source_sha256": "c" * 64,
        "fitting_source_splits": ("train", "validation"),
        "language_candidate_count": 9,
        "maximum_phrase_tokens": 4,
        "request_object_time_quota": 7,
        "request_object_location_quota": 1,
        "request_object_ending_quota": 1,
        "bootstrap_resamples": 2_000,
        "bootstrap_seed": 7,
        "design_status": "exploratory_test_exposed",
    }
    values.update(updates)
    return CandidateGenerationV2Spec.model_validate(values)


def message(
    index: int,
    split: BenchmarkSplit,
    spans: tuple[str, ...],
) -> BenchmarkMessage:
    return BenchmarkMessage(
        message_id=f"msg-{index:020x}",
        profile_id="synthetic-test",
        split=split,
        template_id=f"{split.value}-{index}",
        topic=f"{split.value}-topic",
        text=" ".join(spans),
        target_spans=spans,
    )


def benchmark() -> GeneratedBenchmark:
    train = tuple(
        message(
            index,
            BenchmarkSplit.TRAIN,
            (
                f"Could you action{index}",
                f"the item{index}",
                f"to the room{index}",
                "please.",
            ),
        )
        for index in range(1, 13)
    )
    validation = tuple(
        message(
            100 + index,
            BenchmarkSplit.VALIDATION,
            (
                "Before",
                f"time{index}",
                f"Could you action{index}",
                f"the item{index}",
                "please.",
            ),
        )
        for index in range(1, 13)
    )
    test = (
        message(
            999,
            BenchmarkSplit.TEST,
            (
                "Could you action1",
                "the item1",
                "Before time1",
                "please.",
            ),
        ),
    )
    return GeneratedBenchmark(
        schema_version="1.0",
        source_sha256="c" * 64,
        profile_ids=("synthetic-test",),
        messages={
            BenchmarkSplit.TRAIN: train,
            BenchmarkSplit.VALIDATION: validation,
            BenchmarkSplit.TEST: test,
        },
    )


def baseline_trials(source: GeneratedBenchmark) -> Any:
    target = source.messages[BenchmarkSplit.TEST][0]
    rows = []
    confirmed: list[str] = []
    for index, intended in enumerate(target.target_spans):
        rows.append(
            SimpleNamespace(
                trial_id=f"language-{target.message_id}-{index:02d}",
                profile_id=target.profile_id,
                message_id=target.message_id,
                span_index=index,
                message_span_count=len(target.target_spans),
                confirmed_context=" ".join(confirmed),
                intended_text=intended,
                target_available=False,
            )
        )
        confirmed.append(intended)
    return cast(Any, rows)


def test_tracked_candidate_v2_recipe_is_locked_and_strict() -> None:
    tracked = load_candidate_generation_v2_spec(
        ROOT / "configs/publication/candidate_generation_v2_exploratory.yaml"
    )
    assert tracked.fitting_source_splits == ("train", "validation")
    assert tracked.language_candidate_count == 9
    assert tracked.design_status == "exploratory_test_exposed"
    with pytest.raises(ValidationError, match="validation"):
        spec(fitting_source_splits=("train", "test"))
    with pytest.raises(ValidationError, match="routing quotas"):
        spec(request_object_time_quota=6)


def test_candidate_bank_is_test_invariant_and_generator_has_no_target_argument() -> None:
    source = benchmark()
    first = build_candidate_bank_v2(source, spec())
    changed_test = source.model_copy(
        update={
            "messages": {
                **source.messages,
                BenchmarkSplit.TEST: (
                    message(
                        998,
                        BenchmarkSplit.TEST,
                        ("A hidden replacement", "must not affect fitting"),
                    ),
                ),
            }
        }
    )
    second = build_candidate_bank_v2(changed_test, spec())

    assert first.digest() == second.digest()
    assert not (
        set(first.source_message_ids)
        & {item.message_id for item in source.messages[BenchmarkSplit.TEST]}
    )
    parameters = inspect.signature(TargetBlindContextualGeneratorV2.generate).parameters
    assert "intended_text" not in parameters
    assert "target" not in parameters
    candidates = TargetBlindContextualGeneratorV2(first, spec()).generate(
        profile_id="synthetic-test",
        confirmed_context="Could you action1",
        span_index=1,
    )
    assert len(candidates) == 9
    assert len({candidate.text.casefold() for candidate in candidates}) == 9


def test_candidate_v2_evaluation_and_artifacts_are_paired_and_verified(
    tmp_path: Path,
) -> None:
    source = benchmark()
    recipe = spec()
    bank = build_candidate_bank_v2(source, recipe)
    result = evaluate_candidate_generation_v2(
        benchmark=source,
        baseline_trials=baseline_trials(source),
        bank=bank,
        spec=recipe,
        primary_language_manifest_sha256="b" * 64,
        protocol_sha256="a" * 64,
    )

    overall = next(metric for metric in result.metrics if metric.scope == "overall")
    assert result.intended_target_exposed_to_generator is False
    assert overall.baseline_target_availability_rate == 0.0
    assert overall.v2_target_availability_rate > 0.0
    assert len(result.intervals) == 4
    manifest = write_candidate_generation_v2_artifacts(
        result,
        bank,
        recipe,
        tmp_path,
        git_sha="b239179",
        package_versions={"python": "3.12"},
        device={"system": "test"},
    )
    restored = read_candidate_generation_v2_artifacts(tmp_path)
    assert restored == (result, bank, manifest)

    (tmp_path / "metrics.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_candidate_generation_v2_artifacts(tmp_path)
