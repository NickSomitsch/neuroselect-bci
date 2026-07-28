from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.evaluation.candidate_generation_step4 import (
    AblatedCandidateGenerator,
    CandidateGenerationDataset,
    CandidateGenerationMethod,
    CandidateGenerationStep4Spec,
    TargetBlindTwoStageOpeningGenerator,
    evaluate_candidate_generation_step4,
    load_candidate_generation_step4_spec,
    robustness_evaluation_spans,
    validate_robustness_opening_holdout,
)
from neuroselect.evaluation.candidate_generation_step4_artifacts import (
    read_candidate_generation_step4_artifacts,
    write_candidate_generation_step4_artifacts,
)
from neuroselect.evaluation.candidate_generation_v2 import (
    CandidateGenerationV2Spec,
    build_candidate_bank_v2,
)
from neuroselect.synthetic import (
    BenchmarkMessage,
    BenchmarkSplit,
    GeneratedBenchmark,
    generate_from_sources,
)

ROOT = Path(__file__).resolve().parents[2]


def v2_spec() -> CandidateGenerationV2Spec:
    return CandidateGenerationV2Spec.model_validate(
        {
            "schema_version": "1.0",
            "experiment_id": "test-v2",
            "protocol_revision": "candidate-generation-v2-exploratory-v1",
            "generated_at": datetime.fromisoformat("2026-07-28T09:00:00+02:00"),
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
    )


def step4_spec(**updates: object) -> CandidateGenerationStep4Spec:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": "test-step4",
        "protocol_revision": "candidate-generation-step4-v1",
        "locked_at": datetime.fromisoformat("2026-07-28T09:30:00+02:00"),
        "publication_protocol": Path("protocol.yaml"),
        "expected_protocol_sha256": "a" * 64,
        "step3_artifacts": Path("step3"),
        "expected_step3_manifest_sha256": "b" * 64,
        "expected_step3_config_sha256": "c" * 64,
        "robustness_benchmark_spec": Path("robustness.yaml"),
        "profiles_directory": Path("profiles"),
        "expected_robustness_source_sha256": "d" * 64,
        "methods": tuple(CandidateGenerationMethod),
        "language_candidate_count": 9,
        "maximum_phrase_tokens": 4,
        "bootstrap_resamples": 2_000,
        "bootstrap_seed": 9,
        "design_status": "locked_before_execution_exploratory",
        "outcome_based_omission_forbidden": True,
    }
    values.update(updates)
    return CandidateGenerationStep4Spec.model_validate(values)


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
    stems = tuple(f"Stem {index}" for index in range(9))
    actions = tuple(f"action{index}" for index in range(9))
    train = tuple(
        message(
            index + 1,
            BenchmarkSplit.TRAIN,
            (
                f"{stems[index]} {actions[index]}",
                f"the item{index}",
                f"to the room{index}",
                f"ending{index}.",
            ),
        )
        for index in range(9)
    )
    validation = tuple(
        message(
            index + 101,
            BenchmarkSplit.VALIDATION,
            (
                f"{stems[index]} {actions[(index + 1) % 9]}",
                f"the item{index}",
                "Before",
                f"time{index}",
                f"ending{index}.",
            ),
        )
        for index in range(9)
    )
    test = (
        message(
            999,
            BenchmarkSplit.TEST,
            (
                f"{stems[0]} {actions[2]}",
                "the item0",
                "Before time0",
                "ending0.",
            ),
        ),
    )
    return GeneratedBenchmark(
        schema_version="1.0",
        source_sha256="d" * 64,
        profile_ids=("synthetic-test",),
        messages={
            BenchmarkSplit.TRAIN: train,
            BenchmarkSplit.VALIDATION: validation,
            BenchmarkSplit.TEST: test,
        },
    )


def test_step4_recipe_and_tracked_robustness_source_are_locked() -> None:
    tracked = load_candidate_generation_step4_spec(
        ROOT / "configs/publication/candidate_generation_step4.yaml"
    )
    assert tracked.methods == tuple(CandidateGenerationMethod)
    assert tracked.outcome_based_omission_forbidden is True
    source = generate_from_sources(
        ROOT / tracked.robustness_benchmark_spec,
        ROOT / tracked.profiles_directory,
    )
    assert source.source_sha256 == tracked.expected_robustness_source_sha256
    validate_robustness_opening_holdout(source)
    with pytest.raises(ValidationError, match="methods"):
        step4_spec(methods=(CandidateGenerationMethod.FULL_V2,))


def test_step4_generators_are_target_blind_and_openings_are_compositional() -> None:
    source = benchmark()
    validate_robustness_opening_holdout(source)
    bank = build_candidate_bank_v2(source, v2_spec())
    for generator_type in (AblatedCandidateGenerator, TargetBlindTwoStageOpeningGenerator):
        for method_name in ("generate", "generate_stems", "generate_actions"):
            method = getattr(generator_type, method_name, None)
            if method is None:
                continue
            parameters = inspect.signature(method).parameters
            assert "target" not in parameters
            assert "intended_text" not in parameters

    opening = TargetBlindTwoStageOpeningGenerator(bank)
    stems = opening.generate_stems(profile_id="synthetic-test")
    assert "Stem 0" in stems
    actions = opening.generate_actions(
        profile_id="synthetic-test",
        selected_stem="Stem 0",
    )
    assert "action2" in actions


def test_step4_evaluation_artifact_round_trip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    source = benchmark()
    recipe = step4_spec()
    language_recipe = v2_spec()
    bank = build_candidate_bank_v2(source, language_recipe)
    spans = robustness_evaluation_spans(source)
    result = evaluate_candidate_generation_step4(
        spec=recipe,
        v2_spec=language_recipe,
        existing_spans=spans,
        existing_bank=bank,
        robustness_spans=spans,
        robustness_bank=bank,
        protocol_sha256="a" * 64,
        step3_manifest_sha256="b" * 64,
    )
    assert result.intended_target_exposed_to_generators is False
    assert len(result.trials) == len(spans) * len(CandidateGenerationMethod) * 2
    robustness_opening = next(
        metric
        for metric in result.metrics
        if metric.dataset_id is CandidateGenerationDataset.ROBUSTNESS_HOLDOUT
        and metric.method is CandidateGenerationMethod.TWO_STAGE_OPENING
        and metric.scope == "opening"
    )
    assert robustness_opening.availability_rate == 1.0
    manifest = write_candidate_generation_step4_artifacts(
        result,
        bank,
        bank,
        source,
        recipe,
        tmp_path,
        git_sha="81cc5e4",
        package_versions={"python": "3.12"},
        device={"system": "test"},
    )
    restored = read_candidate_generation_step4_artifacts(tmp_path)
    assert restored == (result, bank, bank, manifest)

    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(metrics_path.read_text() + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_candidate_generation_step4_artifacts(tmp_path)
