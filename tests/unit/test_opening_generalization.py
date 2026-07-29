from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.evaluation.opening_generalization import (
    OpeningChallenge,
    OpeningGeneralizationSource,
    OpeningGeneralizationSpec,
    OpeningMethod,
    TargetBlindGlobalTwoStageOpeningGenerator,
    TargetBlindIntentThreeStageOpeningGenerator,
    TargetBlindPhraseOpeningGenerator,
    build_opening_training_bank,
    evaluate_opening_generalization,
    generate_opening_records,
    load_opening_generalization_source,
    load_opening_generalization_spec,
    validate_opening_holdouts,
)
from neuroselect.evaluation.opening_generalization_artifacts import (
    read_opening_generalization_artifacts,
    write_opening_generalization_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]


def experiment_spec(**updates: object) -> OpeningGeneralizationSpec:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": "test-opening-generalization",
        "protocol_revision": "opening-generalization-experiment-v1",
        "locked_at": datetime.fromisoformat("2026-07-28T14:16:00+02:00"),
        "publication_protocol": Path("protocol.yaml"),
        "expected_protocol_sha256": "a" * 64,
        "step4_artifacts": Path("step4"),
        "expected_step4_manifest_sha256": "b" * 64,
        "benchmark_source": Path("opening.yaml"),
        "expected_benchmark_source_sha256": "c" * 64,
        "methods": tuple(OpeningMethod),
        "candidate_budget": 9,
        "bootstrap_resamples": 2_000,
        "bootstrap_seed": 11,
        "design_status": "locked_before_execution_exploratory",
        "outcome_based_omission_forbidden": True,
    }
    values.update(updates)
    return OpeningGeneralizationSpec.model_validate(values)


def tracked_source() -> OpeningGeneralizationSource:
    return load_opening_generalization_source(
        ROOT / "synthetic_data/opening_generalization_v1.yaml"
    )


def test_opening_recipe_and_harder_holdouts_are_locked() -> None:
    spec = load_opening_generalization_spec(
        ROOT / "configs/publication/opening_generalization_v1.yaml"
    )
    source = tracked_source()
    records = generate_opening_records(source)
    counts = validate_opening_holdouts(records)
    assert spec.methods == tuple(OpeningMethod)
    assert source.digest() == spec.expected_benchmark_source_sha256
    assert counts == {
        "fit_record_count": 864,
        "combination_test_count": 288,
        "family_test_count": 384,
        "fitted_stem_count": 24,
        "heldout_family_stem_count": 8,
        "content_count": 48,
        "fit_test_opening_overlap_count": 0,
    }
    assert counts["fitted_stem_count"] > source.candidate_budget
    assert counts["content_count"] > source.candidate_budget
    with pytest.raises(ValidationError, match="methods"):
        experiment_spec(methods=(OpeningMethod.ONE_STAGE_PHRASE,))


def test_hierarchical_generators_are_target_blind() -> None:
    source = tracked_source()
    records = generate_opening_records(source)
    bank = build_opening_training_bank(records, source)
    generators = (
        TargetBlindPhraseOpeningGenerator(bank),
        TargetBlindGlobalTwoStageOpeningGenerator(bank),
        TargetBlindIntentThreeStageOpeningGenerator(bank),
    )
    for generator in generators:
        for method_name in (
            "generate",
            "generate_intents",
            "generate_stems",
            "generate_contents",
        ):
            method = getattr(generator, method_name, None)
            if method is None:
                continue
            parameters = inspect.signature(method).parameters
            assert "target" not in parameters
            assert "intended_opening" not in parameters
            assert "intended_content" not in parameters

    two_stage = generators[1]
    assert isinstance(two_stage, TargetBlindGlobalTwoStageOpeningGenerator)
    stems = two_stage.generate_stems(profile_id=source.profile_ids[0])
    assert len(stems) == source.candidate_budget
    assert "Could you perhaps" not in stems


def test_opening_evaluation_and_artifacts_are_paired_and_verified(
    tmp_path: Path,
) -> None:
    source = tracked_source()
    records = generate_opening_records(source)
    bank = build_opening_training_bank(records, source)
    spec = experiment_spec(expected_benchmark_source_sha256=source.digest())
    result = evaluate_opening_generalization(
        spec=spec,
        source=source,
        records=records,
        bank=bank,
        protocol_sha256="a" * 64,
        step4_manifest_sha256="b" * 64,
    )
    assert result.intended_opening_exposed_to_generators is False
    combination = {
        metric.method: metric
        for metric in result.metrics
        if metric.challenge == OpeningChallenge.HELDOUT_COMBINATION.value
        and metric.scope == "overall"
    }
    assert combination[OpeningMethod.ONE_STAGE_PHRASE].availability_rate == 0.0
    assert (
        combination[OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT].availability_rate
        > combination[OpeningMethod.TWO_STAGE_STEM_CONTENT].availability_rate
    )
    family = [
        metric
        for metric in result.metrics
        if metric.challenge == OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY.value
        and metric.scope == "overall"
    ]
    assert all(metric.availability_rate == 0.0 for metric in family)

    manifest = write_opening_generalization_artifacts(
        result,
        bank,
        records,
        spec,
        tmp_path,
        git_sha="44746b8",
        package_versions={"python": "3.12"},
        device={"system": "test"},
    )
    assert read_opening_generalization_artifacts(tmp_path) == (result, bank, manifest)
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(metrics_path.read_text() + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_opening_generalization_artifacts(tmp_path)
