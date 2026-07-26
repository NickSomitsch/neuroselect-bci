from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from neuroselect.bci import SimulationConfig
from neuroselect.evaluation import (
    ConditionAvailability,
    EvaluationCondition,
    ExperimentConfigurationError,
    SimulatedExperimentRunner,
    SimulatedExperimentSpec,
    capture_runtime_environment,
    condition_catalog,
    load_experiment_spec,
    write_experiment_artifacts,
)
from neuroselect.ranking import RankingDisposition
from neuroselect.synthetic import generate_from_sources, load_profiles

ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/simulated_vertical_slice.yaml"
PROFILES = load_profiles(ROOT / "synthetic_data/profiles")
BENCHMARK = generate_from_sources(
    ROOT / "synthetic_data/benchmark.yaml",
    ROOT / "synthetic_data/profiles",
)


def make_spec(
    *conditions: EvaluationCondition,
    conflict_every: int = 4,
    clean_neural: bool = True,
    profile_ids: tuple[str, ...] = ("synthetic-concise",),
) -> SimulatedExperimentSpec:
    default = load_experiment_spec(DEFAULT_CONFIG)
    simulator = default.simulator
    if clean_neural:
        simulator = SimulationConfig(
            seed=default.seed,
            target_concentration=100.0,
            distractor_concentration=1.0,
            ambiguous_concentration=10.0,
            lapse_probability=0.0,
            ambiguous_probability=0.0,
            timeline_origin=default.simulator.timeline_origin,
        )
    return SimulatedExperimentSpec.model_validate(
        {
            **default.model_dump(),
            "experiment_id": "unit-evaluation",
            "profile_ids": profile_ids,
            "message_limit_per_profile": 1,
            "candidate_count": 6,
            "conditions": conditions,
            "language_conflict_every_n_trials": conflict_every,
            "simulator": simulator,
        }
    )


def run_spec(spec: SimulatedExperimentSpec):  # type: ignore[no-untyped-def]
    return SimulatedExperimentRunner(spec).run(benchmark=BENCHMARK, profiles=PROFILES)


def test_runtime_environment_preserves_torch_build_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")

    package_versions, _ = capture_runtime_environment()

    assert package_versions["torch"] == "2.13.0+cu130"


def test_runtime_environment_captures_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 9))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=24 * 1024**3),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    _, device = capture_runtime_environment()

    assert device["accelerator"] == "cuda"
    assert device["gpu_name"] == "Test GPU"
    assert device["cuda_compute_capability"] == "8.9"
    assert device["cuda_runtime"] == "13.0"
    assert device["gpu_memory_bytes"] == str(24 * 1024**3)


def test_condition_catalog_keeps_artifact_gated_lora_and_complete_runs_unavailable() -> None:
    catalog = condition_catalog()

    assert {item.condition for item in catalog} == set(EvaluationCondition)
    unavailable = {
        item.condition: item.unavailable_reason
        for item in catalog
        if item.availability is ConditionAvailability.UNAVAILABLE
    }
    assert set(unavailable) == {
        EvaluationCondition.D_NEURAL_PERSONALIZED,
        EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
        EvaluationCondition.F_COMPLETE_SYSTEM,
        EvaluationCondition.ABLATION_REMOVE_CONTEXT,
    }
    assert all(unavailable.values())

    with pytest.raises(ExperimentConfigurationError, match="unmet dependencies"):
        run_spec(make_spec(EvaluationCondition.F_COMPLETE_SYSTEM))
    with pytest.raises(ExperimentConfigurationError, match="context-sensitive language"):
        run_spec(make_spec(EvaluationCondition.ABLATION_REMOVE_CONTEXT))


def test_spec_rejects_duplicate_conditions_profiles_naive_time_and_seed_drift() -> None:
    default = load_experiment_spec(DEFAULT_CONFIG)
    payload = default.model_dump()
    payload["conditions"] = [
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.A_BCI_ONLY,
    ]
    with pytest.raises(ValidationError, match="conditions must be unique"):
        SimulatedExperimentSpec.model_validate(payload)

    payload = default.model_dump()
    payload["profile_ids"] = ["synthetic-concise", "synthetic-concise"]
    with pytest.raises(ValidationError, match="profile IDs must be unique"):
        SimulatedExperimentSpec.model_validate(payload)

    payload = default.model_dump()
    payload["evaluation_time"] = "2026-07-18T12:00:00"
    with pytest.raises(ValidationError, match="must include a timezone"):
        SimulatedExperimentSpec.model_validate(payload)

    payload = default.model_dump()
    payload["simulator"] = {**default.simulator.model_dump(), "seed": default.seed + 1}
    with pytest.raises(ValidationError, match="seed must equal"):
        SimulatedExperimentSpec.model_validate(payload)


def test_paired_trials_are_deterministic_held_out_and_leak_free() -> None:
    conditions = (
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
        EvaluationCondition.CURRENT_SAFE_FUSION,
        EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
    )
    spec = make_spec(*conditions, profile_ids=("synthetic-casual", "synthetic-concise"))
    first = run_spec(spec)
    second = run_spec(spec)

    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert len({record.trial_id for record in first.trial_records}) == len(first.trial_records)
    assert all(record.target_available for record in first.trial_records)
    assert all(record.automatic_selection_permitted is False for record in first.trial_records)
    assert all(record.unintended_word is False for record in first.trial_records)

    test_messages = {message.message_id: message for message in BENCHMARK.messages[spec.split]}
    non_test_ids = {
        message.message_id
        for split, messages in BENCHMARK.messages.items()
        if split is not spec.split
        for message in messages
    }
    assert not {record.message_id for record in first.trial_records}.intersection(non_test_ids)
    for record in first.trial_records:
        message = test_messages[record.message_id]
        expected_context = " ".join(message.target_spans[: record.span_index])
        assert record.confirmed_context == expected_context
        assert record.retrieval_query_context_removed is (
            record.condition is EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT
        )


def test_language_conflict_keeps_target_visible_and_neural_evidence_decisive() -> None:
    spec = make_spec(
        EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
        EvaluationCondition.C_NEURAL_LANGUAGE,
        EvaluationCondition.CURRENT_SAFE_FUSION,
        conflict_every=1,
    )
    result = run_spec(spec)
    by_condition = {
        condition: tuple(record for record in result.trial_records if record.condition is condition)
        for condition in spec.conditions
    }

    assert all(record.language_conflict_context for record in result.trial_records)
    assert all(
        record.target_candidate_id in record.candidate_ids for record in result.trial_records
    )
    assert not any(
        record.top_1_correct for record in by_condition[EvaluationCondition.B_GENERIC_LANGUAGE_ONLY]
    )
    assert all(
        record.top_1_correct for record in by_condition[EvaluationCondition.C_NEURAL_LANGUAGE]
    )
    safe = by_condition[EvaluationCondition.CURRENT_SAFE_FUSION]
    assert all(record.top_1_correct for record in safe)
    assert all(record.explicit_selection_completed for record in safe)
    assert all(record.neural_language_conflict for record in safe)


def test_neural_ablations_trigger_safeguards_and_degrade_paired_top_choice() -> None:
    spec = make_spec(
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.ABLATION_UNIFORM_NEURAL,
        EvaluationCondition.ABLATION_SHUFFLED_NEURAL,
    )
    result = run_spec(spec)
    by_condition = {
        condition: tuple(record for record in result.trial_records if record.condition is condition)
        for condition in spec.conditions
    }

    assert all(record.top_1_correct for record in by_condition[EvaluationCondition.A_BCI_ONLY])
    uniform = by_condition[EvaluationCondition.ABLATION_UNIFORM_NEURAL]
    assert all(record.disposition is RankingDisposition.REQUEST_REPEAT for record in uniform)
    assert not any(record.explicit_selection_completed for record in uniform)
    shuffled = by_condition[EvaluationCondition.ABLATION_SHUFFLED_NEURAL]
    assert not any(record.top_1_correct for record in shuffled)
    assert not any(record.explicit_selection_completed for record in shuffled)


def test_retrieval_ablations_are_explicit_and_profile_scoped() -> None:
    spec = make_spec(
        EvaluationCondition.CURRENT_SAFE_FUSION,
        EvaluationCondition.ABLATION_REMOVE_RAG,
        EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL,
        EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL,
        EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT,
    )
    result = run_spec(spec)
    by_condition = {
        condition: tuple(record for record in result.trial_records if record.condition is condition)
        for condition in spec.conditions
    }

    assert all(
        record.retrieval_hit_count == 0
        for record in by_condition[EvaluationCondition.ABLATION_REMOVE_RAG]
    )
    assert all(
        record.retrieval_hit_count == 1
        for record in by_condition[EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL]
    )
    assert all(
        record.retrieval_query_context_removed
        for record in by_condition[EvaluationCondition.ABLATION_REMOVE_RETRIEVAL_CONTEXT]
    )
    assert all(metric.unintended_word_rate == 0 for metric in result.metrics)
    assert all(metric.automatic_selection_violation_count == 0 for metric in result.metrics)


def test_metrics_include_overall_profile_speed_calibration_and_conflict_slices() -> None:
    spec = make_spec(
        EvaluationCondition.A_BCI_ONLY,
        EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
        conflict_every=2,
    )
    result = run_spec(spec)

    assert len(result.metrics) == len(spec.conditions) * 2
    overall = [metric for metric in result.metrics if metric.profile_id is None]
    assert all(metric.correct_selections_per_minute > 0 for metric in overall)
    assert all(metric.words_per_minute > 0 for metric in overall)
    assert all(metric.conflict_trial_count > 0 for metric in overall)
    bci = next(metric for metric in overall if metric.condition is EvaluationCondition.A_BCI_ONLY)
    language = next(
        metric
        for metric in overall
        if metric.condition is EvaluationCondition.B_GENERIC_LANGUAGE_ONLY
    )
    assert bci.neural_expected_calibration_error is not None
    assert bci.neural_multiclass_brier_score is not None
    assert language.neural_expected_calibration_error is None
    assert language.neural_multiclass_brier_score is None
    assert language.conflict_top_1_recall == 0


def test_artifacts_are_byte_stable_and_manifest_checksums_match(tmp_path: Path) -> None:
    result = run_spec(
        make_spec(
            EvaluationCondition.A_BCI_ONLY,
            EvaluationCondition.CURRENT_SAFE_FUSION,
        )
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_manifest = write_experiment_artifacts(result, first_dir, git_sha="b683808")
    second_manifest = write_experiment_artifacts(result, second_dir, git_sha="b683808")

    for filename in ("trials.jsonl", "metrics.json", "manifest.json"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest.package_versions["python"]
    assert first_manifest.package_versions["neuroselect-bci"] == "0.1.0.dev0"
    assert first_manifest.device["system"]
    assert first_manifest.device["machine"]
    assert {model.artifact_id for model in first_manifest.models} == {
        "seeded-neural-simulator",
        "controlled-candidate-protocol",
        "candidate-risk-policy",
        "transparent-ranking-policy",
    }
    assert len((first_dir / "trials.jsonl").read_text().splitlines()) == len(result.trial_records)
    summary = json.loads((first_dir / "metrics.json").read_text())
    assert "trial_records" not in summary
    assert summary["trial_record_count"] == len(result.trial_records)
    output_checksums = {artifact.uri: artifact.sha256 for artifact in first_manifest.outputs}
    assert (
        output_checksums["artifact://trials.jsonl"]
        == hashlib.sha256((first_dir / "trials.jsonl").read_bytes()).hexdigest()
    )
    assert (
        output_checksums["artifact://metrics.json"]
        == hashlib.sha256((first_dir / "metrics.json").read_bytes()).hexdigest()
    )


def test_runner_rejects_unknown_profiles() -> None:
    spec = make_spec(
        EvaluationCondition.A_BCI_ONLY,
        profile_ids=("synthetic-unknown",),
    )
    with pytest.raises(ExperimentConfigurationError, match="unknown synthetic profiles"):
        run_spec(spec)
