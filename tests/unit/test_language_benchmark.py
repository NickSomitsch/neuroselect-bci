from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.core.models import Candidate
from neuroselect.evaluation import (
    HeldOutLanguageBenchmarkRunner,
    HeldOutLanguageSpec,
    LanguageProfileRuntime,
    build_held_out_candidate_vocabulary,
    load_held_out_language_spec,
    read_held_out_language_artifacts,
    write_held_out_language_artifacts,
)
from neuroselect.language import (
    BackendMetadata,
    CandidateGenerationError,
    CandidateGenerationRequest,
    CandidateGenerator,
    CandidateProposal,
    PersonalizationAdapterBundle,
    PersonalizationAdapterManifest,
    PersonalizationCorpusArtifact,
    PersonalizationCorpusManifest,
    PersonalizationProvenance,
    PersonalizedLanguagePipeline,
)
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import (
    BenchmarkMessage,
    BenchmarkSplit,
    GeneratedBenchmark,
    load_profiles,
)

ROOT = Path(__file__).parents[2]
PROFILE = next(
    profile
    for profile in load_profiles(ROOT / "synthetic_data/profiles")
    if profile.profile_id == "synthetic-concise"
)
NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


class ContextBackend:
    metadata = BackendMetadata(
        backend_id="test-language",
        model_id="test/qwen",
        model_revision="b" * 40,
        generator_revision="test-natural-generation-v1",
        prompt_revision="test-prompt-v1",
        deterministic=True,
    )

    def __init__(self, *, fail_after_alpha: bool = False) -> None:
        self.fail_after_alpha = fail_after_alpha

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]:
        if request.confirmed_text == "alpha":
            if self.fail_after_alpha:
                raise CandidateGenerationError("controlled generation failure")
            values = (("omega", 0.6), ("theta", 0.3), ("kappa", 0.1))
        else:
            values = (("gamma", 0.7), ("alpha", 0.2), ("delta", 0.1))
        return tuple(CandidateProposal(text=text, support=support) for text, support in values)


class StaticAdapterPersonalizer:
    def __init__(self, manifest: PersonalizationAdapterManifest) -> None:
        self.provenance = PersonalizationProvenance(
            profile_id=manifest.profile_id,
            evidence_kind="held_out_adapter",
            adapter_id=manifest.adapter_id,
            adapter_sha256=manifest.adapter_sha256,
            base_model_id=manifest.base_model_id,
            base_model_revision=manifest.base_model_revision,
            source_corpus_manifest_sha256=manifest.source_corpus_manifest_sha256,
            training_config_sha256=manifest.training_config_sha256,
            validation_evaluated=True,
            test_evaluated=True,
        )

    def score(
        self,
        request: CandidateGenerationRequest,
        candidates: tuple[Candidate, ...],
    ) -> dict[str, float]:
        del request
        weights = {"alpha": 0.8, "gamma": 0.15, "delta": 0.05}
        raw = {
            candidate.candidate_id: weights.get(candidate.text, 1.0)
            for candidate in candidates
            if candidate.kind.value != "control"
        }
        total = sum(raw.values())
        return {candidate_id: value / total for candidate_id, value in raw.items()}


def benchmark() -> GeneratedBenchmark:
    messages = (
        BenchmarkMessage(
            message_id="msg-" + "1" * 20,
            profile_id=PROFILE.profile_id,
            split=BenchmarkSplit.TEST,
            template_id="test-one",
            topic="test",
            text="alpha beta",
            target_spans=("alpha", "beta"),
        ),
        BenchmarkMessage(
            message_id="msg-" + "2" * 20,
            profile_id=PROFILE.profile_id,
            split=BenchmarkSplit.TEST,
            template_id="test-two",
            topic="test",
            text="gamma",
            target_spans=("gamma",),
        ),
    )
    return GeneratedBenchmark(
        schema_version="1.0",
        source_sha256="a" * 64,
        profile_ids=(PROFILE.profile_id,),
        messages={
            BenchmarkSplit.TRAIN: (),
            BenchmarkSplit.VALIDATION: (),
            BenchmarkSplit.TEST: messages,
        },
    )


def vocabulary_benchmark() -> GeneratedBenchmark:
    messages: dict[BenchmarkSplit, tuple[BenchmarkMessage, ...]] = {
        BenchmarkSplit.TRAIN: (
            BenchmarkMessage(
                message_id="msg-" + "3" * 20,
                profile_id=PROFILE.profile_id,
                split=BenchmarkSplit.TRAIN,
                template_id="train-request",
                topic="test",
                text="Could you close the desk lamp when ready.",
                target_spans=("Could you close", "the desk lamp", "when ready."),
            ),
        ),
        BenchmarkSplit.VALIDATION: (
            BenchmarkMessage(
                message_id="msg-" + "4" * 20,
                profile_id=PROFILE.profile_id,
                split=BenchmarkSplit.VALIDATION,
                template_id="validation-time",
                topic="test",
                text="Before early afternoon Please close the desk lamp when ready.",
                target_spans=(
                    "Before",
                    "early afternoon",
                    "Please close",
                    "the desk lamp",
                    "when ready.",
                ),
            ),
        ),
        BenchmarkSplit.TEST: (
            BenchmarkMessage(
                message_id="msg-" + "5" * 20,
                profile_id=PROFILE.profile_id,
                split=BenchmarkSplit.TEST,
                template_id="test-secret",
                topic="test",
                text="Secret test phrase",
                target_spans=("Secret test phrase",),
            ),
        ),
    }
    return GeneratedBenchmark(
        schema_version="1.0",
        source_sha256="9" * 64,
        profile_ids=(PROFILE.profile_id,),
        messages=messages,
    )


def corpus_manifest() -> PersonalizationCorpusManifest:
    return PersonalizationCorpusManifest(
        schema_version="1.0",
        profile_id=PROFILE.profile_id,
        source_benchmark_sha256="a" * 64,
        profile_style_sha256="c" * 64,
        prompt_revision="personal-next-span-completion-v1",
        artifacts=tuple(
            PersonalizationCorpusArtifact(
                split=split,
                path=path,
                source_message_count=1,
                example_count=1,
                sha256=str(index) * 64,
            )
            for index, (split, path) in enumerate(
                (
                    (BenchmarkSplit.TRAIN, "train.jsonl"),
                    (BenchmarkSplit.VALIDATION, "valid.jsonl"),
                    (BenchmarkSplit.TEST, "test.jsonl"),
                ),
                start=1,
            )
        ),
    )


def adapter_manifest(
    corpus: PersonalizationCorpusManifest,
    *,
    trainer_revision: str = "neuroselect-mlx-lora-dev-v1",
) -> PersonalizationAdapterManifest:
    return PersonalizationAdapterManifest(
        schema_version="1.0",
        adapter_id="lora-synthetic-concise-test",
        profile_id=PROFILE.profile_id,
        base_model_id="test/qwen",
        base_model_revision="b" * 40,
        adapter_file="adapters.safetensors",
        adapter_sha256="d" * 64,
        source_corpus_manifest_sha256=corpus.digest(),
        training_config_sha256="e" * 64,
        trainer_revision=trainer_revision,
        mlx_lm_version="0.31.3",
        trained_at=NOW,
        validation_evaluated=True,
        test_evaluated=True,
    )


def build_runtime(
    store: SQLiteKnowledgeStore,
    *,
    fail_after_alpha: bool = False,
    trainer_revision: str = "neuroselect-mlx-lora-dev-v1",
) -> LanguageProfileRuntime:
    corpus = corpus_manifest()
    manifest = adapter_manifest(corpus, trainer_revision=trainer_revision)
    pipeline = PersonalizedLanguagePipeline(
        CandidateGenerator(ContextBackend(fail_after_alpha=fail_after_alpha)),
        StaticAdapterPersonalizer(manifest),
        LexicalRetriever(store),
    )
    return LanguageProfileRuntime(
        profile=PROFILE,
        adapter=PersonalizationAdapterBundle(Path("/adapter"), manifest),
        corpus_manifest=corpus,
        pipeline_factory=lambda: pipeline,
    )


def spec(**updates: object) -> HeldOutLanguageSpec:
    payload = {
        "schema_version": "1.0",
        "experiment_id": "test-held-out-language",
        "protocol_revision": "held-out-language-personalization-v1",
        "seed": 7,
        "split": "test",
        "candidate_count": 6,
        "maximum_phrase_tokens": 4,
        "retrieval_at": NOW,
        "evidence_tier": "development",
        "maximum_messages_per_profile": None,
        **updates,
    }
    return HeldOutLanguageSpec.model_validate(payload)


def test_spec_loading_is_strict_and_requires_timezone(tmp_path: Path) -> None:
    loaded = load_held_out_language_spec(
        ROOT / "configs/experiments/held_out_language_personalization.yaml"
    )
    assert loaded.split is BenchmarkSplit.TEST
    assert loaded.candidate_count == 12
    assert loaded.maximum_messages_per_profile == 1
    assert loaded.evidence_tier == "development"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_held_out_language_spec(invalid)
    with pytest.raises(ValidationError, match="timezone"):
        spec(retrieval_at=datetime(2026, 7, 18))


def test_candidate_vocabulary_uses_only_train_and_validation_messages() -> None:
    vocabulary = build_held_out_candidate_vocabulary(vocabulary_benchmark())

    assert vocabulary.noun_phrases == ("the desk lamp",)
    assert vocabulary.deadline_phrases == ("Before early afternoon",)
    assert vocabulary.ending_phrases == ("when ready.",)
    assert vocabulary.phrases_for("") == ()
    assert vocabulary.phrases_for("Could you close") == ("the desk lamp",)
    assert vocabulary.phrases_for("Could you close the desk lamp") == ("Before early afternoon",)
    assert vocabulary.phrases_for("Could you close the desk lamp before early afternoon") == (
        "when ready.",
    )
    assert "Secret test phrase" not in vocabulary.model_dump_json()
    assert len(vocabulary.digest()) == 64


def test_runner_records_natural_target_absence_and_ranking() -> None:
    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        cleaned = True

    with SQLiteKnowledgeStore(":memory:") as store:
        for record in PROFILE.knowledge:
            store.add(
                profile_id=PROFILE.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=NOW,
            )
        runtime = replace(build_runtime(store), cleanup=cleanup)
        result = HeldOutLanguageBenchmarkRunner(spec()).run(
            benchmark=benchmark(),
            runtimes=(runtime,),
            generated_at=NOW,
        )

    assert cleaned is True
    assert len(result.trials) == 3
    assert [trial.target_available for trial in result.trials] == [True, False, True]
    assert result.trials[0].generic_rank == 2
    assert result.trials[0].personalized_rank == 1
    assert result.trials[1].generic_rank is None
    overall = result.metrics[0]
    assert overall.profile_id is None
    assert overall.repaired_generation_rate == 0.0
    assert overall.target_availability_rate == pytest.approx(2 / 3)
    assert overall.message_target_availability_rate == pytest.approx(0.5)
    assert overall.generic_top_1_candidate_recall == pytest.approx(1 / 3)
    assert overall.personalized_top_1_candidate_recall == pytest.approx(1 / 3)
    assert overall.generic_message_exact_accuracy == pytest.approx(0.5)
    assert overall.personalized_message_exact_accuracy == 0.0
    assert overall.mean_personalized_rank_improvement_given_available == 0.0
    assert result.claim_eligible is False
    assert result.digest()


def test_runner_records_generation_failure_and_rejects_bad_provenance() -> None:
    with SQLiteKnowledgeStore(":memory:") as store:
        runtime = build_runtime(store, fail_after_alpha=True)
        result = HeldOutLanguageBenchmarkRunner(spec()).run(
            benchmark=benchmark(),
            runtimes=(runtime,),
            generated_at=NOW,
        )
        failure = result.trials[1]
        assert failure.candidate_generation_failed is True
        assert failure.failure_reason == "controlled generation failure"
        assert result.metrics[0].generation_success_rate == pytest.approx(2 / 3)

        wrong_corpus = runtime.corpus_manifest.model_copy(
            update={"source_benchmark_sha256": "f" * 64}
        )
        with pytest.raises(ValueError, match="different benchmark"):
            HeldOutLanguageBenchmarkRunner(spec()).run(
                benchmark=benchmark(),
                runtimes=(replace(runtime, corpus_manifest=wrong_corpus),),
            )
        with pytest.raises(ValueError, match="cover every benchmark profile"):
            HeldOutLanguageBenchmarkRunner(spec()).run(
                benchmark=benchmark(),
                runtimes=(),
            )


def test_runner_resumes_only_exact_protocol_trials() -> None:
    with SQLiteKnowledgeStore(":memory:") as store:
        runtime = build_runtime(store)
        runner = HeldOutLanguageBenchmarkRunner(spec())
        original = runner.run(
            benchmark=benchmark(),
            runtimes=(runtime,),
            generated_at=NOW,
        )
        callbacks: list[tuple[str, int, int]] = []
        resumed = runner.run(
            benchmark=benchmark(),
            runtimes=(runtime,),
            generated_at=NOW,
            resumed_trials=original.trials[:2],
            progress_callback=lambda trial, completed, total: callbacks.append(
                (trial.trial_id, completed, total)
            ),
        )

        assert resumed == original
        assert callbacks == [(original.trials[2].trial_id, 3, 3)]

        mismatched = original.trials[0].model_copy(update={"confirmed_context": "wrong"})
        with pytest.raises(ValueError, match="does not match"):
            runner.run(
                benchmark=benchmark(),
                runtimes=(runtime,),
                resumed_trials=(mismatched,),
            )
        with pytest.raises(ValueError, match="duplicate trial IDs"):
            runner.run(
                benchmark=benchmark(),
                runtimes=(runtime,),
                resumed_trials=(original.trials[0], original.trials[0]),
            )


def test_full_research_configuration_controls_claim_eligibility() -> None:
    with SQLiteKnowledgeStore(":memory:") as store:
        runtime = build_runtime(store, trainer_revision="neuroselect-mlx-lora-v1")
        result = HeldOutLanguageBenchmarkRunner(spec(evidence_tier="research")).run(
            benchmark=benchmark(),
            runtimes=(runtime,),
            generated_at=NOW,
        )
    assert result.claim_eligible is True
    payload = result.model_dump()
    payload["claim_eligible"] = False
    with pytest.raises(ValidationError, match="claim eligibility"):
        type(result).model_validate(payload)


def test_language_artifact_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    with SQLiteKnowledgeStore(":memory:") as store:
        result = HeldOutLanguageBenchmarkRunner(spec()).run(
            benchmark=benchmark(),
            runtimes=(build_runtime(store),),
            generated_at=NOW,
            candidate_vocabulary_sha256="8" * 64,
        )
    manifest = write_held_out_language_artifacts(
        result,
        tmp_path,
        git_sha="a" * 40,
        source_tree_sha256="b" * 64,
        package_versions={"python": "3.12", "neuroselect-bci": "test"},
        device={"system": "test"},
    )
    restored, restored_manifest = read_held_out_language_artifacts(tmp_path)
    assert restored == result
    assert restored_manifest == manifest
    assert manifest.metadata["claim_eligible"] is False
    assert manifest.metadata["working_tree_dirty"] is True
    assert result.candidate_vocabulary_sha256 == "8" * 64
    assert any(
        item.artifact_id == "non-test-candidate-vocabulary" and item.sha256 == "8" * 64
        for item in manifest.datasets
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_held_out_language_artifacts(
            result,
            tmp_path,
            git_sha="a" * 40,
            package_versions={"python": "3.12"},
            device={"system": "test"},
        )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["datasets"] = [
        item
        for item in manifest_payload["datasets"]
        if item["artifact_id"] != "non-test-candidate-vocabulary"
    ]
    (tmp_path / "manifest.json").write_text(
        type(manifest).model_validate(manifest_payload).canonical_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest does not agree"):
        read_held_out_language_artifacts(tmp_path)
    (tmp_path / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_held_out_language_artifacts(tmp_path)
