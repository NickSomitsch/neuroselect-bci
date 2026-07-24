from __future__ import annotations

import importlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from neuroselect.language import (
    CandidateGenerationError,
    CandidateGenerationRequest,
    CandidateGenerator,
    ControlledStylePersonalizer,
    FixtureCandidateBackend,
    LocalModelCandidateBackend,
    LocalModelDependencyError,
    MlxAdapterPersonalizer,
    MlxLanguageRuntime,
    PersonalizationAdapterManifest,
    PersonalizedGenerationResult,
    PersonalizedLanguagePipeline,
    build_mlx_lora_command,
    finalize_personalization_adapter,
    load_local_model_config,
    load_lora_training_config,
    load_personalization_adapter,
    load_personalization_corpus_manifest,
    normalize_log_scores,
    personalization_prompt,
    write_personalization_corpus,
)
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import generate_from_sources, load_profiles

ROOT = Path(__file__).parents[2]
MODEL_CONFIG = ROOT / "configs/models/qwen3_4b_mlx.yaml"
LORA_CONFIG = ROOT / "configs/models/qwen3_4b_lora.yaml"
PROFILES = load_profiles(ROOT / "synthetic_data/profiles")
CONCISE = next(profile for profile in PROFILES if profile.profile_id == "synthetic-concise")
NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)


class FakeRuntime:
    def __init__(
        self,
        *,
        response: str | None = None,
        scores: tuple[float, ...] = (-3.0, -1.0, -2.0, -4.0),
    ) -> None:
        self.response = response or json.dumps(
            {
                "candidates": [
                    {"text": "alpha", "support": 1.0},
                    {"text": "beta", "support": 0.0},
                    {"text": "gamma ray", "support": 0.0},
                    {"text": "delta", "support": 0.0},
                ]
            }
        )
        self.scores = scores
        self.generated_messages: tuple[dict[str, str], ...] = ()
        self.scored_messages: tuple[dict[str, str], ...] = ()

    def generate(self, messages: tuple[dict[str, str], ...], *, max_tokens: int) -> str:
        assert max_tokens == 512
        self.generated_messages = messages
        return self.response

    def score_continuations(
        self,
        messages: tuple[dict[str, str], ...],
        continuations: tuple[str, ...],
    ) -> tuple[float, ...]:
        self.scored_messages = messages
        assert continuations
        return self.scores[: len(continuations)]


def test_local_model_config_is_pinned_and_strict(tmp_path: Path) -> None:
    config = load_local_model_config(MODEL_CONFIG)

    assert config.backend == "mlx-lm"
    assert config.model_revision == "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495"
    assert config.download_enabled is False
    assert config.metadata.model_id == "Qwen/Qwen3-4B-MLX-4bit"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("model_id: unpinned\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_local_model_config(invalid)
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_local_model_config(invalid)


def test_local_backend_replaces_self_reported_support_with_likelihoods() -> None:
    runtime = FakeRuntime()
    backend = LocalModelCandidateBackend(load_local_model_config(MODEL_CONFIG), runtime=runtime)
    request = CandidateGenerationRequest(confirmed_text="Hello", candidate_count=6)
    result = CandidateGenerator(backend).generate(request)
    support_by_text = {
        candidate.text: result.generic_language_support[candidate.candidate_id]
        for candidate in result.candidate_set.candidates[:-3]
    }

    expected = normalize_log_scores((-3.0, -1.0, -2.0, -4.0))
    selected_total = sum(expected[:3])
    assert support_by_text == pytest.approx(
        {
            "beta": expected[1] / selected_total,
            "gamma ray": expected[2] / selected_total,
            "alpha": expected[0] / selected_total,
        }
    )
    assert "placeholders" in runtime.generated_messages[1]["content"]
    assert "Next phrase:" in runtime.scored_messages[0]["content"]
    assert result.backend.deterministic is False


def test_local_backend_fails_on_bad_runtime_scores_and_output() -> None:
    config = load_local_model_config(MODEL_CONFIG)
    request = CandidateGenerationRequest(candidate_count=4)
    bad_count = FakeRuntime(scores=())
    with pytest.raises(CandidateGenerationError, match="different number"):
        LocalModelCandidateBackend(config, runtime=bad_count).generate(request)

    bad_json = FakeRuntime(response="not-json")
    with pytest.raises(CandidateGenerationError, match="invalid structured"):
        LocalModelCandidateBackend(config, runtime=bad_json).generate(request)

    with pytest.raises(ValueError, match="at least one"):
        normalize_log_scores(())
    with pytest.raises(ValueError, match="positive"):
        normalize_log_scores((1.0,), temperature=0.0)
    with pytest.raises(ValueError, match="finite"):
        normalize_log_scores((math.inf,))


class FakeTokenizer:
    def apply_chat_template(self, *_: Any, **__: Any) -> str:
        return "rendered-prompt"

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if value == "rendered-prompt":
            return [1, 2]
        return [3, 4] if value.startswith(" ") else []


class FakeModel:
    def __call__(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.zeros((1, inputs.shape[1], 8), dtype=float)


class FakeMx:
    array = staticmethod(np.asarray)
    take_along_axis = staticmethod(np.take_along_axis)
    mean = staticmethod(np.mean)

    @staticmethod
    def logsumexp(
        values: np.ndarray[Any, Any], *, axis: int, keepdims: bool
    ) -> np.ndarray[Any, Any]:
        maximum = np.max(values, axis=axis, keepdims=True)
        result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
        return result if keepdims else np.squeeze(result, axis=axis)

    @staticmethod
    def eval(value: Any) -> None:
        del value


def test_mlx_runtime_lazy_generation_and_token_likelihood_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_local_model_config(MODEL_CONFIG)
    loaded: dict[str, Any] = {}

    def fake_load(path: str, *, adapter_path: str | None, revision: str | None) -> Any:
        loaded.update(path=path, adapter_path=adapter_path, revision=revision)
        return FakeModel(), FakeTokenizer()

    fake_modules = {
        "mlx_lm": SimpleNamespace(
            load=fake_load,
            generate=lambda *_args, **_kwargs: '{"candidates":[]}',
        ),
        "mlx.core": FakeMx,
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=lambda **kwargs: ("sampler", kwargs)),
        "huggingface_hub": SimpleNamespace(snapshot_download=lambda **_kwargs: "/cached/model"),
    }
    original_import = importlib.import_module
    monkeypatch.setattr(
        "neuroselect.language.local_models.importlib.import_module",
        lambda name: fake_modules.get(name) or original_import(name),
    )
    monkeypatch.setattr("neuroselect.language.local_models.platform.system", lambda: "Darwin")
    monkeypatch.setattr("neuroselect.language.local_models.platform.machine", lambda: "arm64")

    runtime = MlxLanguageRuntime(config, adapter_path="/adapter")
    generated = runtime.generate(({"role": "user", "content": "test"},), max_tokens=10)
    scores = runtime.score_continuations(({"role": "user", "content": "test"},), ("alpha", "beta"))

    assert generated == '{"candidates":[]}'
    assert scores == pytest.approx((-math.log(8), -math.log(8)))
    assert loaded == {
        "path": "/cached/model",
        "adapter_path": "/adapter",
        "revision": None,
    }


def test_mlx_runtime_rejects_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("neuroselect.language.local_models.platform.system", lambda: "Linux")
    with pytest.raises(LocalModelDependencyError, match="Apple silicon"):
        MlxLanguageRuntime(load_local_model_config(MODEL_CONFIG)).generate(
            ({"role": "user", "content": "test"},), max_tokens=1
        )


def _add_profile_knowledge(store: SQLiteKnowledgeStore) -> None:
    for record in CONCISE.knowledge:
        store.add(
            profile_id=CONCISE.profile_id,
            record=KnowledgeRecordInput.model_validate(record.model_dump()),
            at_time=NOW,
        )


def test_controlled_personalization_and_rag_are_candidate_aligned() -> None:
    with SQLiteKnowledgeStore(":memory:") as store:
        _add_profile_knowledge(store)
        result = PersonalizedLanguagePipeline(
            CandidateGenerator(FixtureCandidateBackend()),
            ControlledStylePersonalizer(CONCISE),
            LexicalRetriever(store),
        ).generate(
            CandidateGenerationRequest(confirmed_text="I would like", candidate_count=8),
            profile_id=CONCISE.profile_id,
            at_time=NOW,
        )

    language_ids = set(result.generation.generic_language_support)
    assert set(result.personalization_support) == language_ids
    assert set(result.personalization_lift) == language_ids
    assert set(result.retrieval_support) == language_ids
    assert sum(result.personalization_lift.values()) == pytest.approx(0.0)
    assert any(value > 0.0 for value in result.retrieval_support.values())
    assert result.personalization.evidence_kind == "controlled_fixture"
    assert result.claim_eligible is False

    payload = result.model_dump()
    payload["claim_eligible"] = True
    with pytest.raises(ValidationError, match="claim eligibility"):
        PersonalizedGenerationResult.model_validate(payload)
    with pytest.raises(ValueError, match="does not match"):
        PersonalizedLanguagePipeline(
            CandidateGenerator(FixtureCandidateBackend()),
            ControlledStylePersonalizer(CONCISE),
            LexicalRetriever.__new__(LexicalRetriever),
        ).generate(
            CandidateGenerationRequest(),
            profile_id="synthetic-formal",
            at_time=NOW,
        )


def test_personalization_corpus_is_split_safe_and_checksum_verified(
    tmp_path: Path,
) -> None:
    benchmark = generate_from_sources(
        ROOT / "synthetic_data/benchmark.yaml",
        ROOT / "synthetic_data/profiles",
    )
    manifest = write_personalization_corpus(benchmark, CONCISE, tmp_path)
    directory = tmp_path / CONCISE.profile_id
    restored = load_personalization_corpus_manifest(directory)

    assert restored == manifest
    assert {artifact.path for artifact in manifest.artifacts} == {
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    }
    assert all(
        artifact.example_count > artifact.source_message_count for artifact in manifest.artifacts
    )
    first = json.loads((directory / "train.jsonl").read_text().splitlines()[0])
    assert first["prompt"] == personalization_prompt("")
    assert first["completion"].startswith(" ")
    assert len(manifest.digest()) == 64

    with (directory / "test.jsonl").open("a", encoding="utf-8") as test_file:
        test_file.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_personalization_corpus_manifest(directory)


def test_lora_command_adapter_manifest_and_verified_loading(tmp_path: Path) -> None:
    benchmark = generate_from_sources(
        ROOT / "synthetic_data/benchmark.yaml",
        ROOT / "synthetic_data/profiles",
    )
    corpus = write_personalization_corpus(benchmark, CONCISE, tmp_path / "corpus")
    model_config = load_local_model_config(MODEL_CONFIG)
    training_config = load_lora_training_config(LORA_CONFIG)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"controlled-test-weights")

    command = build_mlx_lora_command(
        model_source="/cached/model",
        corpus_dir=tmp_path / "corpus" / CONCISE.profile_id,
        adapter_dir=adapter_dir,
        config=training_config,
        python_executable="/python",
    )
    assert command[:5] == (
        "/python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
    )
    assert "--mask-prompt" in command
    assert "--grad-checkpoint" in command
    assert "--test" in command

    manifest = finalize_personalization_adapter(
        adapter_dir=adapter_dir,
        corpus_manifest=corpus,
        model_config=model_config,
        training_config=training_config,
        mlx_lm_version="0.31.3",
        trained_at=NOW,
    )
    bundle = load_personalization_adapter(
        adapter_dir,
        expected_profile_id=CONCISE.profile_id,
        expected_model_id=model_config.model_id,
        expected_model_revision=model_config.model_revision,
    )
    assert bundle.manifest == manifest
    assert manifest.validation_evaluated is True
    assert manifest.test_evaluated is True
    assert manifest.training_config_sha256 == training_config.digest()

    fake_backend = LocalModelCandidateBackend(model_config, runtime=FakeRuntime())
    personalizer = MlxAdapterPersonalizer(fake_backend, bundle)
    generated = CandidateGenerator(FixtureCandidateBackend()).generate(
        CandidateGenerationRequest(candidate_count=4)
    )
    support = personalizer.score(
        CandidateGenerationRequest(candidate_count=4),
        generated.candidate_set.candidates,
    )
    assert sum(support.values()) == pytest.approx(1.0)
    assert personalizer.provenance.evidence_kind == "held_out_adapter"

    (adapter_dir / "adapters.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_personalization_adapter(adapter_dir)


def test_adapter_manifest_and_training_config_validation(tmp_path: Path) -> None:
    training = load_lora_training_config(LORA_CONFIG)
    assert training.mask_prompt is True
    assert training.evaluate_test is True

    non_mapping = tmp_path / "lora.yaml"
    non_mapping.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_lora_training_config(non_mapping)

    payload = {
        "schema_version": "1.0",
        "adapter_id": "adapter",
        "profile_id": CONCISE.profile_id,
        "base_model_id": "model",
        "base_model_revision": "a" * 40,
        "adapter_file": "../weights",
        "adapter_sha256": "b" * 64,
        "source_corpus_manifest_sha256": "c" * 64,
        "training_config_sha256": "d" * 64,
        "trainer_revision": "trainer",
        "mlx_lm_version": "0.31.3",
        "trained_at": NOW.isoformat(),
        "validation_evaluated": True,
        "test_evaluated": True,
        "synthetic_data": True,
    }
    with pytest.raises(ValidationError):
        PersonalizationAdapterManifest.model_validate(payload)
