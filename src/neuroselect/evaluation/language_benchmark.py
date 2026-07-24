"""Held-out natural-candidate language and personalization evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import CandidateKind, CandidateSet
from neuroselect.language import (
    BackendMetadata,
    CandidateGenerationError,
    CandidateGenerationRequest,
    PersonalizationAdapterBundle,
    PersonalizationAdapterManifest,
    PersonalizationCorpusManifest,
    PersonalizedLanguagePipeline,
)
from neuroselect.retrieval import CandidateRetrievalEvidence
from neuroselect.synthetic import (
    BenchmarkMessage,
    BenchmarkSplit,
    GeneratedBenchmark,
    SyntheticProfile,
)

DEFAULT_LANGUAGE_BENCHMARK_CONFIG = Path(
    "configs/experiments/held_out_language_personalization.yaml"
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


class HeldOutLanguageSpec(BaseModel):
    """Locked protocol for natural next-span availability and ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    experiment_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal["held-out-language-personalization-v1"]
    seed: int = Field(default=20260724, ge=0)
    split: Literal[BenchmarkSplit.TEST] = BenchmarkSplit.TEST
    candidate_count: Literal[4, 6, 8, 12] = 8
    maximum_phrase_tokens: int = Field(default=4, ge=1, le=8)
    retrieval_at: datetime
    evidence_tier: Literal["development", "research"] = "development"
    maximum_messages_per_profile: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_aware_retrieval_time(self) -> HeldOutLanguageSpec:
        if self.retrieval_at.tzinfo is None or self.retrieval_at.utcoffset() is None:
            raise ValueError("retrieval_at must include a timezone")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


def load_held_out_language_spec(
    path: str | Path = DEFAULT_LANGUAGE_BENCHMARK_CONFIG,
) -> HeldOutLanguageSpec:
    """Load a strict held-out language evaluation recipe."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("held-out language configuration must contain a YAML mapping")
    return HeldOutLanguageSpec.model_validate(payload)


class LanguageBenchmarkTrial(BaseModel):
    """One natural next-span proposal round over a teacher-forced context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str = Field(min_length=1, max_length=160)
    profile_id: str = Field(pattern=r"^synthetic-[a-z0-9-]+$")
    message_id: str
    span_index: int = Field(ge=0)
    message_span_count: int = Field(ge=1)
    confirmed_context: str = Field(max_length=4_000)
    intended_text: str = Field(min_length=1, max_length=160)
    candidate_generation_failed: bool = False
    backend_output_repaired: bool = False
    failure_reason: str | None = Field(default=None, max_length=500)
    candidate_set: CandidateSet | None = None
    intended_candidate_id: str | None = None
    other_candidate_id: str | None = None
    generic_language_support: dict[str, float] = Field(default_factory=dict)
    personalization_support: dict[str, float] = Field(default_factory=dict)
    personalization_lift: dict[str, float] = Field(default_factory=dict)
    retrieval_evidence: tuple[CandidateRetrievalEvidence, ...] = ()
    generic_rank: int | None = Field(default=None, ge=1)
    personalized_rank: int | None = Field(default=None, ge=1)
    adapter_id: str = Field(min_length=1, max_length=160)
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_trial_alignment(self) -> LanguageBenchmarkTrial:
        if self.span_index >= self.message_span_count:
            raise ValueError("span_index must be smaller than message_span_count")
        if self.candidate_generation_failed:
            if self.failure_reason is None:
                raise ValueError("candidate generation failures require a reason")
            if (
                self.candidate_set is not None
                or self.intended_candidate_id is not None
                or self.other_candidate_id is not None
                or self.generic_language_support
                or self.personalization_support
                or self.personalization_lift
                or self.retrieval_evidence
                or self.generic_rank is not None
                or self.personalized_rank is not None
            ):
                raise ValueError("failed candidate rounds cannot contain ranking evidence")
            return self

        if self.failure_reason is not None or self.candidate_set is None:
            raise ValueError("successful candidate rounds require a candidate set and no failure")
        candidate_ids = tuple(item.candidate_id for item in self.candidate_set.candidates)
        language_ids = {
            item.candidate_id
            for item in self.candidate_set.candidates
            if item.kind is not CandidateKind.CONTROL
        }
        if self.other_candidate_id not in candidate_ids:
            raise ValueError("successful candidate rounds require a visible Other control")
        other = next(
            item
            for item in self.candidate_set.candidates
            if item.candidate_id == self.other_candidate_id
        )
        if (
            other.kind is not CandidateKind.CONTROL
            or _normalized_text(other.text).rstrip(".") != "other"
        ):
            raise ValueError("other_candidate_id must identify the Other control")
        for label, values in (
            ("generic support", self.generic_language_support),
            ("personalization support", self.personalization_support),
            ("personalization lift", self.personalization_lift),
        ):
            if set(values) != language_ids:
                raise ValueError(f"{label} must cover exactly the language candidates")
        if not math.isclose(
            sum(self.generic_language_support.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ) or not math.isclose(
            sum(self.personalization_support.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("generic and personalization support must each sum to one")
        expected_lift = {
            candidate_id: self.personalization_support[candidate_id]
            - self.generic_language_support[candidate_id]
            for candidate_id in language_ids
        }
        if any(
            not math.isclose(
                self.personalization_lift[candidate_id],
                expected_lift[candidate_id],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for candidate_id in language_ids
        ):
            raise ValueError("personalization lift must be personal minus generic support")
        retrieval_ids = {item.candidate_id for item in self.retrieval_evidence}
        if retrieval_ids != language_ids:
            raise ValueError("retrieval evidence must cover exactly the language candidates")

        if self.intended_candidate_id is None:
            if self.generic_rank is not None or self.personalized_rank is not None:
                raise ValueError("absent targets cannot have language ranks")
        else:
            if self.intended_candidate_id not in language_ids:
                raise ValueError("intended candidate must be a visible language candidate")
            intended = next(
                item
                for item in self.candidate_set.candidates
                if item.candidate_id == self.intended_candidate_id
            )
            if _normalized_text(intended.text) != _normalized_text(self.intended_text):
                raise ValueError("visible intended candidate text must match the target span")
            if self.generic_rank is None or self.personalized_rank is None:
                raise ValueError("available targets require both language ranks")
        return self

    @property
    def target_available(self) -> bool:
        return self.intended_candidate_id is not None


class LanguageBenchmarkMetrics(BaseModel):
    """Availability and generic/personalized ranking metrics for one scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str | None = None
    trial_count: int = Field(ge=1)
    message_count: int = Field(ge=1)
    generation_success_rate: float = Field(ge=0.0, le=1.0)
    repaired_generation_rate: float = Field(ge=0.0, le=1.0)
    target_availability_rate: float = Field(ge=0.0, le=1.0)
    message_target_availability_rate: float = Field(ge=0.0, le=1.0)
    generic_top_1_candidate_recall: float = Field(ge=0.0, le=1.0)
    generic_top_3_candidate_recall: float = Field(ge=0.0, le=1.0)
    generic_top_1_recall_given_available: float = Field(ge=0.0, le=1.0)
    generic_top_3_recall_given_available: float = Field(ge=0.0, le=1.0)
    generic_mrr_given_available: float = Field(ge=0.0, le=1.0)
    generic_message_exact_accuracy: float = Field(ge=0.0, le=1.0)
    personalized_top_1_candidate_recall: float = Field(ge=0.0, le=1.0)
    personalized_top_3_candidate_recall: float = Field(ge=0.0, le=1.0)
    personalized_top_1_recall_given_available: float = Field(ge=0.0, le=1.0)
    personalized_top_3_recall_given_available: float = Field(ge=0.0, le=1.0)
    personalized_mrr_given_available: float = Field(ge=0.0, le=1.0)
    personalized_message_exact_accuracy: float = Field(ge=0.0, le=1.0)
    mean_personalized_rank_improvement_given_available: float


class HeldOutLanguageResult(BaseModel):
    """Checksum-addressable held-out language evaluation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    run_id: str = Field(min_length=1, max_length=160)
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_vocabulary_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    spec: HeldOutLanguageSpec
    backend: BackendMetadata
    adapters: dict[str, PersonalizationAdapterManifest]
    corpus_manifest_sha256: dict[str, str]
    trials: tuple[LanguageBenchmarkTrial, ...] = Field(min_length=1)
    metrics: tuple[LanguageBenchmarkMetrics, ...] = Field(min_length=2)
    claim_eligible: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> HeldOutLanguageResult:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        if self.config_sha256 != self.spec.digest():
            raise ValueError("result config checksum must match its embedded specification")
        profile_ids = {trial.profile_id for trial in self.trials}
        if set(self.adapters) != profile_ids or set(self.corpus_manifest_sha256) != profile_ids:
            raise ValueError("adapter and corpus provenance must cover every evaluated profile")
        metric_profiles = {metric.profile_id for metric in self.metrics}
        if metric_profiles != {None, *profile_ids}:
            raise ValueError("metrics must contain overall and per-profile scopes")
        expected_claim = (
            self.spec.evidence_tier == "research"
            and self.spec.maximum_messages_per_profile is None
            and all(
                manifest.trainer_revision == "neuroselect-mlx-lora-v1"
                and manifest.validation_evaluated
                and manifest.test_evaluated
                for manifest in self.adapters.values()
            )
        )
        if self.claim_eligible != expected_claim:
            raise ValueError("claim eligibility must follow evidence tier, coverage, and adapters")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class LanguageProfileRuntime:
    """Verified local inputs and lazy pipeline factory for one synthetic profile."""

    profile: SyntheticProfile
    adapter: PersonalizationAdapterBundle
    corpus_manifest: PersonalizationCorpusManifest
    pipeline_factory: Callable[[], PersonalizedLanguagePipeline]
    cleanup: Callable[[], None] | None = None


def _rank(
    values: dict[str, float],
    *,
    target_id: str,
    candidate_order: tuple[str, ...],
) -> int:
    ranked = sorted(values, key=lambda item: (-values[item], candidate_order.index(item)))
    return ranked.index(target_id) + 1


def _divide(numerator: int | float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(
    records: tuple[LanguageBenchmarkTrial, ...],
    *,
    profile_id: str | None,
) -> LanguageBenchmarkMetrics:
    scoped = tuple(
        record for record in records if profile_id is None or record.profile_id == profile_id
    )
    available = tuple(record for record in scoped if record.target_available)
    messages: dict[tuple[str, str], list[LanguageBenchmarkTrial]] = defaultdict(list)
    for record in scoped:
        messages[(record.profile_id, record.message_id)].append(record)
    message_records = tuple(messages.values())

    def top_k(rank_name: Literal["generic_rank", "personalized_rank"], k: int) -> int:
        return sum(
            getattr(record, rank_name) is not None and getattr(record, rank_name) <= k
            for record in scoped
        )

    def message_exact(rank_name: Literal["generic_rank", "personalized_rank"]) -> int:
        return sum(
            len(group) == group[0].message_span_count
            and all(getattr(record, rank_name) == 1 for record in group)
            for group in message_records
        )

    generic_top_1 = top_k("generic_rank", 1)
    generic_top_3 = top_k("generic_rank", 3)
    personalized_top_1 = top_k("personalized_rank", 1)
    personalized_top_3 = top_k("personalized_rank", 3)
    available_count = len(available)
    rank_improvements = [
        record.generic_rank - record.personalized_rank
        for record in available
        if record.generic_rank is not None and record.personalized_rank is not None
    ]
    return LanguageBenchmarkMetrics(
        profile_id=profile_id,
        trial_count=len(scoped),
        message_count=len(message_records),
        generation_success_rate=_divide(
            sum(not record.candidate_generation_failed for record in scoped), len(scoped)
        ),
        repaired_generation_rate=_divide(
            sum(record.backend_output_repaired for record in scoped), len(scoped)
        ),
        target_availability_rate=_divide(available_count, len(scoped)),
        message_target_availability_rate=_divide(
            sum(
                len(group) == group[0].message_span_count
                and all(record.target_available for record in group)
                for group in message_records
            ),
            len(message_records),
        ),
        generic_top_1_candidate_recall=_divide(generic_top_1, len(scoped)),
        generic_top_3_candidate_recall=_divide(generic_top_3, len(scoped)),
        generic_top_1_recall_given_available=_divide(generic_top_1, available_count),
        generic_top_3_recall_given_available=_divide(generic_top_3, available_count),
        generic_mrr_given_available=_divide(
            sum(1.0 / record.generic_rank for record in available if record.generic_rank),
            available_count,
        ),
        generic_message_exact_accuracy=_divide(message_exact("generic_rank"), len(message_records)),
        personalized_top_1_candidate_recall=_divide(personalized_top_1, len(scoped)),
        personalized_top_3_candidate_recall=_divide(personalized_top_3, len(scoped)),
        personalized_top_1_recall_given_available=_divide(personalized_top_1, available_count),
        personalized_top_3_recall_given_available=_divide(personalized_top_3, available_count),
        personalized_mrr_given_available=_divide(
            sum(1.0 / record.personalized_rank for record in available if record.personalized_rank),
            available_count,
        ),
        personalized_message_exact_accuracy=_divide(
            message_exact("personalized_rank"), len(message_records)
        ),
        mean_personalized_rank_improvement_given_available=_divide(
            sum(rank_improvements), len(rank_improvements)
        ),
    )


class HeldOutLanguageBenchmarkRunner:
    """Evaluate natural candidate availability without injecting the target span."""

    def __init__(self, spec: HeldOutLanguageSpec) -> None:
        self.spec = spec

    def run(
        self,
        *,
        benchmark: GeneratedBenchmark,
        runtimes: tuple[LanguageProfileRuntime, ...],
        generated_at: datetime | None = None,
        candidate_vocabulary_sha256: str | None = None,
    ) -> HeldOutLanguageResult:
        runtime_by_profile = {runtime.profile.profile_id: runtime for runtime in runtimes}
        if len(runtime_by_profile) != len(runtimes) or set(runtime_by_profile) != set(
            benchmark.profile_ids
        ):
            raise ValueError("language runtimes must cover every benchmark profile exactly once")
        backend: BackendMetadata | None = None
        records: list[LanguageBenchmarkTrial] = []
        adapters: dict[str, PersonalizationAdapterManifest] = {}
        corpus_digests: dict[str, str] = {}

        for profile_id in sorted(runtime_by_profile):
            runtime = runtime_by_profile[profile_id]
            manifest = runtime.adapter.manifest
            corpus = runtime.corpus_manifest
            if manifest.profile_id != profile_id or corpus.profile_id != profile_id:
                raise ValueError("profile, adapter, and corpus identities must agree")
            if corpus.source_benchmark_sha256 != benchmark.source_sha256:
                raise ValueError("personalization corpus references a different benchmark")
            corpus_digest = corpus.digest()
            if manifest.source_corpus_manifest_sha256 != corpus_digest:
                raise ValueError("adapter was not trained from the supplied corpus manifest")
            adapters[profile_id] = manifest
            corpus_digests[profile_id] = corpus_digest

            pipeline = runtime.pipeline_factory()
            pipeline_backend = pipeline.generator.backend.metadata
            if backend is None:
                backend = pipeline_backend
            elif backend != pipeline_backend:
                raise ValueError("all profiles must use one generic language backend")
            provenance = pipeline.personalizer.provenance
            if (
                provenance.evidence_kind != "held_out_adapter"
                or provenance.profile_id != profile_id
                or provenance.adapter_id != manifest.adapter_id
                or provenance.adapter_sha256 != manifest.adapter_sha256
            ):
                raise ValueError("pipeline personalizer does not match its verified adapter")

            messages = tuple(
                message
                for message in benchmark.messages[self.spec.split]
                if message.profile_id == profile_id
            )
            messages = self._select_messages(messages)
            if not messages:
                raise ValueError(f"no held-out messages selected for {profile_id}")
            try:
                for message in messages:
                    records.extend(self._evaluate_message(message, pipeline, manifest))
            finally:
                del pipeline
                if runtime.cleanup is not None:
                    runtime.cleanup()

        assert backend is not None
        record_tuple = tuple(records)
        profile_metrics = tuple(
            _metrics(record_tuple, profile_id=profile_id) for profile_id in sorted(adapters)
        )
        identity = "\0".join(
            (
                self.spec.digest(),
                benchmark.source_sha256,
                _canonical_json(backend.model_dump(mode="json")),
                candidate_vocabulary_sha256 or "unconstrained",
                *(adapters[profile_id].digest() for profile_id in sorted(adapters)),
            )
        )
        claim_eligible = (
            self.spec.evidence_tier == "research"
            and self.spec.maximum_messages_per_profile is None
            and all(
                manifest.trainer_revision == "neuroselect-mlx-lora-v1"
                and manifest.validation_evaluated
                and manifest.test_evaluated
                for manifest in adapters.values()
            )
        )
        limitations = [
            "The intended spans are synthetic and teacher-forced context is used.",
            (
                "Language support is relative within each visible candidate set, "
                "not calibrated intent."
            ),
            (
                "Target absence is a candidate-generation failure for completion, "
                "not a successful Other."
            ),
            "Retrieval evidence is exported separately and is not folded into language ranks.",
        ]
        if candidate_vocabulary_sha256 is not None:
            limitations.append(
                "Candidate vocabulary constraints use only train and validation messages."
            )
        if not claim_eligible:
            limitations.append(
                "This limited development run is not eligible for personalization-benefit claims."
            )
        return HeldOutLanguageResult(
            schema_version="1.0",
            run_id=f"held-out-language-{_sha256_text(identity)[:20]}",
            generated_at=generated_at or datetime.now(UTC),
            config_sha256=self.spec.digest(),
            benchmark_source_sha256=benchmark.source_sha256,
            candidate_vocabulary_sha256=candidate_vocabulary_sha256,
            spec=self.spec,
            backend=backend,
            adapters=adapters,
            corpus_manifest_sha256=corpus_digests,
            trials=record_tuple,
            metrics=(_metrics(record_tuple, profile_id=None), *profile_metrics),
            claim_eligible=claim_eligible,
            limitations=tuple(limitations),
        )

    def _select_messages(
        self, messages: tuple[BenchmarkMessage, ...]
    ) -> tuple[BenchmarkMessage, ...]:
        limit = self.spec.maximum_messages_per_profile
        if limit is None or limit >= len(messages):
            return messages
        ordered = sorted(
            messages,
            key=lambda message: hashlib.sha256(
                f"{self.spec.seed}:{message.message_id}".encode()
            ).digest(),
        )
        return tuple(ordered[:limit])

    def _evaluate_message(
        self,
        message: BenchmarkMessage,
        pipeline: PersonalizedLanguagePipeline,
        adapter: PersonalizationAdapterManifest,
    ) -> tuple[LanguageBenchmarkTrial, ...]:
        records: list[LanguageBenchmarkTrial] = []
        confirmed: list[str] = []
        for span_index, intended_text in enumerate(message.target_spans):
            context = " ".join(confirmed)
            trial_id = f"language-{message.message_id}-{span_index:02d}"
            request = CandidateGenerationRequest(
                confirmed_text=context,
                candidate_count=self.spec.candidate_count,
                maximum_phrase_tokens=self.spec.maximum_phrase_tokens,
            )
            try:
                result = pipeline.generate(
                    request,
                    profile_id=message.profile_id,
                    at_time=self.spec.retrieval_at,
                )
            except CandidateGenerationError as error:
                records.append(
                    LanguageBenchmarkTrial(
                        trial_id=trial_id,
                        profile_id=message.profile_id,
                        message_id=message.message_id,
                        span_index=span_index,
                        message_span_count=len(message.target_spans),
                        confirmed_context=context,
                        intended_text=intended_text,
                        candidate_generation_failed=True,
                        backend_output_repaired=False,
                        failure_reason=str(error),
                        adapter_id=adapter.adapter_id,
                        adapter_sha256=adapter.adapter_sha256,
                    )
                )
                confirmed.append(intended_text)
                continue

            candidates = result.generation.candidate_set.candidates
            language_candidates = tuple(
                candidate for candidate in candidates if candidate.kind is not CandidateKind.CONTROL
            )
            intended = next(
                (
                    candidate
                    for candidate in language_candidates
                    if _normalized_text(candidate.text) == _normalized_text(intended_text)
                ),
                None,
            )
            intended_id = intended.candidate_id if intended is not None else None
            candidate_order = tuple(candidate.candidate_id for candidate in language_candidates)
            other_id = next(
                candidate_id
                for candidate_id, action in result.generation.control_actions.items()
                if action.value == "other"
            )
            records.append(
                LanguageBenchmarkTrial(
                    trial_id=trial_id,
                    profile_id=message.profile_id,
                    message_id=message.message_id,
                    span_index=span_index,
                    message_span_count=len(message.target_spans),
                    confirmed_context=context,
                    intended_text=intended_text,
                    backend_output_repaired=(result.generation.diagnostics.backend_output_repaired),
                    candidate_set=result.generation.candidate_set,
                    intended_candidate_id=intended_id,
                    other_candidate_id=other_id,
                    generic_language_support=result.generation.generic_language_support,
                    personalization_support=result.personalization_support,
                    personalization_lift=result.personalization_lift,
                    retrieval_evidence=result.retrieval_evidence,
                    generic_rank=(
                        _rank(
                            result.generation.generic_language_support,
                            target_id=intended_id,
                            candidate_order=candidate_order,
                        )
                        if intended_id is not None
                        else None
                    ),
                    personalized_rank=(
                        _rank(
                            result.personalization_support,
                            target_id=intended_id,
                            candidate_order=candidate_order,
                        )
                        if intended_id is not None
                        else None
                    ),
                    adapter_id=adapter.adapter_id,
                    adapter_sha256=adapter.adapter_sha256,
                )
            )
            confirmed.append(intended_text)
        return tuple(records)
