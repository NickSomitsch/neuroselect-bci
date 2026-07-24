"""Candidate-aligned style personalization and RAG evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.core.models import Candidate, CandidateKind
from neuroselect.language.generation import CandidateGenerator
from neuroselect.language.local_models import LocalModelCandidateBackend
from neuroselect.language.models import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    ShortIdentifier,
)
from neuroselect.retrieval.models import CandidateRetrievalEvidence
from neuroselect.retrieval.retriever import LexicalRetriever
from neuroselect.synthetic.models import SyntheticProfile

SHA256_PATTERN = r"^[0-9a-f]{64}$"
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PersonalizationAdapterManifest(BaseModel):
    """Verifiable MLX LoRA adapter provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    adapter_id: ShortIdentifier
    profile_id: str = Field(pattern=r"^synthetic-[a-z0-9-]+$", max_length=128)
    base_model_id: str = Field(min_length=1, max_length=500)
    base_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter_file: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    source_corpus_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    training_config_sha256: str = Field(pattern=SHA256_PATTERN)
    trainer_revision: ShortIdentifier
    mlx_lm_version: ShortIdentifier
    trained_at: datetime
    validation_evaluated: bool
    test_evaluated: bool
    synthetic_data: Literal[True] = True

    @model_validator(mode="after")
    def require_aware_training_time(self) -> PersonalizationAdapterManifest:
        if self.trained_at.tzinfo is None or self.trained_at.utcoffset() is None:
            raise ValueError("trained_at must include a timezone")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class PersonalizationAdapterBundle:
    """Validated adapter directory and manifest."""

    directory: Path
    manifest: PersonalizationAdapterManifest


def load_personalization_adapter(
    path: str | Path,
    *,
    expected_profile_id: str | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
) -> PersonalizationAdapterBundle:
    """Load an adapter only after checking its typed manifest and weight digest."""

    directory = Path(path)
    manifest_path = directory / "manifest.json"
    try:
        manifest = PersonalizationAdapterManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise ValueError(f"adapter manifest not found: {manifest_path}") from error
    adapter_path = directory / manifest.adapter_file
    if not adapter_path.is_file():
        raise ValueError(f"adapter weights not found: {adapter_path}")
    if sha256_file(adapter_path) != manifest.adapter_sha256:
        raise ValueError("adapter weight checksum does not match its manifest")
    if expected_profile_id is not None and manifest.profile_id != expected_profile_id:
        raise ValueError("adapter profile does not match the requested profile")
    if expected_model_id is not None and manifest.base_model_id != expected_model_id:
        raise ValueError("adapter base model does not match the configured model")
    if (
        expected_model_revision is not None
        and manifest.base_model_revision != expected_model_revision
    ):
        raise ValueError("adapter base-model revision does not match the configured model")
    return PersonalizationAdapterBundle(directory=directory, manifest=manifest)


class PersonalizationProvenance(BaseModel):
    """Visible distinction between a real held-out adapter and a controlled proxy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    evidence_kind: Literal["held_out_adapter", "controlled_fixture"]
    adapter_id: ShortIdentifier
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    base_model_id: str
    base_model_revision: str
    source_corpus_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    training_config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    validation_evaluated: bool = False
    test_evaluated: bool = False
    synthetic_data: Literal[True] = True

    @model_validator(mode="after")
    def require_held_out_evidence(self) -> PersonalizationProvenance:
        if self.evidence_kind == "held_out_adapter":
            if self.source_corpus_manifest_sha256 is None:
                raise ValueError("held-out adapters require corpus provenance")
            if self.training_config_sha256 is None:
                raise ValueError("held-out adapters require training-config provenance")
            if not self.validation_evaluated or not self.test_evaluated:
                raise ValueError("held-out adapters require validation and test evaluation")
        elif (
            self.source_corpus_manifest_sha256 is not None
            or self.training_config_sha256 is not None
        ):
            raise ValueError("controlled fixtures cannot claim training provenance")
        return self


class CandidatePersonalizer(Protocol):
    """Candidate-aligned personal support without changing the visible set."""

    provenance: PersonalizationProvenance

    def score(
        self,
        request: CandidateGenerationRequest,
        candidates: tuple[Candidate, ...],
    ) -> dict[str, float]: ...


def _normalize_positive(values: dict[str, float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("personalization scores must be finite and positive")
    total = sum(values.values())
    return {candidate_id: value / total for candidate_id, value in values.items()}


class ControlledStylePersonalizer:
    """Deterministic style proxy for mechanics tests; never presented as a LoRA."""

    def __init__(self, profile: SyntheticProfile) -> None:
        self.profile = profile
        style_payload = json.dumps(
            {
                "style_summary": profile.style_summary,
                "style": profile.style.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        style_sha256 = hashlib.sha256(style_payload.encode()).hexdigest()
        self.provenance = PersonalizationProvenance(
            profile_id=profile.profile_id,
            evidence_kind="controlled_fixture",
            adapter_id=f"controlled-style-{profile.profile_id}-v1",
            adapter_sha256=style_sha256,
            base_model_id="neuroselect/controlled-style-proxy",
            base_model_revision="controlled-style-proxy-v1",
        )

    def score(
        self,
        request: CandidateGenerationRequest,
        candidates: tuple[Candidate, ...],
    ) -> dict[str, float]:
        del request
        preferred = {
            token.casefold()
            for value in self.profile.style.preferred_vocabulary
            for token in TOKEN_PATTERN.findall(value)
        }
        concise = "short" in (
            f"{self.profile.style_summary} {self.profile.style.sentence_pattern}".casefold()
        )
        raw: dict[str, float] = {}
        for candidate in candidates:
            if candidate.kind is CandidateKind.CONTROL:
                continue
            tokens = tuple(token.casefold() for token in TOKEN_PATTERN.findall(candidate.text))
            vocabulary_matches = len(preferred.intersection(tokens))
            length_factor = 1.0 / max(1, len(tokens)) if concise else min(4, len(tokens)) / 4
            raw[candidate.candidate_id] = math.exp(0.7 * vocabulary_matches + 0.35 * length_factor)
        return _normalize_positive(raw)


class MlxAdapterPersonalizer:
    """Score the fixed candidate set with one verified MLX LoRA adapter."""

    def __init__(
        self,
        backend: LocalModelCandidateBackend,
        bundle: PersonalizationAdapterBundle,
    ) -> None:
        manifest = bundle.manifest
        if manifest.base_model_id != backend.config.model_id:
            raise ValueError("adapter and scoring backend use different base models")
        if manifest.base_model_revision != backend.config.model_revision:
            raise ValueError("adapter and scoring backend use different model revisions")
        self.backend = backend
        self.provenance = PersonalizationProvenance(
            profile_id=manifest.profile_id,
            evidence_kind="held_out_adapter",
            adapter_id=manifest.adapter_id,
            adapter_sha256=manifest.adapter_sha256,
            base_model_id=manifest.base_model_id,
            base_model_revision=manifest.base_model_revision,
            source_corpus_manifest_sha256=manifest.source_corpus_manifest_sha256,
            training_config_sha256=manifest.training_config_sha256,
            validation_evaluated=manifest.validation_evaluated,
            test_evaluated=manifest.test_evaluated,
        )

    def score(
        self,
        request: CandidateGenerationRequest,
        candidates: tuple[Candidate, ...],
    ) -> dict[str, float]:
        language_candidates = tuple(
            candidate for candidate in candidates if candidate.kind is not CandidateKind.CONTROL
        )
        support = self.backend.score_texts(
            request, tuple(candidate.text for candidate in language_candidates)
        )
        return {
            candidate.candidate_id: value
            for candidate, value in zip(language_candidates, support, strict=True)
        }


class PersonalizedGenerationResult(BaseModel):
    """Aligned generic, adapter, and retrieval signals for later fusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: CandidateGenerationResult
    profile_id: str
    personalization_support: dict[str, float]
    personalization_lift: dict[str, float]
    retrieval_support: dict[str, float]
    retrieval_evidence: tuple[CandidateRetrievalEvidence, ...]
    personalization: PersonalizationProvenance
    claim_eligible: bool

    @model_validator(mode="after")
    def validate_aligned_signals(self) -> PersonalizedGenerationResult:
        language_ids = {
            candidate.candidate_id
            for candidate in self.generation.candidate_set.candidates
            if candidate.kind is not CandidateKind.CONTROL
        }
        if self.profile_id != self.personalization.profile_id:
            raise ValueError("result and personalization profile IDs must agree")
        for label, values in (
            ("personalization support", self.personalization_support),
            ("personalization lift", self.personalization_lift),
            ("retrieval support", self.retrieval_support),
        ):
            if set(values) != language_ids:
                raise ValueError(f"{label} must cover exactly the language candidates")
        if not math.isclose(
            sum(self.personalization_support.values()),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("personalization support must sum to one")
        expected_retrieval = {
            item.candidate_id: item.retrieval_support for item in self.retrieval_evidence
        }
        if self.retrieval_support != expected_retrieval:
            raise ValueError("retrieval support must match the detailed evidence")
        expected_lift = {
            candidate_id: self.personalization_support[candidate_id]
            - self.generation.generic_language_support[candidate_id]
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
        expected_claim = self.personalization.evidence_kind == "held_out_adapter"
        if self.claim_eligible != expected_claim:
            raise ValueError("claim eligibility must follow personalization evidence kind")
        return self


class PersonalizedLanguagePipeline:
    """Keep generation fixed while deriving separate style and RAG evidence."""

    def __init__(
        self,
        generator: CandidateGenerator,
        personalizer: CandidatePersonalizer,
        retriever: LexicalRetriever,
    ) -> None:
        self.generator = generator
        self.personalizer = personalizer
        self.retriever = retriever

    def generate(
        self,
        request: CandidateGenerationRequest,
        *,
        profile_id: str,
        at_time: datetime,
    ) -> PersonalizedGenerationResult:
        if profile_id != self.personalizer.provenance.profile_id:
            raise ValueError("requested profile does not match the personalizer")
        generation = self.generator.generate(request)
        candidates = generation.candidate_set.candidates
        personal_support = self.personalizer.score(request, candidates)
        lift = {
            candidate_id: personal_support[candidate_id] - generic_support
            for candidate_id, generic_support in generation.generic_language_support.items()
        }
        retrieval = self.retriever.retrieve_for_candidates(
            profile_id=profile_id,
            confirmed_text=request.confirmed_text,
            candidates=candidates,
            at_time=at_time,
        )
        retrieval_support = {
            evidence.candidate_id: evidence.retrieval_support for evidence in retrieval
        }
        return PersonalizedGenerationResult(
            generation=generation,
            profile_id=profile_id,
            personalization_support=personal_support,
            personalization_lift=lift,
            retrieval_support=retrieval_support,
            retrieval_evidence=retrieval,
            personalization=self.personalizer.provenance,
            claim_eligible=(self.personalizer.provenance.evidence_kind == "held_out_adapter"),
        )
