"""Optional MLX-LM candidate generation and exact continuation scoring."""

from __future__ import annotations

import importlib
import math
import platform
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from neuroselect.language.generation import (
    CandidateGenerationError,
)
from neuroselect.language.models import (
    BackendMetadata,
    CandidateGenerationRequest,
    CandidateProposal,
    ShortIdentifier,
)
from neuroselect.language.personalization_data import personalization_prompt

DEFAULT_LOCAL_MODEL_CONFIG = Path("configs/models/qwen3_4b_mlx.yaml")
ChatMessage = dict[str, str]
SUPPORTED_MLX_PLATFORMS = frozenset({("Darwin", "arm64"), ("Linux", "x86_64")})


class LocalModelDependencyError(RuntimeError):
    """Raised when optional local-model dependencies are unavailable."""


class LocalGenerationConfig(BaseModel):
    """Bounded deterministic-decoding settings for candidate proposals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_thinking: Literal[False] = False
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    maximum_new_tokens: int = Field(default=512, ge=64, le=2_048)
    proposal_multiplier: int = Field(default=2, ge=1, le=4)
    minimum_proposals: int = Field(default=12, ge=9, le=36)


class CandidateScoringConfig(BaseModel):
    """Documented conversion from token likelihoods to relative support."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["mean-token-log-likelihood"] = "mean-token-log-likelihood"
    softmax_temperature: float = Field(default=1.0, gt=0.0, le=10.0)


class LocalModelConfig(BaseModel):
    """Pinned local model and prompt provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    backend: Literal["mlx-lm"]
    backend_id: ShortIdentifier
    model_id: str = Field(min_length=1, max_length=500)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: ShortIdentifier
    download_enabled: bool = False
    deterministic: bool = False
    generator_revision: ShortIdentifier
    prompt_revision: ShortIdentifier
    generation: LocalGenerationConfig
    scoring: CandidateScoringConfig

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_id=self.backend_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            generator_revision=self.generator_revision,
            prompt_revision=self.prompt_revision,
            deterministic=self.deterministic,
        )


def load_local_model_config(
    path: str | Path = DEFAULT_LOCAL_MODEL_CONFIG,
) -> LocalModelConfig:
    """Load a strict, revision-pinned local model definition."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("local model configuration must contain a YAML mapping")
    return LocalModelConfig.model_validate(payload)


def resolve_local_model_source(config: LocalModelConfig, *, allow_download: bool = False) -> str:
    """Resolve a local path or an exact cached/downloaded Hugging Face revision."""

    model_source = Path(config.model_id)
    if model_source.exists():
        return str(model_source)
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise LocalModelDependencyError(
            "MLX-LM is optional; install `local-language` on Apple silicon or "
            "`local-language-cuda` on Linux"
        ) from error
    try:
        resolved: str = hub.snapshot_download(
            repo_id=config.model_id,
            revision=config.model_revision,
            local_files_only=not (allow_download or config.download_enabled),
        )
    except Exception as error:
        raise LocalModelDependencyError(
            "the pinned model is not in the local Hugging Face cache; rerun with explicit "
            "download permission"
        ) from error
    return resolved


def normalize_log_scores(
    log_scores: tuple[float, ...], *, temperature: float = 1.0
) -> tuple[float, ...]:
    """Convert finite log scores into stable relative support."""

    if not log_scores:
        raise ValueError("at least one candidate log score is required")
    if temperature <= 0.0:
        raise ValueError("softmax temperature must be positive")
    if any(not math.isfinite(score) for score in log_scores):
        raise ValueError("candidate log scores must be finite")
    scaled = tuple(score / temperature for score in log_scores)
    maximum = max(scaled)
    exponentials = tuple(math.exp(score - maximum) for score in scaled)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


class LanguageModelRuntime(Protocol):
    """Small runtime seam that keeps MLX optional and unit-testable."""

    def generate(self, messages: tuple[ChatMessage, ...], *, max_tokens: int) -> str: ...

    def score_continuations(
        self,
        messages: tuple[ChatMessage, ...],
        continuations: tuple[str, ...],
    ) -> tuple[float, ...]: ...


class CandidateVocabularyProvider(Protocol):
    """Context-only candidate vocabulary used by held-out-safe evaluation."""

    def phrases_for(self, confirmed_text: str) -> tuple[str, ...]: ...


def _generation_messages(
    request: CandidateGenerationRequest,
    *,
    proposal_count: int,
    allowed_phrases: tuple[str, ...] = (),
) -> tuple[ChatMessage, ...]:
    context = request.confirmed_text or "(empty message)"
    vocabulary_instruction = ""
    if allowed_phrases:
        vocabulary_instruction = (
            "\nAllowed phrases learned only from non-test messages: "
            + " | ".join(allowed_phrases)
            + "\nEvery candidate must be copied exactly from this allowed list and must append "
            "grammatically to the confirmed message."
        )
    return (
        {
            "role": "system",
            "content": (
                "You are an autocomplete candidate generator for a person composing an "
                "assistive-communication message. Every candidate must be a fragment that can be "
                "appended verbatim after the confirmed message to continue that person's message. "
                "Do not answer the message, describe an action, issue an interface command, "
                "paraphrase or repeat the confirmed text. When the confirmed message is empty, "
                "propose beginnings of messages the person could compose. For example, after "
                "'Could you' a valid candidate is 'bring the water', while 'Sure, I can' and "
                "'Send message' are invalid. After 'I feel' a valid candidate is 'tired today', "
                "while 'How can I help?' is invalid. Favor probable grammatical continuations "
                "while keeping the alternatives meaningfully distinct. "
                "Return only one JSON object with exactly the shape "
                '{"candidates":[{"text":"short phrase","support":0.0}]}. '
                "Do not propose Other, Back, or Cancel. Do not include markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Confirmed message: {context!r}\n"
                f"Produce exactly {proposal_count} distinct direct continuations. Each candidate "
                f"must contain between one and {request.maximum_phrase_tokens} "
                "whitespace-delimited tokens. The support numbers are placeholders and will be "
                f"replaced by measured model likelihoods.{vocabulary_instruction}"
            ),
        },
    )


def continuation_scoring_messages(
    request: CandidateGenerationRequest,
) -> tuple[ChatMessage, ...]:
    """Build the stable prompt whose assistant continuation is likelihood-scored."""

    return (
        {
            "role": "user",
            "content": personalization_prompt(request.confirmed_text),
        },
    )


class MlxLanguageRuntime:
    """Lazy MLX-LM runtime with cache-only loading unless explicitly allowed."""

    def __init__(
        self,
        config: LocalModelConfig,
        *,
        adapter_path: str | Path | None = None,
        allow_download: bool = False,
    ) -> None:
        self.config = config
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.allow_download = allow_download
        self._model: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None
        self._generate: Any = None
        self._make_sampler: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        runtime_platform = (platform.system(), platform.machine())
        if runtime_platform not in SUPPORTED_MLX_PLATFORMS:
            raise LocalModelDependencyError(
                "the tracked MLX backend requires Apple silicon or Linux x86_64 with an "
                "MLX-compatible NVIDIA GPU"
            )
        try:
            mlx_lm = importlib.import_module("mlx_lm")
            self._mx = importlib.import_module("mlx.core")
            sample_utils = importlib.import_module("mlx_lm.sample_utils")
        except ImportError as error:
            raise LocalModelDependencyError(
                "MLX-LM is optional; install `local-language` on Apple silicon or "
                "`local-language-cuda` on Linux"
            ) from error
        if runtime_platform[0] == "Linux":
            try:
                self._mx.set_default_device(self._mx.gpu)
                if self._mx.default_device().type != self._mx.gpu:
                    raise RuntimeError("MLX did not select its GPU device")
            except Exception as error:
                raise LocalModelDependencyError(
                    "Linux MLX loaded without a usable CUDA GPU; run the language CUDA "
                    "preflight and choose a GPU with compute capability 7.5 or newer"
                ) from error

        resolved_source = resolve_local_model_source(
            self.config, allow_download=self.allow_download
        )

        load = mlx_lm.load
        self._model, self._tokenizer = load(
            resolved_source,
            adapter_path=(str(self.adapter_path) if self.adapter_path is not None else None),
            revision=None,
        )
        self._generate = mlx_lm.generate
        self._make_sampler = sample_utils.make_sampler

    def _render(self, messages: tuple[ChatMessage, ...]) -> str:
        self._load()
        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.generation.enable_thinking,
        )
        if not isinstance(rendered, str):
            raise CandidateGenerationError("tokenizer did not return a text chat prompt")
        return rendered

    def generate(self, messages: tuple[ChatMessage, ...], *, max_tokens: int) -> str:
        prompt = self._render(messages)
        sampler = self._make_sampler(temp=self.config.generation.temperature)
        generated = self._generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        if not isinstance(generated, str):
            raise CandidateGenerationError("MLX-LM returned a non-text generation")
        return generated

    def score_continuations(
        self,
        messages: tuple[ChatMessage, ...],
        continuations: tuple[str, ...],
    ) -> tuple[float, ...]:
        if not continuations:
            raise ValueError("at least one continuation is required")
        prompt = self._render(messages)
        prompt_ids = tuple(self._tokenizer.encode(prompt, add_special_tokens=False))
        if not prompt_ids:
            raise CandidateGenerationError("tokenizer produced an empty scoring prompt")

        scores: list[float] = []
        for continuation in continuations:
            continuation_ids = tuple(
                self._tokenizer.encode(f" {continuation}", add_special_tokens=False)
            )
            if not continuation_ids:
                raise CandidateGenerationError("tokenizer produced an empty continuation")
            combined = (*prompt_ids, *continuation_ids)
            inputs = self._mx.array(combined[:-1])[None, :]
            outputs = self._model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            start = len(prompt_ids) - 1
            stop = start + len(continuation_ids)
            completion_logits = logits[0, start:stop, :]
            log_probabilities = completion_logits - self._mx.logsumexp(
                completion_logits, axis=-1, keepdims=True
            )
            targets = self._mx.array(continuation_ids)[:, None]
            selected = self._mx.take_along_axis(log_probabilities, targets, axis=-1).squeeze(-1)
            self._mx.eval(selected)
            scores.append(float(self._mx.mean(selected).item()))
        return tuple(scores)


class LocalModelCandidateBackend:
    """Generate structured proposals and replace self-reported support with likelihoods."""

    def __init__(
        self,
        config: LocalModelConfig | None = None,
        *,
        runtime: LanguageModelRuntime | None = None,
        adapter_path: str | Path | None = None,
        allow_download: bool = False,
        candidate_vocabulary: CandidateVocabularyProvider | None = None,
    ) -> None:
        self.config = config or load_local_model_config()
        self.metadata = self.config.metadata
        self.runtime = runtime or MlxLanguageRuntime(
            self.config,
            adapter_path=adapter_path,
            allow_download=allow_download,
        )
        self.candidate_vocabulary = candidate_vocabulary
        self.last_output_repaired = False

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]:
        language_quota = request.candidate_count - 3
        allowed_phrases = (
            self.candidate_vocabulary.phrases_for(request.confirmed_text)
            if self.candidate_vocabulary is not None
            else ()
        )
        if allowed_phrases and len(allowed_phrases) < language_quota:
            raise CandidateGenerationError(
                "candidate vocabulary does not cover the visible language quota"
            )
        proposal_count = (
            len(allowed_phrases)
            if allowed_phrases
            else max(
                self.config.generation.minimum_proposals,
                language_quota * self.config.generation.proposal_multiplier,
            )
        )
        messages = _generation_messages(
            request,
            proposal_count=proposal_count,
            allowed_phrases=allowed_phrases,
        )
        raw = self.runtime.generate(messages, max_tokens=self.config.generation.maximum_new_tokens)
        from neuroselect.language.generation import (
            _parse_structured_proposals_with_diagnostics,
        )

        proposals, self.last_output_repaired = _parse_structured_proposals_with_diagnostics(raw)
        if allowed_phrases:
            canonical = {self._vocabulary_key(phrase): phrase for phrase in allowed_phrases}
            filtered: dict[str, CandidateProposal] = {}
            for proposal in proposals:
                key = self._vocabulary_key(proposal.text)
                if key in canonical:
                    filtered.setdefault(
                        key,
                        proposal.model_copy(update={"text": canonical[key]}),
                    )
            proposals = tuple(filtered.values())
            if not proposals:
                raise CandidateGenerationError(
                    "backend returned no candidates from the non-test vocabulary"
                )
        support = self.score_texts(request, tuple(proposal.text for proposal in proposals))
        return tuple(
            proposal.model_copy(update={"support": score})
            for proposal, score in zip(proposals, support, strict=True)
        )

    @staticmethod
    def _vocabulary_key(value: str) -> str:
        return " ".join(value.casefold().split()).rstrip(".,!?;:")

    def score_texts(
        self,
        request: CandidateGenerationRequest,
        candidate_texts: tuple[str, ...],
    ) -> tuple[float, ...]:
        """Return normalized phrase support from exact continuation log likelihood."""

        log_scores = self.runtime.score_continuations(
            continuation_scoring_messages(request), candidate_texts
        )
        if len(log_scores) != len(candidate_texts):
            raise CandidateGenerationError(
                "runtime returned a different number of candidate scores"
            )
        return normalize_log_scores(
            log_scores,
            temperature=self.config.scoring.softmax_temperature,
        )
