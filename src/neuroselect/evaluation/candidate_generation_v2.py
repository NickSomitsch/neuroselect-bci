"""Target-blind exploratory candidate generation fitted only on non-test messages."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation.language_benchmark import LanguageBenchmarkTrial
from neuroselect.synthetic import BenchmarkMessage, BenchmarkSplit, GeneratedBenchmark

DEFAULT_CANDIDATE_GENERATION_V2_CONFIG = Path(
    "configs/publication/candidate_generation_v2_exploratory.yaml"
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(text.split()).rstrip(".,!?;:")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", _normalized(value)))


class CandidateRole(StrEnum):
    OPENING = "opening"
    OBJECT = "object"
    TIME_QUALIFIER = "time_qualifier"
    LOCATION = "location"
    REQUEST_ENDING = "request_ending"
    ENDING = "ending"
    OTHER = "other"


class CandidateGenerationV2Spec(BaseModel):
    """Locked exploratory recipe whose generator never receives a target span."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal["candidate-generation-v2-exploratory-v1"]
    generated_at: datetime
    publication_protocol: Path
    expected_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_language_artifacts: Path
    expected_primary_language_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_spec: Path
    profiles_directory: Path
    expected_benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitting_source_splits: tuple[Literal["train"], Literal["validation"]]
    language_candidate_count: Literal[9] = 9
    maximum_phrase_tokens: Literal[4] = 4
    request_object_time_quota: int = Field(default=7, ge=1, le=9)
    request_object_location_quota: int = Field(default=1, ge=0, le=8)
    request_object_ending_quota: int = Field(default=1, ge=0, le=8)
    bootstrap_resamples: int = Field(default=10_000, ge=2_000, le=100_000)
    bootstrap_seed: int = Field(default=20260727, ge=0)
    design_status: Literal["exploratory_test_exposed"] = "exploratory_test_exposed"

    @model_validator(mode="after")
    def validate_recipe(self) -> CandidateGenerationV2Spec:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("candidate-generation v2 time must include a timezone")
        if self.fitting_source_splits != ("train", "validation"):
            raise ValueError("candidate-generation v2 must fit only train and validation")
        if (
            self.request_object_time_quota
            + self.request_object_location_quota
            + self.request_object_ending_quota
            != self.language_candidate_count
        ):
            raise ValueError("request-object routing quotas must fill the visible candidate set")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


def load_candidate_generation_v2_spec(
    path: str | Path = DEFAULT_CANDIDATE_GENERATION_V2_CONFIG,
) -> CandidateGenerationV2Spec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("candidate-generation v2 config must contain a YAML mapping")
    return CandidateGenerationV2Spec.model_validate(payload)


class CandidateSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["train", "validation"]
    message_id: str
    profile_id: str
    span_indices: tuple[int, ...] = Field(min_length=1, max_length=2)


class CandidateBankEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=160)
    normalized_text: str = Field(min_length=1, max_length=160)
    role: CandidateRole
    occurrence_count: int = Field(ge=1)
    profile_counts: dict[str, int]
    source_contexts: tuple[str, ...]
    source_refs: tuple[CandidateSourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entry(self) -> CandidateBankEntry:
        if self.normalized_text != _normalized(self.text):
            raise ValueError("candidate-bank normalized text must match candidate text")
        if len(self.text.split()) > 4:
            raise ValueError("candidate-bank phrases cannot exceed four tokens")
        if sum(self.profile_counts.values()) != self.occurrence_count:
            raise ValueError("candidate-bank profile counts must cover every occurrence")
        return self


class CandidateBank(BaseModel):
    """Auditable non-test phrase bank used by the deterministic generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    revision: Literal["non-test-contextual-candidates-v2"] = "non-test-contextual-candidates-v2"
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_splits: tuple[Literal["train"], Literal["validation"]]
    source_message_ids: tuple[str, ...] = Field(min_length=1)
    entries: tuple[CandidateBankEntry, ...] = Field(min_length=9)

    @model_validator(mode="after")
    def validate_bank(self) -> CandidateBank:
        if self.source_splits != ("train", "validation"):
            raise ValueError("candidate bank can only contain train and validation sources")
        source_ids = {source.message_id for entry in self.entries for source in entry.source_refs}
        if source_ids - set(self.source_message_ids):
            raise ValueError("candidate-bank entries reference undeclared source messages")
        keys = [(entry.role, entry.normalized_text) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate-bank role/text pairs must be unique")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


def _request_like(value: str) -> bool:
    context = f" {_normalized(value)} "
    return any(
        cue in context
        for cue in (
            " please ",
            " can you ",
            " could you ",
            " would you ",
            " kindly ",
            " remember to ",
            " arrange to ",
            " did you ",
            " confirm ",
        )
    )


def _direct_role(context: str, text: str) -> CandidateRole:
    normalized = _normalized(text)
    if not context:
        return CandidateRole.OPENING
    if text.rstrip().endswith((".", "?", "!")):
        return CandidateRole.REQUEST_ENDING if _request_like(context) else CandidateRole.ENDING
    if normalized.startswith(("the ", "my ", "today's ")):
        return CandidateRole.OBJECT
    if normalized.startswith(("to the ", "at the ", "near the ")):
        return CandidateRole.LOCATION
    return CandidateRole.OTHER


def _is_time_lead(value: str) -> bool:
    normalized = _normalized(value)
    return normalized in {
        "after",
        "around",
        "at",
        "at approximately",
        "before",
        "by",
        "shortly after",
        "shortly before",
        "sometime around",
    }


def build_candidate_bank_v2(
    benchmark: GeneratedBenchmark,
    spec: CandidateGenerationV2Spec,
) -> CandidateBank:
    """Fit only on declared non-test partitions; test messages are never traversed."""

    aggregates: dict[
        tuple[CandidateRole, str],
        dict[str, Any],
    ] = {}
    source_message_ids: set[str] = set()

    def add(
        *,
        text: str,
        role: CandidateRole,
        message: BenchmarkMessage,
        context: str,
        span_indices: tuple[int, ...],
    ) -> None:
        if not text or len(text.split()) > spec.maximum_phrase_tokens:
            return
        key = (role, _normalized(text))
        aggregate = aggregates.setdefault(
            key,
            {
                "text_counts": Counter(),
                "profile_counts": Counter(),
                "contexts": set(),
                "refs": [],
            },
        )
        aggregate["text_counts"][text] += 1
        aggregate["profile_counts"][message.profile_id] += 1
        aggregate["contexts"].add(_normalized(context))
        aggregate["refs"].append(
            CandidateSourceRef(
                split=cast(Literal["train", "validation"], message.split.value),
                message_id=message.message_id,
                profile_id=message.profile_id,
                span_indices=span_indices,
            )
        )

    for split_name in spec.fitting_source_splits:
        split = BenchmarkSplit(split_name)
        for message in benchmark.messages[split]:
            source_message_ids.add(message.message_id)
            confirmed: list[str] = []
            for index, text in enumerate(message.target_spans):
                context = " ".join(confirmed)
                add(
                    text=text,
                    role=_direct_role(context, text),
                    message=message,
                    context=context,
                    span_indices=(index,),
                )
                if (
                    index + 1 < len(message.target_spans)
                    and _is_time_lead(text)
                    and len(f"{text} {message.target_spans[index + 1]}".split())
                    <= spec.maximum_phrase_tokens
                ):
                    add(
                        text=f"{text} {message.target_spans[index + 1]}",
                        role=CandidateRole.TIME_QUALIFIER,
                        message=message,
                        context=context,
                        span_indices=(index, index + 1),
                    )
                confirmed.append(text)

    entries: list[CandidateBankEntry] = []
    for (role, normalized), aggregate in sorted(
        aggregates.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        text = min(
            aggregate["text_counts"],
            key=lambda candidate: (-aggregate["text_counts"][candidate], candidate.casefold()),
        )
        refs = tuple(
            sorted(
                set(aggregate["refs"]),
                key=lambda item: (item.split, item.message_id, item.span_indices),
            )
        )
        entries.append(
            CandidateBankEntry(
                text=text,
                normalized_text=normalized,
                role=role,
                occurrence_count=sum(aggregate["profile_counts"].values()),
                profile_counts=dict(sorted(aggregate["profile_counts"].items())),
                source_contexts=tuple(sorted(aggregate["contexts"]))[:64],
                source_refs=refs,
            )
        )
    return CandidateBank(
        benchmark_source_sha256=benchmark.source_sha256,
        source_splits=spec.fitting_source_splits,
        source_message_ids=tuple(sorted(source_message_ids)),
        entries=tuple(entries),
    )


class GeneratedCandidateV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    role: CandidateRole
    retrieval_score: float
    source_occurrence_count: int = Field(ge=1)


class TargetBlindContextualGeneratorV2:
    """Generate from profile, visible context, and round number only."""

    def __init__(self, bank: CandidateBank, spec: CandidateGenerationV2Spec) -> None:
        self.bank = bank
        self.spec = spec
        self._by_role = {
            role: tuple(entry for entry in bank.entries if entry.role is role)
            for role in CandidateRole
        }
        self._cache: dict[tuple[str, str, int], tuple[GeneratedCandidateV2, ...]] = {}

    @staticmethod
    def _context_similarity(left: str, right: str) -> float:
        left_tokens = _tokens(left)
        right_tokens = _tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(set(left_tokens) & set(right_tokens))
        union = len(set(left_tokens) | set(right_tokens))
        score = intersection / union if union else 0.0
        if left_tokens[-1:] == right_tokens[-1:]:
            score += 0.75
        if left_tokens[-2:] == right_tokens[-2:]:
            score += 1.25
        return score

    def _score(
        self,
        entry: CandidateBankEntry,
        *,
        profile_id: str,
        confirmed_context: str,
    ) -> float:
        similarity = max(
            (
                self._context_similarity(confirmed_context, source_context)
                for source_context in entry.source_contexts
            ),
            default=0.0,
        )
        profile_count = entry.profile_counts.get(profile_id, 0)
        return (
            10.0 * similarity + 2.0 * math.log1p(profile_count) + math.log1p(entry.occurrence_count)
        )

    def _rank_role(
        self,
        role: CandidateRole,
        *,
        profile_id: str,
        confirmed_context: str,
        predicate: Any = None,
    ) -> list[tuple[CandidateBankEntry, float]]:
        entries = self._by_role[role]
        if predicate is not None:
            entries = tuple(entry for entry in entries if predicate(entry))
        return sorted(
            (
                (
                    entry,
                    self._score(
                        entry,
                        profile_id=profile_id,
                        confirmed_context=confirmed_context,
                    ),
                )
                for entry in entries
            ),
            key=lambda item: (-item[1], item[0].normalized_text),
        )

    def _ends_with_role(self, context: str, role: CandidateRole) -> bool:
        normalized = _normalized(context)
        return any(normalized.endswith(entry.normalized_text) for entry in self._by_role[role])

    def generate(
        self,
        *,
        profile_id: str,
        confirmed_context: str,
        span_index: int,
    ) -> tuple[GeneratedCandidateV2, ...]:
        """Return nine candidates without accepting or reading an intended target."""

        key = (profile_id, _normalized(confirmed_context), span_index)
        if key in self._cache:
            return self._cache[key]

        plans: list[tuple[CandidateRole, int, Any]] = []
        if not _normalized(confirmed_context):
            plans = [(CandidateRole.OPENING, self.spec.language_candidate_count, None)]
        elif self._ends_with_role(confirmed_context, CandidateRole.TIME_QUALIFIER):
            plans = [(CandidateRole.REQUEST_ENDING, self.spec.language_candidate_count, None)]
        elif self._ends_with_role(confirmed_context, CandidateRole.OBJECT):
            plans = [
                (
                    CandidateRole.TIME_QUALIFIER,
                    self.spec.request_object_time_quota,
                    lambda entry: entry.normalized_text.startswith("before "),
                ),
                (
                    CandidateRole.LOCATION,
                    self.spec.request_object_location_quota,
                    None,
                ),
                (
                    CandidateRole.REQUEST_ENDING,
                    self.spec.request_object_ending_quota,
                    None,
                ),
            ]
        elif _request_like(confirmed_context):
            plans = [(CandidateRole.OBJECT, self.spec.language_candidate_count, None)]
        else:
            plans = [
                (CandidateRole.OBJECT, 4, None),
                (CandidateRole.REQUEST_ENDING, 2, None),
                (CandidateRole.TIME_QUALIFIER, 2, None),
                (CandidateRole.ENDING, 1, None),
            ]

        selected: list[tuple[CandidateBankEntry, float]] = []
        seen: set[str] = set()
        for role, quota, predicate in plans:
            added = 0
            for entry, score in self._rank_role(
                role,
                profile_id=profile_id,
                confirmed_context=confirmed_context,
                predicate=predicate,
            ):
                if entry.normalized_text in seen:
                    continue
                selected.append((entry, score))
                seen.add(entry.normalized_text)
                added += 1
                if added == quota:
                    break

        if len(selected) < self.spec.language_candidate_count:
            fallback = sorted(
                (
                    (
                        entry,
                        self._score(
                            entry,
                            profile_id=profile_id,
                            confirmed_context=confirmed_context,
                        ),
                    )
                    for entry in self.bank.entries
                ),
                key=lambda item: (-item[1], item[0].role.value, item[0].normalized_text),
            )
            for entry, score in fallback:
                if entry.normalized_text in seen:
                    continue
                selected.append((entry, score))
                seen.add(entry.normalized_text)
                if len(selected) == self.spec.language_candidate_count:
                    break
        if len(selected) != self.spec.language_candidate_count:
            raise ValueError("candidate bank cannot fill the configured visible candidate quota")

        result = tuple(
            GeneratedCandidateV2(
                text=entry.text,
                role=entry.role,
                retrieval_score=score,
                source_occurrence_count=entry.occurrence_count,
            )
            for entry, score in selected
        )
        self._cache[key] = result
        return result


class CandidateGenerationV2Trial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str
    profile_id: str
    message_id: str
    span_index: int = Field(ge=0)
    message_span_count: int = Field(ge=1)
    confirmed_context: str
    intended_text: str
    baseline_target_available: bool
    candidates: tuple[GeneratedCandidateV2, ...] = Field(min_length=9, max_length=9)
    target_rank: int | None = Field(default=None, ge=1, le=9)

    @model_validator(mode="after")
    def validate_target_rank(self) -> CandidateGenerationV2Trial:
        keys = [_normalized(candidate.text) for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate-generation v2 trials require unique candidates")
        expected = (
            keys.index(_normalized(self.intended_text)) + 1
            if _normalized(self.intended_text) in keys
            else None
        )
        if self.target_rank != expected:
            raise ValueError("candidate-generation v2 target rank must be scored post hoc")
        return self

    @property
    def target_available(self) -> bool:
        return self.target_rank is not None


class CandidateGenerationV2Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: str
    trial_count: int = Field(ge=1)
    message_count: int = Field(ge=1)
    baseline_target_availability_rate: float = Field(ge=0.0, le=1.0)
    v2_target_availability_rate: float = Field(ge=0.0, le=1.0)
    availability_delta: float = Field(ge=-1.0, le=1.0)
    baseline_message_availability_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    v2_message_availability_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    message_availability_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    v2_top1_rate: float = Field(ge=0.0, le=1.0)
    v2_top3_rate: float = Field(ge=0.0, le=1.0)
    gained_trial_count: int = Field(ge=0)
    lost_trial_count: int = Field(ge=0)


class CandidateGenerationV2Interval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: str
    metric: Literal["availability_delta", "message_availability_delta"]
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    resamples: int = Field(ge=2_000)
    sampling_unit: Literal["messages_within_fixed_profile_strata"]


class CandidateGenerationV2Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_language_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fitting_source_splits: tuple[Literal["train"], Literal["validation"]]
    intended_target_exposed_to_generator: Literal[False] = False
    design_status: Literal["exploratory_test_exposed"]
    trials: tuple[CandidateGenerationV2Trial, ...] = Field(min_length=1)
    metrics: tuple[CandidateGenerationV2Metrics, ...] = Field(min_length=1)
    intervals: tuple[CandidateGenerationV2Interval, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def _message_metrics(
    trials: Sequence[CandidateGenerationV2Trial],
    scope: str,
    *,
    include_complete_messages: bool = True,
) -> CandidateGenerationV2Metrics:
    messages: dict[tuple[str, str], list[CandidateGenerationV2Trial]] = defaultdict(list)
    for trial in trials:
        messages[(trial.profile_id, trial.message_id)].append(trial)
    baseline = np.asarray([trial.baseline_target_available for trial in trials], dtype=float)
    v2 = np.asarray([trial.target_available for trial in trials], dtype=float)
    baseline_message = np.asarray(
        [all(trial.baseline_target_available for trial in group) for group in messages.values()],
        dtype=float,
    )
    v2_message = np.asarray(
        [all(trial.target_available for trial in group) for group in messages.values()],
        dtype=float,
    )
    return CandidateGenerationV2Metrics(
        scope=scope,
        trial_count=len(trials),
        message_count=len(messages),
        baseline_target_availability_rate=float(np.mean(baseline)),
        v2_target_availability_rate=float(np.mean(v2)),
        availability_delta=float(np.mean(v2 - baseline)),
        baseline_message_availability_rate=(
            float(np.mean(baseline_message)) if include_complete_messages else None
        ),
        v2_message_availability_rate=(
            float(np.mean(v2_message)) if include_complete_messages else None
        ),
        message_availability_delta=(
            float(np.mean(v2_message - baseline_message)) if include_complete_messages else None
        ),
        v2_top1_rate=float(np.mean([trial.target_rank == 1 for trial in trials])),
        v2_top3_rate=float(
            np.mean([trial.target_rank is not None and trial.target_rank <= 3 for trial in trials])
        ),
        gained_trial_count=sum(
            trial.target_available and not trial.baseline_target_available for trial in trials
        ),
        lost_trial_count=sum(
            trial.baseline_target_available and not trial.target_available for trial in trials
        ),
    )


def _bootstrap_intervals(
    trials: Sequence[CandidateGenerationV2Trial],
    spec: CandidateGenerationV2Spec,
    scope: str,
) -> tuple[CandidateGenerationV2Interval, ...]:
    by_profile_message: dict[str, dict[str, list[CandidateGenerationV2Trial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trial in trials:
        by_profile_message[trial.profile_id][trial.message_id].append(trial)
    profiles = tuple(sorted(by_profile_message))
    point = _message_metrics(trials, scope)
    assert point.message_availability_delta is not None
    availability = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    availability_denominator = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    messages = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    message_denominator = 0
    rng = np.random.default_rng(
        spec.bootstrap_seed + int.from_bytes(hashlib.sha256(scope.encode()).digest()[:4], "big")
    )
    for profile in profiles:
        groups = tuple(by_profile_message[profile].values())
        span_delta = np.asarray(
            [
                sum(trial.target_available - trial.baseline_target_available for trial in group)
                for group in groups
            ],
            dtype=np.float64,
        )
        span_count = np.asarray([len(group) for group in groups], dtype=np.float64)
        message_delta = np.asarray(
            [
                float(all(trial.target_available for trial in group))
                - float(all(trial.baseline_target_available for trial in group))
                for group in groups
            ],
            dtype=np.float64,
        )
        for start in range(0, spec.bootstrap_resamples, 1_000):
            stop = min(start + 1_000, spec.bootstrap_resamples)
            indices = rng.integers(0, len(groups), size=(stop - start, len(groups)))
            availability[start:stop] += span_delta[indices].sum(axis=1)
            availability_denominator[start:stop] += span_count[indices].sum(axis=1)
            messages[start:stop] += message_delta[indices].sum(axis=1)
        message_denominator += len(groups)
    availability /= availability_denominator
    messages /= message_denominator
    alpha = 0.025
    return (
        CandidateGenerationV2Interval(
            scope=scope,
            metric="availability_delta",
            estimate=point.availability_delta,
            lower_bound=float(np.quantile(availability, alpha)),
            upper_bound=float(np.quantile(availability, 1.0 - alpha)),
            resamples=spec.bootstrap_resamples,
            sampling_unit="messages_within_fixed_profile_strata",
        ),
        CandidateGenerationV2Interval(
            scope=scope,
            metric="message_availability_delta",
            estimate=point.message_availability_delta,
            lower_bound=float(np.quantile(messages, alpha)),
            upper_bound=float(np.quantile(messages, 1.0 - alpha)),
            resamples=spec.bootstrap_resamples,
            sampling_unit="messages_within_fixed_profile_strata",
        ),
    )


def evaluate_candidate_generation_v2(
    *,
    benchmark: GeneratedBenchmark,
    baseline_trials: Sequence[LanguageBenchmarkTrial],
    bank: CandidateBank,
    spec: CandidateGenerationV2Spec,
    primary_language_manifest_sha256: str,
    protocol_sha256: str,
) -> CandidateGenerationV2Result:
    """Generate target-blind candidates, then compare intended spans only post hoc."""

    test_messages = {
        message.message_id: message for message in benchmark.messages[BenchmarkSplit.TEST]
    }
    if set(bank.source_message_ids) & set(test_messages):
        raise ValueError("candidate bank contains test-message provenance")
    if bank.benchmark_source_sha256 != benchmark.source_sha256:
        raise ValueError("candidate bank references a different benchmark")
    generator = TargetBlindContextualGeneratorV2(bank, spec)
    records: list[CandidateGenerationV2Trial] = []
    for baseline in baseline_trials:
        message = test_messages.get(baseline.message_id)
        if message is None:
            raise ValueError("baseline language trial is absent from the benchmark test split")
        expected_context = " ".join(message.target_spans[: baseline.span_index])
        expected_target = message.target_spans[baseline.span_index]
        if (
            baseline.profile_id != message.profile_id
            or baseline.confirmed_context != expected_context
            or baseline.intended_text != expected_target
        ):
            raise ValueError("baseline language trial disagrees with benchmark provenance")
        candidates = generator.generate(
            profile_id=baseline.profile_id,
            confirmed_context=baseline.confirmed_context,
            span_index=baseline.span_index,
        )
        keys = [_normalized(candidate.text) for candidate in candidates]
        target_key = _normalized(baseline.intended_text)
        records.append(
            CandidateGenerationV2Trial(
                trial_id=baseline.trial_id,
                profile_id=baseline.profile_id,
                message_id=baseline.message_id,
                span_index=baseline.span_index,
                message_span_count=baseline.message_span_count,
                confirmed_context=baseline.confirmed_context,
                intended_text=baseline.intended_text,
                baseline_target_available=baseline.target_available,
                candidates=candidates,
                target_rank=(keys.index(target_key) + 1 if target_key in keys else None),
            )
        )
    record_tuple = tuple(records)
    scopes: dict[str, tuple[CandidateGenerationV2Trial, ...]] = {"overall": record_tuple}
    for profile in sorted({record.profile_id for record in record_tuple}):
        scopes[profile] = tuple(record for record in record_tuple if record.profile_id == profile)
    for span_index in sorted({record.span_index for record in record_tuple}):
        scopes[f"span-{span_index}"] = tuple(
            record for record in record_tuple if record.span_index == span_index
        )
    metrics = tuple(
        _message_metrics(
            rows,
            scope,
            include_complete_messages=not scope.startswith("span-"),
        )
        for scope, rows in scopes.items()
    )
    intervals = tuple(
        interval
        for scope, rows in scopes.items()
        if not scope.startswith("span-")
        for interval in _bootstrap_intervals(rows, spec, scope)
    )
    result_payload = {
        "config_sha256": spec.digest(),
        "protocol_sha256": protocol_sha256,
        "primary_language_manifest_sha256": primary_language_manifest_sha256,
        "benchmark_source_sha256": benchmark.source_sha256,
        "candidate_bank_sha256": bank.digest(),
        "trial_fingerprint": _sha256_text(
            _canonical_json(
                [
                    {
                        "trial_id": record.trial_id,
                        "candidate_texts": [candidate.text for candidate in record.candidates],
                    }
                    for record in record_tuple
                ]
            )
        ),
    }
    run_id = f"candidate-generation-v2-{_sha256_text(_canonical_json(result_payload))[:20]}"
    return CandidateGenerationV2Result(
        run_id=run_id,
        generated_at=spec.generated_at,
        config_sha256=spec.digest(),
        protocol_sha256=protocol_sha256,
        primary_language_manifest_sha256=primary_language_manifest_sha256,
        benchmark_source_sha256=benchmark.source_sha256,
        candidate_bank_sha256=bank.digest(),
        fitting_source_splits=spec.fitting_source_splits,
        design_status=spec.design_status,
        trials=record_tuple,
        metrics=metrics,
        intervals=intervals,
        limitations=(
            "This is an exploratory supplement; the benchmark test structure and primary result "
            "were already visible before v2 was implemented.",
            "Candidate-bank fitting reads only train and validation messages, while intended test "
            "spans are used only for post-hoc scoring.",
            "Profile conditioning and grammar routing differ from the frozen generic v1 generator.",
            "Exact-span availability is a strict synthetic metric, not evidence of intended "
            "communication or live BCI performance.",
        ),
    )
