"""Locked Step 4 ablations and held-out-combination robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation.candidate_generation_v2 import (
    CandidateBank,
    CandidateBankEntry,
    CandidateGenerationV2Spec,
    CandidateGenerationV2Trial,
    CandidateRole,
    GeneratedCandidateV2,
    TargetBlindContextualGeneratorV2,
    _normalized,
    _request_like,
    _tokens,
)
from neuroselect.synthetic import BenchmarkSplit, GeneratedBenchmark

DEFAULT_CANDIDATE_GENERATION_STEP4_CONFIG = Path(
    "configs/publication/candidate_generation_step4.yaml"
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class CandidateGenerationMethod(StrEnum):
    FULL_V2 = "full_v2"
    NO_PROFILE_CONDITIONING = "no_profile_conditioning"
    NO_GRAMMAR_ROUTING = "no_grammar_routing"
    FREQUENCY_ONLY = "frequency_only"
    TWO_STAGE_OPENING = "two_stage_opening"


class CandidateGenerationDataset(StrEnum):
    EXISTING_EXPOSED = "existing_exposed"
    ROBUSTNESS_HOLDOUT = "robustness_holdout"


class CandidateGenerationStep4Spec(BaseModel):
    """Checksum-pinned Step 4 protocol frozen before executing the analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal["candidate-generation-step4-v1"]
    locked_at: datetime
    publication_protocol: Path
    expected_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step3_artifacts: Path
    expected_step3_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_step3_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robustness_benchmark_spec: Path
    profiles_directory: Path
    expected_robustness_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    methods: tuple[CandidateGenerationMethod, ...]
    language_candidate_count: Literal[9] = 9
    maximum_phrase_tokens: Literal[4] = 4
    bootstrap_resamples: int = Field(default=10_000, ge=2_000, le=100_000)
    bootstrap_seed: int = Field(default=20260728, ge=0)
    design_status: Literal["locked_before_execution_exploratory"]
    outcome_based_omission_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_protocol(self) -> CandidateGenerationStep4Spec:
        if self.locked_at.tzinfo is None or self.locked_at.utcoffset() is None:
            raise ValueError("Step 4 lock time must include a timezone")
        if self.methods != tuple(CandidateGenerationMethod):
            raise ValueError("Step 4 methods and their order are locked")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


def load_candidate_generation_step4_spec(
    path: str | Path = DEFAULT_CANDIDATE_GENERATION_STEP4_CONFIG,
) -> CandidateGenerationStep4Spec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("Step 4 config must contain a YAML mapping")
    return CandidateGenerationStep4Spec.model_validate(payload)


@dataclass(frozen=True)
class EvaluationSpan:
    trial_id: str
    profile_id: str
    message_id: str
    span_index: int
    message_span_count: int
    confirmed_context: str
    intended_text: str


def existing_evaluation_spans(
    trials: Sequence[CandidateGenerationV2Trial],
) -> tuple[EvaluationSpan, ...]:
    return tuple(
        EvaluationSpan(
            trial_id=trial.trial_id,
            profile_id=trial.profile_id,
            message_id=trial.message_id,
            span_index=trial.span_index,
            message_span_count=trial.message_span_count,
            confirmed_context=trial.confirmed_context,
            intended_text=trial.intended_text,
        )
        for trial in trials
    )


def robustness_evaluation_spans(benchmark: GeneratedBenchmark) -> tuple[EvaluationSpan, ...]:
    records: list[EvaluationSpan] = []
    for message in benchmark.messages[BenchmarkSplit.TEST]:
        confirmed: list[str] = []
        for span_index, intended_text in enumerate(message.target_spans):
            records.append(
                EvaluationSpan(
                    trial_id=f"step4-{message.message_id}-{span_index:02d}",
                    profile_id=message.profile_id,
                    message_id=message.message_id,
                    span_index=span_index,
                    message_span_count=len(message.target_spans),
                    confirmed_context=" ".join(confirmed),
                    intended_text=intended_text,
                )
            )
            confirmed.append(intended_text)
    return tuple(records)


class AblatedCandidateGenerator:
    """Target-blind generator implementing one prespecified v2 ablation."""

    def __init__(
        self,
        bank: CandidateBank,
        spec: CandidateGenerationV2Spec,
        method: CandidateGenerationMethod,
    ) -> None:
        if method not in {
            CandidateGenerationMethod.NO_PROFILE_CONDITIONING,
            CandidateGenerationMethod.NO_GRAMMAR_ROUTING,
            CandidateGenerationMethod.FREQUENCY_ONLY,
        }:
            raise ValueError(f"unsupported ablation method: {method}")
        self.bank = bank
        self.spec = spec
        self.method = method
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
        if self.method is CandidateGenerationMethod.FREQUENCY_ONLY:
            return math.log1p(entry.occurrence_count)
        similarity = max(
            (
                self._context_similarity(confirmed_context, source_context)
                for source_context in entry.source_contexts
            ),
            default=0.0,
        )
        profile_count = (
            0
            if self.method is CandidateGenerationMethod.NO_PROFILE_CONDITIONING
            else entry.profile_counts.get(profile_id, 0)
        )
        return (
            10.0 * similarity + 2.0 * math.log1p(profile_count) + math.log1p(entry.occurrence_count)
        )

    def _rank(
        self,
        entries: Sequence[CandidateBankEntry],
        *,
        profile_id: str,
        confirmed_context: str,
        predicate: Any = None,
    ) -> list[tuple[CandidateBankEntry, float]]:
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
            key=lambda item: (-item[1], item[0].role.value, item[0].normalized_text),
        )

    def _ends_with_role(self, context: str, role: CandidateRole) -> bool:
        normalized = _normalized(context)
        return any(normalized.endswith(entry.normalized_text) for entry in self._by_role[role])

    def _plans(self, confirmed_context: str) -> list[tuple[CandidateRole, int, Any]]:
        if not _normalized(confirmed_context):
            return [(CandidateRole.OPENING, self.spec.language_candidate_count, None)]
        if self._ends_with_role(confirmed_context, CandidateRole.TIME_QUALIFIER):
            return [(CandidateRole.REQUEST_ENDING, self.spec.language_candidate_count, None)]
        if self._ends_with_role(confirmed_context, CandidateRole.OBJECT):
            return [
                (
                    CandidateRole.TIME_QUALIFIER,
                    self.spec.request_object_time_quota,
                    lambda entry: entry.normalized_text.startswith("before "),
                ),
                (CandidateRole.LOCATION, self.spec.request_object_location_quota, None),
                (CandidateRole.REQUEST_ENDING, self.spec.request_object_ending_quota, None),
            ]
        if _request_like(confirmed_context):
            return [(CandidateRole.OBJECT, self.spec.language_candidate_count, None)]
        return [
            (CandidateRole.OBJECT, 4, None),
            (CandidateRole.REQUEST_ENDING, 2, None),
            (CandidateRole.TIME_QUALIFIER, 2, None),
            (CandidateRole.ENDING, 1, None),
        ]

    def generate(
        self,
        *,
        profile_id: str,
        confirmed_context: str,
        span_index: int,
    ) -> tuple[GeneratedCandidateV2, ...]:
        """Return candidates without accepting an intended span or target."""

        key = (profile_id, _normalized(confirmed_context), span_index)
        if key in self._cache:
            return self._cache[key]

        selected: list[tuple[CandidateBankEntry, float]] = []
        seen: set[str] = set()
        if self.method is CandidateGenerationMethod.NO_GRAMMAR_ROUTING:
            plans: list[tuple[CandidateRole, int, Any]] = []
        else:
            plans = self._plans(confirmed_context)
        for role, quota, predicate in plans:
            added = 0
            for entry, score in self._rank(
                self._by_role[role],
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
            for entry, score in self._rank(
                self.bank.entries,
                profile_id=profile_id,
                confirmed_context=confirmed_context,
            ):
                if entry.normalized_text in seen:
                    continue
                selected.append((entry, score))
                seen.add(entry.normalized_text)
                if len(selected) == self.spec.language_candidate_count:
                    break
        if len(selected) != self.spec.language_candidate_count:
            raise ValueError("candidate bank cannot fill the visible candidate quota")

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


@dataclass(frozen=True)
class _OpeningComponent:
    text: str
    normalized_text: str
    occurrence_count: int
    profile_counts: Counter[str]


class TargetBlindTwoStageOpeningGenerator:
    """Compose an opening from a selected stem and a second action menu."""

    def __init__(self, bank: CandidateBank, candidate_count: int = 9) -> None:
        self.candidate_count = candidate_count
        stem_aggregates: dict[str, dict[str, Any]] = {}
        action_aggregates: dict[str, dict[str, Any]] = {}
        joint_counts: Counter[tuple[str, str]] = Counter()
        for entry in bank.entries:
            if entry.role is not CandidateRole.OPENING:
                continue
            words = entry.text.split()
            if len(words) < 2:
                continue
            stem_text = " ".join(words[:-1])
            action_text = words[-1]
            stem_key = _normalized(stem_text)
            action_key = _normalized(action_text)
            stem = stem_aggregates.setdefault(
                stem_key,
                {"text": stem_text, "count": 0, "profiles": Counter()},
            )
            action = action_aggregates.setdefault(
                action_key,
                {"text": action_text, "count": 0, "profiles": Counter()},
            )
            stem["count"] += entry.occurrence_count
            action["count"] += entry.occurrence_count
            stem["profiles"].update(entry.profile_counts)
            action["profiles"].update(entry.profile_counts)
            joint_counts[(stem_key, action_key)] += entry.occurrence_count
        if not stem_aggregates or not action_aggregates:
            raise ValueError("opening bank cannot fill the two-stage candidate menus")
        self._stems = tuple(
            _OpeningComponent(
                text=value["text"],
                normalized_text=key,
                occurrence_count=value["count"],
                profile_counts=value["profiles"],
            )
            for key, value in sorted(stem_aggregates.items())
        )
        self._actions = tuple(
            _OpeningComponent(
                text=value["text"],
                normalized_text=key,
                occurrence_count=value["count"],
                profile_counts=value["profiles"],
            )
            for key, value in sorted(action_aggregates.items())
        )
        self._joint_counts = joint_counts

    @staticmethod
    def _component_score(component: _OpeningComponent, profile_id: str) -> float:
        return 2.0 * math.log1p(component.profile_counts.get(profile_id, 0)) + math.log1p(
            component.occurrence_count
        )

    def generate_stems(self, *, profile_id: str) -> tuple[str, ...]:
        """Generate the first menu from the profile only."""

        ranked = sorted(
            self._stems,
            key=lambda item: (
                -self._component_score(item, profile_id),
                item.normalized_text,
            ),
        )
        return tuple(item.text for item in ranked[: self.candidate_count])

    def generate_actions(
        self,
        *,
        profile_id: str,
        selected_stem: str,
    ) -> tuple[str, ...]:
        """Generate menu two from the observed user selection, never a target action."""

        stem_key = _normalized(selected_stem)
        ranked = sorted(
            self._actions,
            key=lambda item: (
                -(
                    4.0 * math.log1p(self._joint_counts[(stem_key, item.normalized_text)])
                    + self._component_score(item, profile_id)
                ),
                item.normalized_text,
            ),
        )
        return tuple(item.text for item in ranked[: self.candidate_count])


class CandidateGenerationStep4Trial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: CandidateGenerationDataset
    method: CandidateGenerationMethod
    trial_id: str
    profile_id: str
    message_id: str
    span_index: int = Field(ge=0)
    message_span_count: int = Field(ge=1)
    confirmed_context: str
    intended_text: str
    selection_stages: Literal[1, 2]
    stage_one_target_text: str
    stage_one_candidates: tuple[str, ...] = Field(min_length=1, max_length=9)
    stage_one_target_rank: int | None = Field(default=None, ge=1, le=9)
    stage_two_target_text: str | None = None
    stage_two_candidates: tuple[str, ...] = Field(default=())
    stage_two_target_rank: int | None = Field(default=None, ge=1, le=9)
    target_available: bool

    @model_validator(mode="after")
    def validate_posthoc_scoring(self) -> CandidateGenerationStep4Trial:
        if len({_normalized(value) for value in self.stage_one_candidates}) != len(
            self.stage_one_candidates
        ):
            raise ValueError("stage-one candidates must be unique")
        expected_one = _rank(self.stage_one_candidates, self.stage_one_target_text)
        if self.stage_one_target_rank != expected_one:
            raise ValueError("stage-one target rank must be scored post hoc")
        if self.selection_stages == 1:
            if (
                self.stage_two_target_text is not None
                or self.stage_two_candidates
                or self.stage_two_target_rank is not None
            ):
                raise ValueError("single-stage trials cannot contain stage-two data")
            expected_available = expected_one is not None
        else:
            if self.stage_two_target_text is None:
                raise ValueError("two-stage trials require a stage-two target component")
            if self.stage_two_candidates and not 1 <= len(self.stage_two_candidates) <= 9:
                raise ValueError("generated stage-two menus must contain at most nine candidates")
            expected_two = (
                _rank(self.stage_two_candidates, self.stage_two_target_text)
                if self.stage_two_candidates
                else None
            )
            if self.stage_two_target_rank != expected_two:
                raise ValueError("stage-two target rank must be scored post hoc")
            expected_available = expected_one is not None and expected_two is not None
        if self.target_available != expected_available:
            raise ValueError("target availability must agree with posthoc component ranks")
        return self


def _rank(candidates: Sequence[str], target: str) -> int | None:
    keys = [_normalized(value) for value in candidates]
    target_key = _normalized(target)
    return keys.index(target_key) + 1 if target_key in keys else None


class CandidateGenerationStep4Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: CandidateGenerationDataset
    method: CandidateGenerationMethod
    scope: str
    unit: Literal["span", "message"]
    available_count: int = Field(ge=0)
    denominator: int = Field(ge=1)
    availability_rate: float = Field(ge=0.0, le=1.0)
    mean_selection_stages: float | None = Field(default=None, ge=1.0, le=2.0)


class CandidateGenerationStep4Contrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: CandidateGenerationDataset
    contrast_id: str
    scope: Literal["overall", "opening", "complete_messages"]
    reference_method: CandidateGenerationMethod
    comparator_method: CandidateGenerationMethod
    estimate: float = Field(ge=-1.0, le=1.0)
    lower_bound: float = Field(ge=-1.0, le=1.0)
    upper_bound: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    resamples: int = Field(ge=2_000)
    sampling_unit: Literal["messages_within_fixed_profile_strata"]


class CandidateGenerationStep4Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step3_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    existing_candidate_bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robustness_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robustness_candidate_bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robustness_opening_overlap_count: Literal[0] = 0
    intended_target_exposed_to_generators: Literal[False] = False
    action_stage_conditioning: Literal["observed_selected_stem_only"]
    design_status: Literal["locked_before_execution_exploratory"]
    trials: tuple[CandidateGenerationStep4Trial, ...] = Field(min_length=1)
    metrics: tuple[CandidateGenerationStep4Metric, ...] = Field(min_length=1)
    contrasts: tuple[CandidateGenerationStep4Contrast, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def validate_robustness_opening_holdout(benchmark: GeneratedBenchmark) -> None:
    fit = tuple(
        message
        for split in (BenchmarkSplit.TRAIN, BenchmarkSplit.VALIDATION)
        for message in benchmark.messages[split]
    )
    test = benchmark.messages[BenchmarkSplit.TEST]
    fit_openings = {_normalized(message.target_spans[0]) for message in fit}
    test_openings = {_normalized(message.target_spans[0]) for message in test}
    if fit_openings & test_openings:
        raise ValueError("robustness test opening combinations overlap fitting data")

    def components(openings: set[str]) -> tuple[set[str], set[str]]:
        stems: set[str] = set()
        actions: set[str] = set()
        for opening in openings:
            words = opening.split()
            if len(words) < 2:
                raise ValueError("robustness openings must have a stem and action")
            stems.add(" ".join(words[:-1]))
            actions.add(words[-1])
        return stems, actions

    fit_stems, fit_actions = components(fit_openings)
    test_stems, test_actions = components(test_openings)
    if not test_stems <= fit_stems or not test_actions <= fit_actions:
        raise ValueError("robustness test components must each be observable in fitting data")


def _trial(
    *,
    dataset_id: CandidateGenerationDataset,
    method: CandidateGenerationMethod,
    item: EvaluationSpan,
    candidate_texts: tuple[str, ...],
) -> CandidateGenerationStep4Trial:
    return CandidateGenerationStep4Trial(
        dataset_id=dataset_id,
        method=method,
        trial_id=item.trial_id,
        profile_id=item.profile_id,
        message_id=item.message_id,
        span_index=item.span_index,
        message_span_count=item.message_span_count,
        confirmed_context=item.confirmed_context,
        intended_text=item.intended_text,
        selection_stages=1,
        stage_one_target_text=item.intended_text,
        stage_one_candidates=candidate_texts,
        stage_one_target_rank=_rank(candidate_texts, item.intended_text),
        target_available=_rank(candidate_texts, item.intended_text) is not None,
    )


def _two_stage_trial(
    *,
    dataset_id: CandidateGenerationDataset,
    item: EvaluationSpan,
    opening_generator: TargetBlindTwoStageOpeningGenerator,
) -> CandidateGenerationStep4Trial:
    words = item.intended_text.split()
    if len(words) < 2:
        raise ValueError("two-stage opening targets require a stem and final action")
    target_stem = " ".join(words[:-1])
    target_action = words[-1]
    stems = opening_generator.generate_stems(profile_id=item.profile_id)
    stem_rank = _rank(stems, target_stem)
    actions = (
        opening_generator.generate_actions(
            profile_id=item.profile_id,
            selected_stem=target_stem,
        )
        if stem_rank is not None
        else ()
    )
    action_rank = _rank(actions, target_action) if actions else None
    return CandidateGenerationStep4Trial(
        dataset_id=dataset_id,
        method=CandidateGenerationMethod.TWO_STAGE_OPENING,
        trial_id=item.trial_id,
        profile_id=item.profile_id,
        message_id=item.message_id,
        span_index=item.span_index,
        message_span_count=item.message_span_count,
        confirmed_context=item.confirmed_context,
        intended_text=item.intended_text,
        selection_stages=2,
        stage_one_target_text=target_stem,
        stage_one_candidates=stems,
        stage_one_target_rank=stem_rank,
        stage_two_target_text=target_action,
        stage_two_candidates=actions,
        stage_two_target_rank=action_rank,
        target_available=stem_rank is not None and action_rank is not None,
    )


def _metrics(
    trials: Sequence[CandidateGenerationStep4Trial],
) -> tuple[CandidateGenerationStep4Metric, ...]:
    records: list[CandidateGenerationStep4Metric] = []
    datasets = tuple(CandidateGenerationDataset)
    methods = tuple(CandidateGenerationMethod)
    for dataset_id in datasets:
        for method in methods:
            rows = tuple(
                trial
                for trial in trials
                if trial.dataset_id is dataset_id and trial.method is method
            )
            if not rows:
                raise ValueError(f"missing Step 4 trials for {dataset_id}/{method}")
            span_scopes = {
                "overall": rows,
                "opening": tuple(row for row in rows if row.span_index == 0),
                "later": tuple(row for row in rows if row.span_index > 0),
            }
            for profile_id in sorted({row.profile_id for row in rows}):
                span_scopes[f"profile:{profile_id}"] = tuple(
                    row for row in rows if row.profile_id == profile_id
                )
            for scope, selected in span_scopes.items():
                available = sum(row.target_available for row in selected)
                records.append(
                    CandidateGenerationStep4Metric(
                        dataset_id=dataset_id,
                        method=method,
                        scope=scope,
                        unit="span",
                        available_count=available,
                        denominator=len(selected),
                        availability_rate=available / len(selected),
                        mean_selection_stages=float(
                            np.mean([row.selection_stages for row in selected])
                        ),
                    )
                )
            by_message: dict[tuple[str, str], list[CandidateGenerationStep4Trial]] = defaultdict(
                list
            )
            for row in rows:
                by_message[(row.profile_id, row.message_id)].append(row)
            for message_profile_id in (None, *sorted({key[0] for key in by_message})):
                groups = tuple(
                    group
                    for (group_profile, _), group in by_message.items()
                    if message_profile_id is None or group_profile == message_profile_id
                )
                available = sum(all(row.target_available for row in group) for group in groups)
                records.append(
                    CandidateGenerationStep4Metric(
                        dataset_id=dataset_id,
                        method=method,
                        scope=(
                            "complete_messages"
                            if message_profile_id is None
                            else f"complete_messages:{message_profile_id}"
                        ),
                        unit="message",
                        available_count=available,
                        denominator=len(groups),
                        availability_rate=available / len(groups),
                    )
                )
    return tuple(records)


def _scope_value(
    rows: Sequence[CandidateGenerationStep4Trial],
    scope: str,
) -> tuple[float, float]:
    if scope == "overall":
        return float(sum(row.target_available for row in rows)), float(len(rows))
    if scope == "opening":
        opening = tuple(row for row in rows if row.span_index == 0)
        return float(sum(row.target_available for row in opening)), float(len(opening))
    if scope == "complete_messages":
        return float(all(row.target_available for row in rows)), 1.0
    raise ValueError(f"unsupported bootstrap scope: {scope}")


def _contrast(
    *,
    dataset_id: CandidateGenerationDataset,
    reference_method: CandidateGenerationMethod,
    comparator_method: CandidateGenerationMethod,
    scope: Literal["overall", "opening", "complete_messages"],
    trials: Sequence[CandidateGenerationStep4Trial],
    spec: CandidateGenerationStep4Spec,
) -> CandidateGenerationStep4Contrast:
    by_method_message: dict[
        CandidateGenerationMethod,
        dict[tuple[str, str], tuple[CandidateGenerationStep4Trial, ...]],
    ] = {}
    for method in (reference_method, comparator_method):
        groups: dict[tuple[str, str], list[CandidateGenerationStep4Trial]] = defaultdict(list)
        for trial in trials:
            if trial.dataset_id is dataset_id and trial.method is method:
                groups[(trial.profile_id, trial.message_id)].append(trial)
        by_method_message[method] = {
            key: tuple(sorted(rows, key=lambda row: row.span_index)) for key, rows in groups.items()
        }
    reference = by_method_message[reference_method]
    comparator = by_method_message[comparator_method]
    if set(reference) != set(comparator):
        raise ValueError("Step 4 contrast methods do not contain paired messages")

    by_profile: dict[str, list[tuple[float, float]]] = defaultdict(list)
    total_delta = 0.0
    total_denominator = 0.0
    for key in sorted(reference):
        ref_numerator, ref_denominator = _scope_value(reference[key], scope)
        cmp_numerator, cmp_denominator = _scope_value(comparator[key], scope)
        if ref_denominator != cmp_denominator:
            raise ValueError("paired Step 4 methods have different denominators")
        delta = ref_numerator - cmp_numerator
        by_profile[key[0]].append((delta, ref_denominator))
        total_delta += delta
        total_denominator += ref_denominator
    point = total_delta / total_denominator

    seed_material = f"{dataset_id.value}:{reference_method.value}:{comparator_method.value}:{scope}"
    seed_offset = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big")
    rng = np.random.default_rng(spec.bootstrap_seed + seed_offset)
    numerators = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    denominators = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    for profile_groups in by_profile.values():
        deltas = np.asarray([group[0] for group in profile_groups], dtype=np.float64)
        weights = np.asarray([group[1] for group in profile_groups], dtype=np.float64)
        for start in range(0, spec.bootstrap_resamples, 1_000):
            stop = min(start + 1_000, spec.bootstrap_resamples)
            indices = rng.integers(
                0,
                len(profile_groups),
                size=(stop - start, len(profile_groups)),
            )
            numerators[start:stop] += deltas[indices].sum(axis=1)
            denominators[start:stop] += weights[indices].sum(axis=1)
    samples = numerators / denominators
    return CandidateGenerationStep4Contrast(
        dataset_id=dataset_id,
        contrast_id=(f"{reference_method.value}-minus-{comparator_method.value}-{scope}"),
        scope=scope,
        reference_method=reference_method,
        comparator_method=comparator_method,
        estimate=point,
        lower_bound=float(np.quantile(samples, 0.025)),
        upper_bound=float(np.quantile(samples, 0.975)),
        resamples=spec.bootstrap_resamples,
        sampling_unit="messages_within_fixed_profile_strata",
    )


def evaluate_candidate_generation_step4(
    *,
    spec: CandidateGenerationStep4Spec,
    v2_spec: CandidateGenerationV2Spec,
    existing_spans: Sequence[EvaluationSpan],
    existing_bank: CandidateBank,
    robustness_spans: Sequence[EvaluationSpan],
    robustness_bank: CandidateBank,
    protocol_sha256: str,
    step3_manifest_sha256: str,
) -> CandidateGenerationStep4Result:
    """Run all locked methods; intended targets are used only for posthoc scoring."""

    all_trials: list[CandidateGenerationStep4Trial] = []
    inputs = (
        (
            CandidateGenerationDataset.EXISTING_EXPOSED,
            existing_spans,
            existing_bank,
        ),
        (
            CandidateGenerationDataset.ROBUSTNESS_HOLDOUT,
            robustness_spans,
            robustness_bank,
        ),
    )
    for dataset_id, spans, bank in inputs:
        full = TargetBlindContextualGeneratorV2(bank, v2_spec)
        ablations = {
            method: AblatedCandidateGenerator(bank, v2_spec, method)
            for method in (
                CandidateGenerationMethod.NO_PROFILE_CONDITIONING,
                CandidateGenerationMethod.NO_GRAMMAR_ROUTING,
                CandidateGenerationMethod.FREQUENCY_ONLY,
            )
        }
        opening = TargetBlindTwoStageOpeningGenerator(bank, spec.language_candidate_count)
        for item in spans:
            full_candidates = full.generate(
                profile_id=item.profile_id,
                confirmed_context=item.confirmed_context,
                span_index=item.span_index,
            )
            all_trials.append(
                _trial(
                    dataset_id=dataset_id,
                    method=CandidateGenerationMethod.FULL_V2,
                    item=item,
                    candidate_texts=tuple(candidate.text for candidate in full_candidates),
                )
            )
            for method, generator in ablations.items():
                candidates = generator.generate(
                    profile_id=item.profile_id,
                    confirmed_context=item.confirmed_context,
                    span_index=item.span_index,
                )
                all_trials.append(
                    _trial(
                        dataset_id=dataset_id,
                        method=method,
                        item=item,
                        candidate_texts=tuple(candidate.text for candidate in candidates),
                    )
                )
            if item.span_index == 0:
                all_trials.append(
                    _two_stage_trial(
                        dataset_id=dataset_id,
                        item=item,
                        opening_generator=opening,
                    )
                )
            else:
                all_trials.append(
                    _trial(
                        dataset_id=dataset_id,
                        method=CandidateGenerationMethod.TWO_STAGE_OPENING,
                        item=item,
                        candidate_texts=tuple(candidate.text for candidate in full_candidates),
                    )
                )

    trials = tuple(all_trials)
    metrics = _metrics(trials)
    contrasts: list[CandidateGenerationStep4Contrast] = []
    comparison_pairs = (
        (
            CandidateGenerationMethod.FULL_V2,
            CandidateGenerationMethod.NO_PROFILE_CONDITIONING,
        ),
        (
            CandidateGenerationMethod.FULL_V2,
            CandidateGenerationMethod.NO_GRAMMAR_ROUTING,
        ),
        (
            CandidateGenerationMethod.FULL_V2,
            CandidateGenerationMethod.FREQUENCY_ONLY,
        ),
        (
            CandidateGenerationMethod.TWO_STAGE_OPENING,
            CandidateGenerationMethod.FULL_V2,
        ),
    )
    for dataset_id in CandidateGenerationDataset:
        for reference_method, comparator_method in comparison_pairs:
            for scope in ("overall", "opening", "complete_messages"):
                contrasts.append(
                    _contrast(
                        dataset_id=dataset_id,
                        reference_method=reference_method,
                        comparator_method=comparator_method,
                        scope=scope,
                        trials=trials,
                        spec=spec,
                    )
                )

    fingerprint = _sha256_text(
        _canonical_json(
            [
                {
                    "dataset_id": trial.dataset_id,
                    "method": trial.method,
                    "trial_id": trial.trial_id,
                    "target_available": trial.target_available,
                    "stage_one_target_rank": trial.stage_one_target_rank,
                    "stage_two_target_rank": trial.stage_two_target_rank,
                }
                for trial in trials
            ]
        )
    )
    payload = {
        "config_sha256": spec.digest(),
        "protocol_sha256": protocol_sha256,
        "step3_manifest_sha256": step3_manifest_sha256,
        "existing_candidate_bank_sha256": existing_bank.digest(),
        "robustness_source_sha256": robustness_bank.benchmark_source_sha256,
        "robustness_candidate_bank_sha256": robustness_bank.digest(),
        "trial_fingerprint": fingerprint,
    }
    return CandidateGenerationStep4Result(
        run_id=f"candidate-step4-{_sha256_text(_canonical_json(payload))[:20]}",
        generated_at=spec.locked_at,
        config_sha256=spec.digest(),
        protocol_sha256=protocol_sha256,
        step3_manifest_sha256=step3_manifest_sha256,
        existing_candidate_bank_sha256=existing_bank.digest(),
        robustness_source_sha256=robustness_bank.benchmark_source_sha256,
        robustness_candidate_bank_sha256=robustness_bank.digest(),
        robustness_opening_overlap_count=0,
        intended_target_exposed_to_generators=False,
        action_stage_conditioning="observed_selected_stem_only",
        design_status=spec.design_status,
        trials=trials,
        metrics=metrics,
        contrasts=tuple(contrasts),
        limitations=(
            "Both language datasets are synthetic; no live participant communication is tested.",
            "The existing benchmark remains test-exposed exploratory evidence.",
            "The robustness benchmark was developer-authored and locked before execution, not "
            "independently preregistered or externally collected.",
            "Two-stage opening availability requires an extra BCI selection and uses a "
            "teacher-forced observed stem selection during offline replay.",
            "Availability measures candidate-set coverage, not end-to-end communication benefit.",
        ),
    )
