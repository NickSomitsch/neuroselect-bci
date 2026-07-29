"""Target-blind hierarchical opening generalization with explicit selection cost."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_OPENING_GENERALIZATION_CONFIG = Path("configs/publication/opening_generalization_v1.yaml")
TestChallenge = Literal["heldout_combination", "heldout_paraphrase_family"]
ContrastMetric = Literal["availability_rate", "coverage_per_required_selection"]


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(".,!?;:")


def _rank(candidates: Sequence[str], target: str) -> int | None:
    keys = [_normalized(candidate) for candidate in candidates]
    target_key = _normalized(target)
    return keys.index(target_key) + 1 if target_key in keys else None


class OpeningIntent(StrEnum):
    REQUEST = "request"
    PREFERENCE = "preference"
    CLARIFICATION = "clarification"
    STATUS = "status"


class OpeningSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class OpeningChallenge(StrEnum):
    FITTING = "fitting"
    HELDOUT_COMBINATION = "heldout_combination"
    HELDOUT_PARAPHRASE_FAMILY = "heldout_paraphrase_family"


class OpeningMethod(StrEnum):
    ONE_STAGE_PHRASE = "one_stage_phrase"
    TWO_STAGE_STEM_CONTENT = "two_stage_stem_content"
    THREE_STAGE_INTENT_STEM_CONTENT = "three_stage_intent_stem_content"


class PairPartitionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modulus: Literal[12] = 12
    train_residues: tuple[int, ...]
    validation_residues: tuple[int, ...]
    combination_test_residues: tuple[int, ...]

    @model_validator(mode="after")
    def validate_partition(self) -> PairPartitionSpec:
        partitions = (
            self.train_residues,
            self.validation_residues,
            self.combination_test_residues,
        )
        flattened = tuple(value for partition in partitions for value in partition)
        if sorted(flattened) != list(range(self.modulus)):
            raise ValueError("opening pair residues must partition the complete modulus")
        if any(not partition for partition in partitions):
            raise ValueError("every opening pair partition must be non-empty")
        return self


class OpeningIntentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: OpeningIntent
    fitted_stems: tuple[str, ...] = Field(min_length=6)
    heldout_family_stems: tuple[str, ...] = Field(min_length=2)
    contents: tuple[str, ...] = Field(min_length=12)

    @model_validator(mode="after")
    def validate_vocabulary(self) -> OpeningIntentSpec:
        collections = (self.fitted_stems, self.heldout_family_stems, self.contents)
        for values in collections:
            if len({_normalized(value) for value in values}) != len(values):
                raise ValueError("opening source vocabulary must be unique")
        if {_normalized(value) for value in self.fitted_stems} & {
            _normalized(value) for value in self.heldout_family_stems
        }:
            raise ValueError("held-out family stems cannot occur in fitted stems")
        return self


class OpeningGeneralizationSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["neuroselect-opening-generalization-v1"]
    revision: Literal["opening-generalization-v1"]
    seed: int = Field(ge=0)
    profile_ids: tuple[str, ...] = Field(min_length=4)
    candidate_budget: Literal[9] = 9
    maximum_opening_tokens: int = Field(default=5, ge=2, le=8)
    pair_partition: PairPartitionSpec
    intents: tuple[OpeningIntentSpec, ...]

    @model_validator(mode="after")
    def validate_source(self) -> OpeningGeneralizationSource:
        if tuple(intent.intent_id for intent in self.intents) != tuple(OpeningIntent):
            raise ValueError("opening intents and their order are locked")
        if len(set(self.profile_ids)) != len(self.profile_ids):
            raise ValueError("opening profile IDs must be unique")
        all_stems = [
            _normalized(stem)
            for intent in self.intents
            for stem in (*intent.fitted_stems, *intent.heldout_family_stems)
        ]
        all_contents = [
            _normalized(content) for intent in self.intents for content in intent.contents
        ]
        if len(all_stems) != len(set(all_stems)):
            raise ValueError("opening stems must be unique across intents")
        if len(all_contents) != len(set(all_contents)):
            raise ValueError("opening contents must be unique across intents")
        for intent in self.intents:
            openings = (
                f"{stem} {content}"
                for stem in (*intent.fitted_stems, *intent.heldout_family_stems)
                for content in intent.contents
            )
            if any(len(opening.split()) > self.maximum_opening_tokens for opening in openings):
                raise ValueError("opening source exceeds the maximum opening token count")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def load_opening_generalization_source(path: str | Path) -> OpeningGeneralizationSource:
    with Path(path).open(encoding="utf-8") as source_file:
        payload: Any = yaml.safe_load(source_file)
    if not isinstance(payload, dict):
        raise ValueError("opening-generalization source must contain a YAML mapping")
    return OpeningGeneralizationSource.model_validate(payload)


class OpeningGeneralizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str = Field(min_length=1, max_length=160)
    protocol_revision: Literal["opening-generalization-experiment-v1"]
    locked_at: datetime
    publication_protocol: Path
    expected_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step4_artifacts: Path
    expected_step4_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source: Path
    expected_benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    methods: tuple[OpeningMethod, ...]
    candidate_budget: Literal[9] = 9
    bootstrap_resamples: int = Field(default=10_000, ge=2_000, le=100_000)
    bootstrap_seed: int = Field(default=20260728, ge=0)
    design_status: Literal["locked_before_execution_exploratory"]
    outcome_based_omission_forbidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_protocol(self) -> OpeningGeneralizationSpec:
        if self.locked_at.tzinfo is None or self.locked_at.utcoffset() is None:
            raise ValueError("opening-generalization lock time must include a timezone")
        if self.methods != tuple(OpeningMethod):
            raise ValueError("opening-generalization methods and order are locked")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def load_opening_generalization_spec(
    path: str | Path = DEFAULT_OPENING_GENERALIZATION_CONFIG,
) -> OpeningGeneralizationSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("opening-generalization config must contain a YAML mapping")
    return OpeningGeneralizationSpec.model_validate(payload)


class OpeningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(pattern=r"^opening-[0-9a-f]{20}$")
    profile_id: str
    split: OpeningSplit
    challenge: OpeningChallenge
    intent_id: OpeningIntent
    stem: str
    content: str
    opening: str

    @model_validator(mode="after")
    def validate_opening(self) -> OpeningRecord:
        if self.opening != f"{self.stem} {self.content}":
            raise ValueError("opening text must be the exact stem-content composition")
        if self.split is OpeningSplit.TEST and self.challenge is OpeningChallenge.FITTING:
            raise ValueError("test openings require a held-out challenge label")
        if self.split is not OpeningSplit.TEST and self.challenge is not OpeningChallenge.FITTING:
            raise ValueError("fitting openings cannot carry a test challenge label")
        return self


def _opening_record(
    *,
    profile_id: str,
    split: OpeningSplit,
    challenge: OpeningChallenge,
    intent_id: OpeningIntent,
    stem: str,
    content: str,
) -> OpeningRecord:
    opening = f"{stem} {content}"
    material = f"{profile_id}\0{split.value}\0{challenge.value}\0{intent_id.value}\0{opening}"
    return OpeningRecord(
        record_id=f"opening-{hashlib.sha256(material.encode()).hexdigest()[:20]}",
        profile_id=profile_id,
        split=split,
        challenge=challenge,
        intent_id=intent_id,
        stem=stem,
        content=content,
        opening=opening,
    )


def generate_opening_records(
    source: OpeningGeneralizationSource,
) -> tuple[OpeningRecord, ...]:
    records: list[OpeningRecord] = []
    partition = source.pair_partition
    split_by_residue = {
        **dict.fromkeys(partition.train_residues, OpeningSplit.TRAIN),
        **dict.fromkeys(partition.validation_residues, OpeningSplit.VALIDATION),
        **dict.fromkeys(partition.combination_test_residues, OpeningSplit.TEST),
    }
    for profile_id in source.profile_ids:
        for intent in source.intents:
            for stem_index, stem in enumerate(intent.fitted_stems):
                for content_index, content in enumerate(intent.contents):
                    residue = (content_index - 2 * stem_index) % partition.modulus
                    split = split_by_residue[residue]
                    records.append(
                        _opening_record(
                            profile_id=profile_id,
                            split=split,
                            challenge=(
                                OpeningChallenge.HELDOUT_COMBINATION
                                if split is OpeningSplit.TEST
                                else OpeningChallenge.FITTING
                            ),
                            intent_id=intent.intent_id,
                            stem=stem,
                            content=content,
                        )
                    )
            for stem in intent.heldout_family_stems:
                for content in intent.contents:
                    records.append(
                        _opening_record(
                            profile_id=profile_id,
                            split=OpeningSplit.TEST,
                            challenge=OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY,
                            intent_id=intent.intent_id,
                            stem=stem,
                            content=content,
                        )
                    )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.split.value,
                record.challenge.value,
                record.profile_id,
                record.intent_id.value,
                record.opening.casefold(),
            ),
        )
    )


def validate_opening_holdouts(records: Sequence[OpeningRecord]) -> dict[str, int]:
    fit = tuple(record for record in records if record.split is not OpeningSplit.TEST)
    combination = tuple(
        record for record in records if record.challenge is OpeningChallenge.HELDOUT_COMBINATION
    )
    family = tuple(
        record
        for record in records
        if record.challenge is OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY
    )
    fit_openings = {_normalized(record.opening) for record in fit}
    test_openings = {_normalized(record.opening) for record in (*combination, *family)}
    if fit_openings & test_openings:
        raise ValueError("opening test phrases overlap fitted opening phrases")
    fit_stems = {_normalized(record.stem) for record in fit}
    fit_contents = {_normalized(record.content) for record in fit}
    if not {_normalized(record.stem) for record in combination} <= fit_stems:
        raise ValueError("combination-holdout stems must be observed during fitting")
    if not {_normalized(record.content) for record in combination} <= fit_contents:
        raise ValueError("combination-holdout contents must be observed during fitting")
    if {_normalized(record.stem) for record in family} & fit_stems:
        raise ValueError("paraphrase-family stems must be completely absent from fitting")
    if not {_normalized(record.content) for record in family} <= fit_contents:
        raise ValueError("paraphrase-family contents must be observed during fitting")
    return {
        "fit_record_count": len(fit),
        "combination_test_count": len(combination),
        "family_test_count": len(family),
        "fitted_stem_count": len(fit_stems),
        "heldout_family_stem_count": len({_normalized(record.stem) for record in family}),
        "content_count": len(fit_contents),
        "fit_test_opening_overlap_count": 0,
    }


class OpeningSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    profile_id: str
    split: Literal["train", "validation"]


class OpeningBankEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opening: str
    normalized_opening: str
    intent_id: OpeningIntent
    stem: str
    content: str
    occurrence_count: int = Field(ge=1)
    profile_counts: dict[str, int]
    source_refs: tuple[OpeningSourceRef, ...] = Field(min_length=1)


class OpeningTrainingBank(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    revision: Literal["opening-training-bank-v1"] = "opening-training-bank-v1"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_splits: tuple[Literal["train"], Literal["validation"]]
    entries: tuple[OpeningBankEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bank(self) -> OpeningTrainingBank:
        if self.source_splits != ("train", "validation"):
            raise ValueError("opening bank can fit only train and validation records")
        keys = [entry.normalized_opening for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("opening bank phrases must be unique")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def build_opening_training_bank(
    records: Sequence[OpeningRecord],
    source: OpeningGeneralizationSource,
) -> OpeningTrainingBank:
    aggregates: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.split is OpeningSplit.TEST:
            continue
        key = _normalized(record.opening)
        aggregate = aggregates.setdefault(
            key,
            {
                "record": record,
                "profiles": Counter(),
                "refs": [],
            },
        )
        aggregate["profiles"][record.profile_id] += 1
        aggregate["refs"].append(
            OpeningSourceRef(
                record_id=record.record_id,
                profile_id=record.profile_id,
                split=cast(Literal["train", "validation"], record.split.value),
            )
        )
    entries = tuple(
        OpeningBankEntry(
            opening=aggregate["record"].opening,
            normalized_opening=key,
            intent_id=aggregate["record"].intent_id,
            stem=aggregate["record"].stem,
            content=aggregate["record"].content,
            occurrence_count=sum(aggregate["profiles"].values()),
            profile_counts=dict(sorted(aggregate["profiles"].items())),
            source_refs=tuple(
                sorted(
                    aggregate["refs"],
                    key=lambda ref: (ref.split, ref.profile_id, ref.record_id),
                )
            ),
        )
        for key, aggregate in sorted(aggregates.items())
    )
    return OpeningTrainingBank(
        source_sha256=source.digest(),
        source_splits=("train", "validation"),
        entries=entries,
    )


class _Component(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    occurrence_count: int
    profile_counts: dict[str, int]


def _rank_components(
    components: Sequence[_Component],
    profile_id: str,
    budget: int,
) -> tuple[str, ...]:
    ranked = sorted(
        components,
        key=lambda component: (
            -(
                2.0 * math.log1p(component.profile_counts.get(profile_id, 0))
                + math.log1p(component.occurrence_count)
            ),
            _normalized(component.text),
        ),
    )
    return tuple(component.text for component in ranked[:budget])


def _components(
    entries: Sequence[OpeningBankEntry],
    field: Literal["stem", "content"],
) -> tuple[_Component, ...]:
    aggregates: dict[str, dict[str, Any]] = {}
    for entry in entries:
        text = getattr(entry, field)
        aggregate = aggregates.setdefault(
            _normalized(text),
            {"text": text, "count": 0, "profiles": Counter()},
        )
        aggregate["count"] += entry.occurrence_count
        aggregate["profiles"].update(entry.profile_counts)
    return tuple(
        _Component(
            text=value["text"],
            occurrence_count=value["count"],
            profile_counts=dict(value["profiles"]),
        )
        for _, value in sorted(aggregates.items())
    )


class TargetBlindPhraseOpeningGenerator:
    def __init__(self, bank: OpeningTrainingBank, budget: int = 9) -> None:
        self.bank = bank
        self.budget = budget

    def generate(self, *, profile_id: str) -> tuple[str, ...]:
        ranked = sorted(
            self.bank.entries,
            key=lambda entry: (
                -(
                    2.0 * math.log1p(entry.profile_counts.get(profile_id, 0))
                    + math.log1p(entry.occurrence_count)
                ),
                entry.normalized_opening,
            ),
        )
        return tuple(entry.opening for entry in ranked[: self.budget])


class TargetBlindGlobalTwoStageOpeningGenerator:
    def __init__(self, bank: OpeningTrainingBank, budget: int = 9) -> None:
        self.budget = budget
        self._stems = _components(bank.entries, "stem")
        self._intent_by_stem = {_normalized(entry.stem): entry.intent_id for entry in bank.entries}
        self._contents_by_intent = {
            intent: _components(
                tuple(entry for entry in bank.entries if entry.intent_id is intent),
                "content",
            )
            for intent in OpeningIntent
        }

    def generate_stems(self, *, profile_id: str) -> tuple[str, ...]:
        return _rank_components(self._stems, profile_id, self.budget)

    def generate_contents(
        self,
        *,
        profile_id: str,
        selected_stem: str,
    ) -> tuple[str, ...]:
        intent = self._intent_by_stem.get(_normalized(selected_stem))
        if intent is None:
            raise ValueError("selected stem is absent from the fitted opening bank")
        return _rank_components(self._contents_by_intent[intent], profile_id, self.budget)


class TargetBlindIntentThreeStageOpeningGenerator:
    def __init__(self, bank: OpeningTrainingBank, budget: int = 9) -> None:
        self.budget = budget
        self._entries_by_intent = {
            intent: tuple(entry for entry in bank.entries if entry.intent_id is intent)
            for intent in OpeningIntent
        }
        self._intents = tuple(
            _Component(
                text=intent.value,
                occurrence_count=sum(
                    entry.occurrence_count for entry in self._entries_by_intent[intent]
                ),
                profile_counts=dict(
                    sum(
                        (
                            Counter(entry.profile_counts)
                            for entry in self._entries_by_intent[intent]
                        ),
                        Counter(),
                    )
                ),
            )
            for intent in OpeningIntent
        )
        self._stems_by_intent = {
            intent: _components(self._entries_by_intent[intent], "stem") for intent in OpeningIntent
        }
        self._contents_by_intent = {
            intent: _components(self._entries_by_intent[intent], "content")
            for intent in OpeningIntent
        }
        self._intent_by_stem = {_normalized(entry.stem): entry.intent_id for entry in bank.entries}

    def generate_intents(self, *, profile_id: str) -> tuple[str, ...]:
        return _rank_components(self._intents, profile_id, self.budget)

    def generate_stems(
        self,
        *,
        profile_id: str,
        selected_intent: str,
    ) -> tuple[str, ...]:
        intent = OpeningIntent(selected_intent)
        return _rank_components(self._stems_by_intent[intent], profile_id, self.budget)

    def generate_contents(
        self,
        *,
        profile_id: str,
        selected_intent: str,
        selected_stem: str,
    ) -> tuple[str, ...]:
        intent = OpeningIntent(selected_intent)
        if self._intent_by_stem.get(_normalized(selected_stem)) is not intent:
            raise ValueError("selected stem does not belong to the selected intent")
        return _rank_components(self._contents_by_intent[intent], profile_id, self.budget)


class OpeningSelectionStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_index: int = Field(ge=1, le=3)
    component: Literal["opening", "intent", "stem", "content"]
    candidates: tuple[str, ...] = Field(min_length=1, max_length=9)
    intended_component: str
    intended_rank: int | None = Field(default=None, ge=1, le=9)

    @model_validator(mode="after")
    def validate_rank(self) -> OpeningSelectionStage:
        if len({_normalized(value) for value in self.candidates}) != len(self.candidates):
            raise ValueError("opening-stage candidates must be unique")
        if self.intended_rank != _rank(self.candidates, self.intended_component):
            raise ValueError("opening-stage target rank must be scored post hoc")
        return self


class OpeningGeneralizationTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    profile_id: str
    challenge: TestChallenge
    intent_id: OpeningIntent
    stem: str
    content: str
    intended_opening: str
    method: OpeningMethod
    planned_selection_count: Literal[1, 2, 3]
    stages: tuple[OpeningSelectionStage, ...] = Field(min_length=1, max_length=3)
    target_available: bool
    candidate_exposure_count: int = Field(ge=1, le=27)

    @model_validator(mode="after")
    def validate_trial(self) -> OpeningGeneralizationTrial:
        if tuple(stage.stage_index for stage in self.stages) != tuple(
            range(1, len(self.stages) + 1)
        ):
            raise ValueError("opening stages must be consecutive")
        expected_available = len(self.stages) == self.planned_selection_count and all(
            stage.intended_rank is not None for stage in self.stages
        )
        if self.target_available != expected_available:
            raise ValueError("opening availability must agree with all generated stages")
        if self.candidate_exposure_count != sum(len(stage.candidates) for stage in self.stages):
            raise ValueError("candidate exposure count must cover every shown menu")
        return self


class OpeningGeneralizationMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    challenge: TestChallenge
    method: OpeningMethod
    scope: str
    trial_count: int = Field(ge=1)
    available_count: int = Field(ge=0)
    availability_rate: float = Field(ge=0.0, le=1.0)
    planned_selections: Literal[1, 2, 3]
    coverage_per_required_selection: float = Field(ge=0.0, le=1.0)
    mean_menus_reached: float = Field(ge=1.0, le=3.0)
    mean_candidate_exposures: float = Field(ge=1.0, le=27.0)


class OpeningGeneralizationContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    challenge: TestChallenge
    metric: ContrastMetric
    contrast_id: str
    reference_method: OpeningMethod
    comparator_method: OpeningMethod
    estimate: float = Field(ge=-1.0, le=1.0)
    lower_bound: float = Field(ge=-1.0, le=1.0)
    upper_bound: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    resamples: int = Field(ge=2_000)
    sampling_unit: Literal["openings_within_fixed_profile_strata"]


class OpeningGeneralizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    generated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step4_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intended_opening_exposed_to_generators: Literal[False] = False
    downstream_conditioning: Literal["teacher_forced_observed_selections"]
    design_status: Literal["locked_before_execution_exploratory"]
    holdout_counts: dict[str, int]
    trials: tuple[OpeningGeneralizationTrial, ...] = Field(min_length=1)
    metrics: tuple[OpeningGeneralizationMetric, ...] = Field(min_length=1)
    contrasts: tuple[OpeningGeneralizationContrast, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


def _stage(
    index: int,
    component: Literal["opening", "intent", "stem", "content"],
    candidates: tuple[str, ...],
    target: str,
) -> OpeningSelectionStage:
    return OpeningSelectionStage(
        stage_index=index,
        component=component,
        candidates=candidates,
        intended_component=target,
        intended_rank=_rank(candidates, target),
    )


def _trial_for_record(
    record: OpeningRecord,
    method: OpeningMethod,
    one_stage: TargetBlindPhraseOpeningGenerator,
    two_stage: TargetBlindGlobalTwoStageOpeningGenerator,
    three_stage: TargetBlindIntentThreeStageOpeningGenerator,
) -> OpeningGeneralizationTrial:
    stages: list[OpeningSelectionStage] = []
    if method is OpeningMethod.ONE_STAGE_PHRASE:
        stages.append(
            _stage(
                1,
                "opening",
                one_stage.generate(profile_id=record.profile_id),
                record.opening,
            )
        )
    elif method is OpeningMethod.TWO_STAGE_STEM_CONTENT:
        stem_stage = _stage(
            1,
            "stem",
            two_stage.generate_stems(profile_id=record.profile_id),
            record.stem,
        )
        stages.append(stem_stage)
        if stem_stage.intended_rank is not None:
            stages.append(
                _stage(
                    2,
                    "content",
                    two_stage.generate_contents(
                        profile_id=record.profile_id,
                        selected_stem=record.stem,
                    ),
                    record.content,
                )
            )
    else:
        intent_stage = _stage(
            1,
            "intent",
            three_stage.generate_intents(profile_id=record.profile_id),
            record.intent_id.value,
        )
        stages.append(intent_stage)
        if intent_stage.intended_rank is not None:
            stem_stage = _stage(
                2,
                "stem",
                three_stage.generate_stems(
                    profile_id=record.profile_id,
                    selected_intent=record.intent_id.value,
                ),
                record.stem,
            )
            stages.append(stem_stage)
            if stem_stage.intended_rank is not None:
                stages.append(
                    _stage(
                        3,
                        "content",
                        three_stage.generate_contents(
                            profile_id=record.profile_id,
                            selected_intent=record.intent_id.value,
                            selected_stem=record.stem,
                        ),
                        record.content,
                    )
                )
    planned_by_method: dict[OpeningMethod, Literal[1, 2, 3]] = {
        OpeningMethod.ONE_STAGE_PHRASE: 1,
        OpeningMethod.TWO_STAGE_STEM_CONTENT: 2,
        OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT: 3,
    }
    planned = planned_by_method[method]
    available = len(stages) == planned and all(stage.intended_rank is not None for stage in stages)
    return OpeningGeneralizationTrial(
        record_id=record.record_id,
        profile_id=record.profile_id,
        challenge=cast(TestChallenge, record.challenge.value),
        intent_id=record.intent_id,
        stem=record.stem,
        content=record.content,
        intended_opening=record.opening,
        method=method,
        planned_selection_count=planned,
        stages=tuple(stages),
        target_available=available,
        candidate_exposure_count=sum(len(stage.candidates) for stage in stages),
    )


def _metrics(
    trials: Sequence[OpeningGeneralizationTrial],
) -> tuple[OpeningGeneralizationMetric, ...]:
    records: list[OpeningGeneralizationMetric] = []
    for challenge in (
        OpeningChallenge.HELDOUT_COMBINATION,
        OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY,
    ):
        for method in OpeningMethod:
            challenge_value = cast(TestChallenge, challenge.value)
            method_rows = tuple(
                trial
                for trial in trials
                if trial.challenge == challenge_value and trial.method is method
            )
            scopes = {"overall": method_rows}
            for intent in OpeningIntent:
                scopes[f"intent:{intent.value}"] = tuple(
                    trial for trial in method_rows if trial.intent_id is intent
                )
            for scope, rows in scopes.items():
                available = sum(trial.target_available for trial in rows)
                planned = rows[0].planned_selection_count
                records.append(
                    OpeningGeneralizationMetric(
                        challenge=challenge_value,
                        method=method,
                        scope=scope,
                        trial_count=len(rows),
                        available_count=available,
                        availability_rate=available / len(rows),
                        planned_selections=planned,
                        coverage_per_required_selection=(available / len(rows)) / planned,
                        mean_menus_reached=float(np.mean([len(trial.stages) for trial in rows])),
                        mean_candidate_exposures=float(
                            np.mean([trial.candidate_exposure_count for trial in rows])
                        ),
                    )
                )
    return tuple(records)


def _bootstrap_contrast(
    *,
    challenge: OpeningChallenge,
    metric: ContrastMetric,
    reference_method: OpeningMethod,
    comparator_method: OpeningMethod,
    trials: Sequence[OpeningGeneralizationTrial],
    spec: OpeningGeneralizationSpec,
) -> OpeningGeneralizationContrast:
    by_method = {
        method: {
            trial.record_id: trial
            for trial in trials
            if trial.challenge == challenge.value and trial.method is method
        }
        for method in (reference_method, comparator_method)
    }
    reference = by_method[reference_method]
    comparator = by_method[comparator_method]
    if set(reference) != set(comparator):
        raise ValueError("opening methods do not contain paired trial IDs")
    scales = {
        OpeningMethod.ONE_STAGE_PHRASE: 1.0,
        OpeningMethod.TWO_STAGE_STEM_CONTENT: 0.5,
        OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT: 1.0 / 3.0,
    }
    reference_scale = scales[reference_method] if metric != "availability_rate" else 1.0
    comparator_scale = scales[comparator_method] if metric != "availability_rate" else 1.0
    deltas_by_profile: dict[str, list[float]] = defaultdict(list)
    for record_id in sorted(reference):
        ref = reference[record_id]
        cmp = comparator[record_id]
        deltas_by_profile[ref.profile_id].append(
            float(ref.target_available) * reference_scale
            - float(cmp.target_available) * comparator_scale
        )
    point = float(np.mean([value for values in deltas_by_profile.values() for value in values]))
    seed_material = f"{challenge.value}:{metric}:{reference_method.value}:{comparator_method.value}"
    offset = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big")
    rng = np.random.default_rng(spec.bootstrap_seed + offset)
    samples = np.zeros(spec.bootstrap_resamples, dtype=np.float64)
    denominator = 0
    for values in deltas_by_profile.values():
        array = np.asarray(values, dtype=np.float64)
        for start in range(0, spec.bootstrap_resamples, 1_000):
            stop = min(start + 1_000, spec.bootstrap_resamples)
            indices = rng.integers(0, len(array), size=(stop - start, len(array)))
            samples[start:stop] += array[indices].sum(axis=1)
        denominator += len(array)
    samples /= denominator
    return OpeningGeneralizationContrast(
        challenge=cast(TestChallenge, challenge.value),
        metric=metric,
        contrast_id=(f"{reference_method.value}-minus-{comparator_method.value}-{metric}"),
        reference_method=reference_method,
        comparator_method=comparator_method,
        estimate=point,
        lower_bound=float(np.quantile(samples, 0.025)),
        upper_bound=float(np.quantile(samples, 0.975)),
        resamples=spec.bootstrap_resamples,
        sampling_unit="openings_within_fixed_profile_strata",
    )


def evaluate_opening_generalization(
    *,
    spec: OpeningGeneralizationSpec,
    source: OpeningGeneralizationSource,
    records: Sequence[OpeningRecord],
    bank: OpeningTrainingBank,
    protocol_sha256: str,
    step4_manifest_sha256: str,
) -> OpeningGeneralizationResult:
    if bank.source_sha256 != source.digest():
        raise ValueError("opening bank refers to a different benchmark source")
    holdout_counts = validate_opening_holdouts(records)
    one_stage = TargetBlindPhraseOpeningGenerator(bank, spec.candidate_budget)
    two_stage = TargetBlindGlobalTwoStageOpeningGenerator(bank, spec.candidate_budget)
    three_stage = TargetBlindIntentThreeStageOpeningGenerator(
        bank,
        spec.candidate_budget,
    )
    test_records = tuple(record for record in records if record.split is OpeningSplit.TEST)
    trials = tuple(
        _trial_for_record(record, method, one_stage, two_stage, three_stage)
        for record in test_records
        for method in OpeningMethod
    )
    metrics = _metrics(trials)
    comparisons = (
        (OpeningMethod.TWO_STAGE_STEM_CONTENT, OpeningMethod.ONE_STAGE_PHRASE),
        (
            OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT,
            OpeningMethod.ONE_STAGE_PHRASE,
        ),
        (
            OpeningMethod.THREE_STAGE_INTENT_STEM_CONTENT,
            OpeningMethod.TWO_STAGE_STEM_CONTENT,
        ),
    )
    contrast_metrics: tuple[ContrastMetric, ...] = (
        "availability_rate",
        "coverage_per_required_selection",
    )
    contrasts = tuple(
        _bootstrap_contrast(
            challenge=challenge,
            metric=metric,
            reference_method=reference,
            comparator_method=comparator,
            trials=trials,
            spec=spec,
        )
        for challenge in (
            OpeningChallenge.HELDOUT_COMBINATION,
            OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY,
        )
        for metric in contrast_metrics
        for reference, comparator in comparisons
    )
    fingerprint = _digest(
        [
            {
                "record_id": trial.record_id,
                "method": trial.method,
                "available": trial.target_available,
                "ranks": [stage.intended_rank for stage in trial.stages],
            }
            for trial in trials
        ]
    )
    identity = {
        "config_sha256": spec.digest(),
        "source_sha256": source.digest(),
        "bank_sha256": bank.digest(),
        "trial_fingerprint": fingerprint,
    }
    return OpeningGeneralizationResult(
        run_id=f"opening-generalization-{_digest(identity)[:20]}",
        generated_at=spec.locked_at,
        config_sha256=spec.digest(),
        protocol_sha256=protocol_sha256,
        step4_manifest_sha256=step4_manifest_sha256,
        benchmark_source_sha256=source.digest(),
        training_bank_sha256=bank.digest(),
        intended_opening_exposed_to_generators=False,
        downstream_conditioning="teacher_forced_observed_selections",
        design_status=spec.design_status,
        holdout_counts=holdout_counts,
        trials=trials,
        metrics=metrics,
        contrasts=contrasts,
        limitations=(
            "The benchmark is synthetic and developer-authored rather than participant language.",
            "Combination tests reuse fitted stems and content words but withhold their exact "
            "pairs.",
            "Paraphrase-family tests contain stems absent from fitting and therefore expose the "
            "closed-vocabulary boundary of all three retrieval methods.",
            "Hierarchical replay teacher-forces each correct upstream selection before generating "
            "the next target-blind menu.",
            "Coverage-per-selection is a descriptive interface-efficiency measure, not a live BCI "
            "communication-rate estimate.",
        ),
    )
