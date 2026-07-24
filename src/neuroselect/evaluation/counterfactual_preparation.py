"""Prepare checksum-addressed language/P300 inputs for counterfactual fusion."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.bci import FlashLayout, FlashProbabilityTrial
from neuroselect.decoding import DecoderEvaluation
from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.counterfactual import flash_trials_from_decoder_evaluation
from neuroselect.evaluation.counterfactual_models import (
    CounterfactualExperimentInput,
    CounterfactualFusionSpec,
    CounterfactualFusionTrial,
)
from neuroselect.evaluation.language_benchmark import (
    HeldOutLanguageResult,
    LanguageBenchmarkTrial,
)
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus

DEFAULT_PREPARATION_CONFIG = Path("configs/experiments/counterfactual_input_development.yaml")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class CounterfactualPreparationSpec(BaseModel):
    """Locked rules for pairing complete language messages with recorded EEG trials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "2.0"] = "1.0"
    preparation_revision: Literal[
        "language-p300-paired-input-v1",
        "subject-profile-balanced-paired-input-v2",
    ] = "language-p300-paired-input-v1"
    experiment_id: str = Field(min_length=1, max_length=160)
    seed: int = Field(default=20260724, ge=0)
    evidence_tier: Literal["development", "research"] = "development"
    maximum_messages: int | None = Field(default=1, ge=1)
    message_order: Literal["seeded-sha256-v1"] = "seeded-sha256-v1"
    eeg_trial_order: Literal[
        "source-selection-id-v1",
        "seeded-subject-sha256-v1",
    ] = "source-selection-id-v1"
    layout_revision: Literal["balanced-event-signatures-v1"] = "balanced-event-signatures-v1"
    sampling_revision: Literal[
        "capacity-greedy-v1",
        "subject-profile-balanced-complete-message-v1",
    ] = "capacity-greedy-v1"
    required_profile_ids: tuple[str, ...] = ()
    required_eeg_subject_ids: tuple[str, ...] = ()
    messages_per_profile_per_eeg_subject: int | None = Field(default=None, ge=1)
    required_message_span_count: int | None = Field(default=None, ge=1)
    inference_scope: Literal[
        "development-only",
        "study-p-dataset-specific-descriptive",
    ] = "development-only"

    @model_validator(mode="after")
    def validate_evidence_tier(self) -> CounterfactualPreparationSpec:
        balanced = self.sampling_revision == "subject-profile-balanced-complete-message-v1"
        if self.preparation_revision == "language-p300-paired-input-v1":
            if (
                self.schema_version != "1.0"
                or balanced
                or self.required_profile_ids
                or self.required_eeg_subject_ids
                or self.messages_per_profile_per_eeg_subject is not None
                or self.required_message_span_count is not None
                or self.inference_scope != "development-only"
            ):
                raise ValueError("v1 preparation cannot declare balanced research sampling")
            if self.evidence_tier == "research":
                raise ValueError("research preparation requires the balanced v2 protocol")
            return self
        if (
            self.schema_version != "2.0"
            or self.evidence_tier != "research"
            or self.maximum_messages is not None
            or not balanced
            or self.eeg_trial_order != "seeded-subject-sha256-v1"
            or not self.required_profile_ids
            or not self.required_eeg_subject_ids
            or self.messages_per_profile_per_eeg_subject is None
            or self.required_message_span_count is None
            or self.inference_scope != "study-p-dataset-specific-descriptive"
        ):
            raise ValueError(
                "balanced v2 research preparation requires complete sampling parameters"
            )
        if len(self.required_profile_ids) != len(set(self.required_profile_ids)):
            raise ValueError("balanced research profile IDs must be unique")
        if len(self.required_eeg_subject_ids) != len(set(self.required_eeg_subject_ids)):
            raise ValueError("balanced research EEG subject IDs must be unique")
        return self

    @property
    def planned_counterfactual_trial_count(self) -> int | None:
        if self.sampling_revision != "subject-profile-balanced-complete-message-v1":
            return None
        assert self.messages_per_profile_per_eeg_subject is not None
        assert self.required_message_span_count is not None
        return (
            len(self.required_profile_ids)
            * len(self.required_eeg_subject_ids)
            * self.messages_per_profile_per_eeg_subject
            * self.required_message_span_count
        )

    @property
    def planned_trials_per_eeg_subject(self) -> int | None:
        if self.sampling_revision != "subject-profile-balanced-complete-message-v1":
            return None
        assert self.messages_per_profile_per_eeg_subject is not None
        assert self.required_message_span_count is not None
        return (
            len(self.required_profile_ids)
            * self.messages_per_profile_per_eeg_subject
            * self.required_message_span_count
        )

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        if self.preparation_revision == "language-p300-paired-input-v1":
            for field in (
                "sampling_revision",
                "required_profile_ids",
                "required_eeg_subject_ids",
                "messages_per_profile_per_eeg_subject",
                "required_message_span_count",
                "inference_scope",
            ):
                payload.pop(field)
        return _sha256_text(_canonical_json(payload))


def load_counterfactual_preparation_spec(
    path: str | Path = DEFAULT_PREPARATION_CONFIG,
) -> CounterfactualPreparationSpec:
    """Load a strict language/P300 input-preparation recipe."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("counterfactual preparation config must contain a YAML mapping")
    return CounterfactualPreparationSpec.model_validate(payload)


class CounterfactualInputBuilder:
    """Pair complete held-out messages with distinct recorded P300 selection trials."""

    def __init__(
        self,
        preparation_spec: CounterfactualPreparationSpec,
        fusion_spec: CounterfactualFusionSpec,
    ) -> None:
        self.preparation_spec = preparation_spec
        self.fusion_spec = fusion_spec

    def build(
        self,
        *,
        language_result: HeldOutLanguageResult,
        decoder_evaluation: DecoderEvaluation,
        source_decoder_manifest_sha256: str,
        original_task_evaluation_sha256: str,
        source_language_manifest_sha256: str,
        source_language_result_sha256: str,
        prepared_at: datetime,
    ) -> CounterfactualExperimentInput:
        """Build a portable input without changing candidate or source-flash evidence."""

        if self.fusion_spec.schema_version != "2.0":
            raise ValueError("language/P300 preparation requires counterfactual protocol v2")
        if self.fusion_spec.personalization_evidence_kind != "held_out_adapter":
            raise ValueError("trained language artifacts require held-out-adapter evidence")
        flash_trials = flash_trials_from_decoder_evaluation(decoder_evaluation)
        messages = self._eligible_messages(language_result)
        if self.preparation_spec.sampling_revision == (
            "subject-profile-balanced-complete-message-v1"
        ):
            trial_pairs = self._balanced_trial_pairs(messages, flash_trials)
        else:
            selected_messages = self._select_messages(messages, len(flash_trials))
            language_trials = tuple(trial for message in selected_messages for trial in message)
            selected_flash_trials = flash_trials[: len(language_trials)]
            trial_pairs = tuple(zip(language_trials, selected_flash_trials, strict=True))
        prepared_trials = tuple(
            self._prepare_trial(language_trial, flash_trial)
            for language_trial, flash_trial in trial_pairs
        )
        language_trials = tuple(pair[0] for pair in trial_pairs)
        selected_flash_trials = tuple(pair[1] for pair in trial_pairs)
        source_claim_eligible = (
            self.preparation_spec.evidence_tier == "research"
            and self.preparation_spec.maximum_messages is None
            and language_result.claim_eligible
            and self.preparation_spec.planned_counterfactual_trial_count == len(language_trials)
        )
        limitations = (
            (
                f"Mapped {len(language_trials)} of {len(language_result.trials)} language "
                f"trials and {len(selected_flash_trials)} of {len(flash_trials)} labeled EEG "
                "selection trials while preserving complete messages."
            ),
            *(
                (
                    "The counterfactual sample is a preregistered balanced subset; the complete "
                    "held-out language result remains separate component evidence.",
                )
                if self.preparation_spec.sampling_revision
                == "subject-profile-balanced-complete-message-v1"
                else ()
            ),
            *(
                (
                    "Inference is limited to descriptive comparisons within this pinned Study P "
                    "offline replay sample; it is not population or clinical inference.",
                )
                if self.preparation_spec.inference_scope == "study-p-dataset-specific-descriptive"
                else ()
            ),
            (
                "Source flash codes are occurrence-level Study P event identifiers; the source "
                "layout uses deterministic balanced event-signature groups, not an observed "
                "NeuroSelect display."
            ),
            *(
                ()
                if language_result.claim_eligible
                else (
                    "The source language evaluation is development evidence and is not "
                    "claim-eligible.",
                )
            ),
            *(
                ()
                if source_claim_eligible
                else (
                    "This prepared development input is not eligible for counterfactual "
                    "performance claims.",
                )
            ),
        )
        return CounterfactualExperimentInput(
            schema_version="2.0",
            prepared_at=prepared_at,
            preparation_revision=self.preparation_spec.preparation_revision,
            preparation_config_sha256=self.preparation_spec.digest(),
            source_decoder_manifest_sha256=source_decoder_manifest_sha256,
            original_task_evaluation_sha256=original_task_evaluation_sha256,
            source_language_manifest_sha256=source_language_manifest_sha256,
            source_language_result_sha256=source_language_result_sha256,
            source_evidence_claim_eligible=source_claim_eligible,
            preparation_limitations=limitations,
            spec=self.fusion_spec,
            trials=prepared_trials,
        )

    def _eligible_messages(
        self,
        language_result: HeldOutLanguageResult,
    ) -> tuple[tuple[LanguageBenchmarkTrial, ...], ...]:
        grouped: dict[tuple[str, str], list[LanguageBenchmarkTrial]] = defaultdict(list)
        for trial in language_result.trials:
            grouped[(trial.profile_id, trial.message_id)].append(trial)
        eligible: list[tuple[LanguageBenchmarkTrial, ...]] = []
        for key, raw_trials in grouped.items():
            trials = tuple(sorted(raw_trials, key=lambda item: item.span_index))
            expected_count = trials[0].message_span_count
            if len(trials) != expected_count or tuple(item.span_index for item in trials) != tuple(
                range(expected_count)
            ):
                raise ValueError(f"language result contains an incomplete message: {key}")
            if any(
                trial.candidate_generation_failed
                or trial.candidate_set is None
                or trial.other_candidate_id is None
                for trial in trials
            ):
                continue
            candidate_counts = {
                len(trial.candidate_set.candidates)
                for trial in trials
                if trial.candidate_set is not None
            }
            if len(candidate_counts) != 1:
                raise ValueError(f"one language message changes candidate count: {key}")
            eligible.append(trials)
        if not eligible:
            raise ValueError("language result contains no complete successful messages")
        return tuple(
            sorted(
                eligible,
                key=lambda trials: hashlib.sha256(
                    (
                        f"{self.preparation_spec.seed}:"
                        f"{trials[0].profile_id}:{trials[0].message_id}"
                    ).encode()
                ).digest(),
            )
        )

    def _balanced_trial_pairs(
        self,
        messages: tuple[tuple[LanguageBenchmarkTrial, ...], ...],
        flash_trials: tuple[FlashProbabilityTrial, ...],
    ) -> tuple[tuple[LanguageBenchmarkTrial, FlashProbabilityTrial], ...]:
        spec = self.preparation_spec
        assert spec.messages_per_profile_per_eeg_subject is not None
        assert spec.required_message_span_count is not None
        assert spec.planned_trials_per_eeg_subject is not None
        messages_by_profile: dict[str, list[tuple[LanguageBenchmarkTrial, ...]]] = defaultdict(list)
        for message in messages:
            if (
                message[0].profile_id in spec.required_profile_ids
                and len(message) == spec.required_message_span_count
            ):
                messages_by_profile[message[0].profile_id].append(message)
        required_messages_per_profile = (
            len(spec.required_eeg_subject_ids) * spec.messages_per_profile_per_eeg_subject
        )
        for profile_id in spec.required_profile_ids:
            available = len(messages_by_profile[profile_id])
            if available < required_messages_per_profile:
                raise ValueError(
                    f"profile {profile_id} provides {available} eligible complete messages; "
                    f"balanced sampling requires {required_messages_per_profile}"
                )

        trials_by_subject: dict[str, list[FlashProbabilityTrial]] = defaultdict(list)
        for flash_trial in flash_trials:
            if flash_trial.subject_id in spec.required_eeg_subject_ids:
                trials_by_subject[flash_trial.subject_id].append(flash_trial)
        for subject_id in spec.required_eeg_subject_ids:
            trials_by_subject[subject_id].sort(
                key=lambda trial: hashlib.sha256(
                    (
                        f"{spec.seed}:{subject_id}:{trial.session_id}:{trial.selection_trial_id}"
                    ).encode()
                ).digest()
            )
            available = len(trials_by_subject[subject_id])
            if available < spec.planned_trials_per_eeg_subject:
                raise ValueError(
                    f"EEG subject {subject_id} provides {available} usable selection trials; "
                    f"balanced sampling requires {spec.planned_trials_per_eeg_subject}"
                )

        pairs: list[tuple[LanguageBenchmarkTrial, FlashProbabilityTrial]] = []
        for subject_index, subject_id in enumerate(spec.required_eeg_subject_ids):
            subject_language_trials: list[LanguageBenchmarkTrial] = []
            message_start = subject_index * spec.messages_per_profile_per_eeg_subject
            message_stop = message_start + spec.messages_per_profile_per_eeg_subject
            for profile_id in spec.required_profile_ids:
                selected = messages_by_profile[profile_id][message_start:message_stop]
                subject_language_trials.extend(trial for message in selected for trial in message)
            selected_flash_trials = trials_by_subject[subject_id][
                : spec.planned_trials_per_eeg_subject
            ]
            pairs.extend(zip(subject_language_trials, selected_flash_trials, strict=True))
        assert spec.planned_counterfactual_trial_count is not None
        if len(pairs) != spec.planned_counterfactual_trial_count:
            raise ValueError("balanced counterfactual sample size does not match its protocol")
        return tuple(pairs)

    def _select_messages(
        self,
        messages: tuple[tuple[LanguageBenchmarkTrial, ...], ...],
        flash_trial_count: int,
    ) -> tuple[tuple[LanguageBenchmarkTrial, ...], ...]:
        selected: list[tuple[LanguageBenchmarkTrial, ...]] = []
        used_trials = 0
        limit = self.preparation_spec.maximum_messages
        for message in messages:
            if limit is not None and len(selected) >= limit:
                break
            if used_trials + len(message) > flash_trial_count:
                continue
            selected.append(message)
            used_trials += len(message)
        if not selected:
            minimum = min(len(message) for message in messages)
            raise ValueError(
                f"decoder provides {flash_trial_count} trials but the shortest complete "
                f"language message requires {minimum}"
            )
        return tuple(selected)

    def _prepare_trial(
        self,
        language_trial: LanguageBenchmarkTrial,
        flash_trial: FlashProbabilityTrial,
    ) -> CounterfactualFusionTrial:
        assert language_trial.candidate_set is not None
        assert language_trial.other_candidate_id is not None
        layout = self._source_layout(language_trial, flash_trial)
        material = (
            f"{self.preparation_spec.digest()}:{language_trial.trial_id}:"
            f"{flash_trial.selection_trial_id}"
        )
        return CounterfactualFusionTrial(
            trial_id=f"prepared-{_sha256_text(material)[:20]}",
            synthetic_profile_id=language_trial.profile_id,
            message_id=language_trial.message_id,
            span_index=language_trial.span_index,
            message_span_count=language_trial.message_span_count,
            candidate_set=language_trial.candidate_set,
            flash_layout=layout,
            flash_trial=flash_trial,
            intended_text=language_trial.intended_text,
            intended_candidate_id=language_trial.intended_candidate_id,
            other_candidate_id=language_trial.other_candidate_id,
            candidate_generation_failed=False,
            confirmed_context=language_trial.confirmed_context,
            generic_language_support=language_trial.generic_language_support,
            personalization_lift=language_trial.personalization_lift,
            personalization_adapter_id=language_trial.adapter_id,
            personalization_adapter_sha256=language_trial.adapter_sha256,
            retrieval_evidence=language_trial.retrieval_evidence,
        )

    def _source_layout(
        self,
        language_trial: LanguageBenchmarkTrial,
        flash_trial: FlashProbabilityTrial,
    ) -> FlashLayout:
        assert language_trial.candidate_set is not None
        candidate_ids = tuple(
            candidate.candidate_id for candidate in language_trial.candidate_set.candidates
        )
        observed_codes = tuple(dict.fromkeys(event.stimulus_code for event in flash_trial.events))
        target_codes = tuple(
            code for code in observed_codes if code in flash_trial.recorded_target_codes
        )
        non_target_codes = tuple(
            code for code in observed_codes if code not in flash_trial.recorded_target_codes
        )
        remaining_candidates = candidate_ids[1:]
        if not target_codes:
            raise ValueError("source EEG trial has no recorded target signature")
        if len(non_target_codes) < len(remaining_candidates):
            raise ValueError(
                "source EEG trial has too few non-target event codes for the candidate layout"
            )
        event_counts = {
            code: sum(event.stimulus_code == code for event in flash_trial.events)
            for code in non_target_codes
        }
        source_order = {code: index for index, code in enumerate(observed_codes)}
        buckets: list[list[int]] = [[] for _ in remaining_candidates]
        loads = [0 for _ in remaining_candidates]
        for code in sorted(
            non_target_codes,
            key=lambda item: (-event_counts[item], source_order[item]),
        ):
            bucket_index = min(
                range(len(buckets)),
                key=lambda index: (loads[index], index),
            )
            buckets[bucket_index].append(code)
            loads[bucket_index] += event_counts[code]
        signatures = {
            candidate_ids[0]: target_codes,
            **{
                candidate_id: tuple(sorted(bucket, key=source_order.__getitem__))
                for candidate_id, bucket in zip(remaining_candidates, buckets, strict=True)
            },
        }
        layout_material = (
            f"{self.preparation_spec.layout_revision}:"
            f"{language_trial.candidate_set.candidate_set_id}:"
            f"{flash_trial.selection_trial_id}"
        )
        return FlashLayout(
            layout_id=f"source-layout-{_sha256_text(layout_material)[:20]}",
            candidate_ids=candidate_ids,
            stimulus_codes=observed_codes,
            candidate_code_sets=signatures,
        )


def write_counterfactual_input_artifacts(
    experiment_input: CounterfactualExperimentInput,
    preparation_spec: CounterfactualPreparationSpec,
    output_dir: str | Path,
    *,
    git_sha: str,
    overwrite: bool = False,
    source_tree_sha256: str | None = None,
    package_versions: dict[str, str] | None = None,
    device: dict[str, str] | None = None,
) -> RunManifest:
    """Write canonical input JSON and a source/output checksum manifest."""

    if (
        experiment_input.preparation_config_sha256 is None
        or experiment_input.preparation_revision is None
        or experiment_input.source_language_manifest_sha256 is None
        or experiment_input.source_language_result_sha256 is None
    ):
        raise ValueError("prepared input is missing preparation or language provenance")
    if (
        experiment_input.preparation_revision != preparation_spec.preparation_revision
        or experiment_input.preparation_config_sha256 != preparation_spec.digest()
    ):
        raise ValueError("preparation specification does not agree with the prepared input")
    destination = Path(output_dir)
    input_path = destination / "input.json"
    manifest_path = destination / "manifest.json"
    existing = [str(path) for path in (input_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite counterfactual input artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    input_content = experiment_input.canonical_json() + "\n"
    input_path.write_text(input_content, encoding="utf-8")
    captured_packages, captured_device = capture_runtime_environment()
    profile_trial_counts: dict[str, int] = defaultdict(int)
    eeg_subject_trial_counts: dict[str, int] = defaultdict(int)
    for trial in experiment_input.trials:
        profile_trial_counts[trial.resolved_profile_id] += 1
        eeg_subject_trial_counts[trial.flash_trial.subject_id] += 1
    adapters = {
        trial.personalization_adapter_id: trial.personalization_adapter_sha256
        for trial in experiment_input.trials
        if trial.personalization_adapter_id is not None
        and trial.personalization_adapter_sha256 is not None
    }
    manifest = RunManifest(
        run_id=f"counterfactual-input-{experiment_input.digest()[:20]}",
        run_kind=RunKind.COMPONENT_EVALUATION,
        status=RunStatus.COMPLETED,
        started_at=experiment_input.prepared_at,
        completed_at=experiment_input.prepared_at,
        git_sha=git_sha,
        config_sha256=experiment_input.preparation_config_sha256,
        random_seeds={"language_message_pairing": preparation_spec.seed},
        package_versions=captured_packages if package_versions is None else package_versions,
        device=captured_device if device is None else device,
        datasets=(
            ArtifactRef(
                artifact_id="source-language-manifest",
                uri="artifact://source-language-manifest",
                sha256=experiment_input.source_language_manifest_sha256,
            ),
            ArtifactRef(
                artifact_id="source-language-result",
                uri="artifact://source-language-result",
                sha256=experiment_input.source_language_result_sha256,
            ),
            ArtifactRef(
                artifact_id="source-decoder-manifest",
                uri="artifact://source-decoder-manifest",
                sha256=experiment_input.source_decoder_manifest_sha256,
            ),
            ArtifactRef(
                artifact_id="original-task-evaluation",
                uri="artifact://original-task-evaluation",
                sha256=experiment_input.original_task_evaluation_sha256,
            ),
        ),
        models=tuple(
            ArtifactRef(
                artifact_id=f"personalization-{adapter_id}",
                uri=f"model://personalization/{adapter_id}",
                sha256=digest,
                revision=adapter_id,
            )
            for adapter_id, digest in sorted(adapters.items())
        ),
        outputs=(
            ArtifactRef(
                artifact_id="prepared-counterfactual-input",
                uri="artifact://input.json",
                sha256=_sha256_text(input_content),
                revision=experiment_input.preparation_revision,
            ),
        ),
        metadata={
            "evidence_kind": "counterfactual_input_preparation",
            "claim_eligible": experiment_input.source_evidence_claim_eligible,
            "input_sha256": experiment_input.digest(),
            "source_trial_count": len(experiment_input.trials),
            "sampling_revision": preparation_spec.sampling_revision,
            "inference_scope": preparation_spec.inference_scope,
            "profile_trial_counts": dict(sorted(profile_trial_counts.items())),
            "eeg_subject_trial_counts": dict(sorted(eeg_subject_trial_counts.items())),
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_counterfactual_input_artifacts(
    directory: str | Path,
) -> tuple[CounterfactualExperimentInput, RunManifest]:
    """Verify a prepared input file and cross-check all source checksums."""

    source = Path(directory)
    input_content = (source / "input.json").read_text(encoding="utf-8")
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    output_ref = next(
        (item for item in manifest.outputs if item.uri == "artifact://input.json"),
        None,
    )
    if output_ref is None or output_ref.sha256 != _sha256_text(input_content):
        raise ValueError("counterfactual input artifact SHA-256 mismatch")
    experiment_input = CounterfactualExperimentInput.model_validate_json(input_content)
    if (
        experiment_input.preparation_config_sha256 is None
        or experiment_input.source_language_manifest_sha256 is None
        or experiment_input.source_language_result_sha256 is None
    ):
        raise ValueError("prepared input is missing required manifest provenance")
    expected_datasets = {
        "artifact://source-language-manifest": (experiment_input.source_language_manifest_sha256),
        "artifact://source-language-result": (experiment_input.source_language_result_sha256),
        "artifact://source-decoder-manifest": (experiment_input.source_decoder_manifest_sha256),
        "artifact://original-task-evaluation": (experiment_input.original_task_evaluation_sha256),
    }
    expected_models = {
        f"model://personalization/{trial.personalization_adapter_id}": (
            trial.personalization_adapter_sha256
        )
        for trial in experiment_input.trials
        if trial.personalization_adapter_id is not None
        and trial.personalization_adapter_sha256 is not None
    }
    profile_trial_counts: dict[str, int] = defaultdict(int)
    eeg_subject_trial_counts: dict[str, int] = defaultdict(int)
    for trial in experiment_input.trials:
        profile_trial_counts[trial.resolved_profile_id] += 1
        eeg_subject_trial_counts[trial.flash_trial.subject_id] += 1
    balanced_v2 = (
        experiment_input.preparation_revision == "subject-profile-balanced-paired-input-v2"
    )
    balanced_metadata_valid = not balanced_v2 or (
        manifest.metadata.get("sampling_revision") == "subject-profile-balanced-complete-message-v1"
        and manifest.metadata.get("inference_scope") == "study-p-dataset-specific-descriptive"
        and manifest.metadata.get("profile_trial_counts")
        == dict(sorted(profile_trial_counts.items()))
        and manifest.metadata.get("eeg_subject_trial_counts")
        == dict(sorted(eeg_subject_trial_counts.items()))
        and len(set(profile_trial_counts.values())) == 1
        and len(set(eeg_subject_trial_counts.values())) == 1
    )
    if (
        manifest.run_kind is not RunKind.COMPONENT_EVALUATION
        or manifest.status is not RunStatus.COMPLETED
        or manifest.config_sha256 != experiment_input.preparation_config_sha256
        or {item.uri: item.sha256 for item in manifest.datasets} != expected_datasets
        or {item.uri: item.sha256 for item in manifest.models} != expected_models
        or output_ref.revision != experiment_input.preparation_revision
        or manifest.metadata.get("input_sha256") != experiment_input.digest()
        or manifest.metadata.get("claim_eligible")
        is not experiment_input.source_evidence_claim_eligible
        or not balanced_metadata_valid
    ):
        raise ValueError("counterfactual input manifest does not agree with the input")
    return experiment_input, manifest
