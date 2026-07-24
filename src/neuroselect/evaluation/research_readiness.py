"""Fail-closed readiness checks for research-grade evidence expansion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.decoding import DecoderCheckpointMetadata, DecoderEvaluation
from neuroselect.evaluation.counterfactual import (
    flash_trials_from_decoder_evaluation,
    load_counterfactual_spec,
)
from neuroselect.evaluation.counterfactual_preparation import (
    load_counterfactual_preparation_spec,
)
from neuroselect.evaluation.language_artifacts import (
    read_held_out_language_artifacts,
)
from neuroselect.evaluation.language_benchmark import (
    load_held_out_language_spec,
)
from neuroselect.language import (
    load_local_model_config,
    load_lora_training_config,
    load_personalization_adapter,
)
from neuroselect.provenance import RunKind, RunManifest
from neuroselect.synthetic import BenchmarkSplit, generate_from_sources

DEFAULT_RESEARCH_EXPANSION_CONFIG = Path("configs/experiments/research_evidence_expansion.yaml")
PRIMARY_RESEARCH_CONDITIONS = (
    "a_bci_only",
    "b_generic_language_only",
    "c_neural_language",
    "d_neural_personalized",
    "e_neural_personalized_rag",
    "f_complete_system",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchExpansionSpec(BaseModel):
    """Tracked inputs and exact completeness requirements after Step 8."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    expansion_revision: Literal["full-language-balanced-p300-v2"]
    expansion_id: str = Field(min_length=1, max_length=160)
    evaluated_at: datetime
    benchmark_spec: Path
    profiles: Path
    language_evaluation_config: Path
    language_training_config: Path
    language_model_config: Path
    counterfactual_preparation_config: Path
    counterfactual_fusion_config: Path
    adapter_root: Path
    adapter_suffix: str = Field(min_length=1, max_length=80)
    language_artifacts: Path
    decoder_artifacts: Path
    required_profile_ids: tuple[str, ...] = Field(min_length=1)
    required_eeg_subject_ids: tuple[str, ...] = Field(min_length=1)
    required_training_eeg_subject_ids: tuple[str, ...] = Field(min_length=1)
    required_validation_eeg_subject_ids: tuple[str, ...] = Field(min_length=1)
    require_clean_sources: bool = True

    @model_validator(mode="after")
    def validate_recipe(self) -> ResearchExpansionSpec:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("research expansion evaluation time must include a timezone")
        if len(self.required_profile_ids) != len(set(self.required_profile_ids)):
            raise ValueError("research expansion profile IDs must be unique")
        if len(self.required_eeg_subject_ids) != len(set(self.required_eeg_subject_ids)):
            raise ValueError("research expansion EEG subject IDs must be unique")
        if len(self.required_training_eeg_subject_ids) != len(
            set(self.required_training_eeg_subject_ids)
        ):
            raise ValueError("research expansion training EEG subject IDs must be unique")
        if len(self.required_validation_eeg_subject_ids) != len(
            set(self.required_validation_eeg_subject_ids)
        ):
            raise ValueError("research expansion validation EEG subject IDs must be unique")
        subject_groups = (
            set(self.required_eeg_subject_ids),
            set(self.required_training_eeg_subject_ids),
            set(self.required_validation_eeg_subject_ids),
        )
        if any(
            left & right
            for index, left in enumerate(subject_groups)
            for right in subject_groups[index + 1 :]
        ):
            raise ValueError("research expansion EEG subject groups must be disjoint")
        return self

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


class ResearchReadinessCheck(BaseModel):
    """One actionable research-evidence requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1, max_length=160)
    ready: bool
    observed: str = Field(min_length=1, max_length=500)
    required: str = Field(min_length=1, max_length=500)
    detail: str = Field(min_length=1, max_length=1_000)


class ResearchExpansionReadiness(BaseModel):
    """Machine-readable capacity and provenance result after Step 8."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    expansion_id: str
    evaluated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_message_count: int = Field(ge=1)
    required_language_trial_count: int = Field(ge=1)
    planned_counterfactual_trial_count: int = Field(ge=1)
    planned_trials_per_eeg_subject: int = Field(ge=1)
    available_p300_trial_count: int = Field(ge=0)
    ready: bool
    checks: tuple[ResearchReadinessCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> ResearchExpansionReadiness:
        if self.ready != all(check.ready for check in self.checks):
            raise ValueError("research readiness must agree with every requirement")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_research_expansion_spec(
    path: str | Path = DEFAULT_RESEARCH_EXPANSION_CONFIG,
) -> ResearchExpansionSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("research expansion config must contain a YAML mapping")
    return ResearchExpansionSpec.model_validate(payload)


def _check(
    check_id: str,
    ready: bool,
    *,
    observed: object,
    required: object,
    detail: str,
) -> ResearchReadinessCheck:
    return ResearchReadinessCheck(
        check_id=check_id,
        ready=ready,
        observed=str(observed),
        required=str(required),
        detail=detail,
    )


def _adapter_check(
    spec: ResearchExpansionSpec,
    *,
    training_digest: str,
) -> ResearchReadinessCheck:
    model = load_local_model_config(spec.language_model_config)
    verified: list[str] = []
    failures: list[str] = []
    for profile_id in spec.required_profile_ids:
        directory = spec.adapter_root / f"{profile_id}{spec.adapter_suffix}"
        try:
            bundle = load_personalization_adapter(
                directory,
                expected_profile_id=profile_id,
                expected_model_id=model.model_id,
                expected_model_revision=model.model_revision,
            )
        except (OSError, ValueError) as error:
            failures.append(f"{profile_id}: {error}")
            continue
        manifest = bundle.manifest
        if (
            manifest.trainer_revision != "neuroselect-mlx-lora-v1"
            or manifest.training_config_sha256 != training_digest
            or not manifest.validation_evaluated
            or not manifest.test_evaluated
        ):
            failures.append(f"{profile_id}: adapter is not research-config eligible")
            continue
        verified.append(profile_id)
    return _check(
        "research-adapters",
        len(verified) == len(spec.required_profile_ids),
        observed=f"{len(verified)}/{len(spec.required_profile_ids)} verified",
        required="one research-config adapter per required profile",
        detail=(
            "All research adapters are present and checksum-valid."
            if not failures
            else "; ".join(failures)
        ),
    )


def _language_artifact_check(
    spec: ResearchExpansionSpec,
    *,
    language_config_sha256: str,
    required_trial_count: int,
) -> ResearchReadinessCheck:
    try:
        result, manifest = read_held_out_language_artifacts(spec.language_artifacts)
    except (OSError, ValueError) as error:
        return _check(
            "full-language-evaluation",
            False,
            observed="missing or invalid",
            required=f"{required_trial_count} held-out trials, claim eligible",
            detail=str(error),
        )
    profiles = {trial.profile_id for trial in result.trials}
    clean = manifest.metadata.get("working_tree_dirty") is False
    ready = (
        result.config_sha256 == language_config_sha256
        and result.claim_eligible
        and len(result.trials) == required_trial_count
        and profiles == set(spec.required_profile_ids)
        and (clean or not spec.require_clean_sources)
    )
    return _check(
        "full-language-evaluation",
        ready,
        observed=(
            f"{len(result.trials)} trials, claim eligible={result.claim_eligible}, clean={clean}"
        ),
        required=f"{required_trial_count} trials, all profiles, claim eligible and clean",
        detail=(
            "The full held-out language artifact satisfies the research recipe."
            if ready
            else "The language artifact is limited, dirty, or uses a different recipe."
        ),
    )


@dataclass(frozen=True)
class _DecoderState:
    trial_count: int
    trial_counts_by_subject: dict[str, int]
    training_subject_ids: set[str]
    validation_subject_ids: set[str]
    clean: bool
    detail: str


def _decoder_state(spec: ResearchExpansionSpec) -> _DecoderState:
    directory = spec.decoder_artifacts
    try:
        manifest = RunManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        evaluation_path = directory / "evaluation.json"
        metadata_path = directory / "decoder.json"
        evaluation_output = next(
            item for item in manifest.outputs if item.uri == "artifact://evaluation.json"
        )
        metadata_output = next(
            item for item in manifest.outputs if item.uri == "artifact://decoder.json"
        )
        if _sha256_file(evaluation_path) != evaluation_output.sha256:
            raise ValueError("decoder evaluation checksum does not match its manifest")
        if _sha256_file(metadata_path) != metadata_output.sha256:
            raise ValueError("decoder metadata checksum does not match its manifest")
        if manifest.run_kind is not RunKind.EEG_ORIGINAL_TASK:
            raise ValueError("decoder artifact has the wrong run kind")
        evaluation = DecoderEvaluation.model_validate_json(
            evaluation_path.read_text(encoding="utf-8")
        )
        metadata = DecoderCheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if manifest.config_sha256 != metadata.config.digest():
            raise ValueError("decoder metadata configuration does not match its manifest")
        flash_trials = flash_trials_from_decoder_evaluation(evaluation)
    except (OSError, StopIteration, ValueError) as error:
        return _DecoderState(0, {}, set(), set(), False, str(error))
    trial_counts_by_subject: dict[str, int] = {}
    for trial in flash_trials:
        trial_counts_by_subject[trial.subject_id] = (
            trial_counts_by_subject.get(trial.subject_id, 0) + 1
        )
    return _DecoderState(
        trial_count=len(flash_trials),
        trial_counts_by_subject=trial_counts_by_subject,
        training_subject_ids=set(metadata.training_summary.training_subject_ids),
        validation_subject_ids=set(metadata.training_summary.calibration_subject_ids),
        clean=manifest.metadata.get("working_tree_dirty") is False,
        detail=(
            "Decoder evaluation and metadata checksums verified without loading the checkpoint."
        ),
    )


def assess_research_expansion(
    spec: ResearchExpansionSpec,
) -> ResearchExpansionReadiness:
    """Assess full-language and balanced P300 capacity without starting training."""

    benchmark = generate_from_sources(spec.benchmark_spec, spec.profiles)
    messages = tuple(
        message
        for message in benchmark.messages[BenchmarkSplit.TEST]
        if message.profile_id in spec.required_profile_ids
    )
    required_trial_count = sum(len(message.target_spans) for message in messages)
    language_spec = load_held_out_language_spec(spec.language_evaluation_config)
    training_spec = load_lora_training_config(spec.language_training_config)
    preparation_spec = load_counterfactual_preparation_spec(spec.counterfactual_preparation_config)
    fusion_spec = load_counterfactual_spec(spec.counterfactual_fusion_config)
    decoder = _decoder_state(spec)
    planned_trial_count = preparation_spec.planned_counterfactual_trial_count
    planned_trials_per_subject = preparation_spec.planned_trials_per_eeg_subject
    if planned_trial_count is None or planned_trials_per_subject is None:
        raise ValueError("research preparation must define a balanced sample size")
    p300_subjects = set(decoder.trial_counts_by_subject)
    capacity_shortfalls = {
        subject_id: planned_trials_per_subject - decoder.trial_counts_by_subject.get(subject_id, 0)
        for subject_id in spec.required_eeg_subject_ids
        if decoder.trial_counts_by_subject.get(subject_id, 0) < planned_trials_per_subject
    }
    checks = [
        _check(
            "language-protocol",
            (
                language_spec.evidence_tier == "research"
                and language_spec.maximum_messages_per_profile is None
            ),
            observed=(
                f"tier={language_spec.evidence_tier}, "
                f"limit={language_spec.maximum_messages_per_profile}"
            ),
            required="research tier with no message limit",
            detail="The held-out language recipe must evaluate the complete test split.",
        ),
        _check(
            "lora-training-protocol",
            (
                training_spec.trainer_revision == "neuroselect-mlx-lora-v1"
                and training_spec.validation_batches != 0
                and training_spec.test_batches != 0
                and training_spec.evaluate_test
            ),
            observed=training_spec.trainer_revision,
            required="neuroselect-mlx-lora-v1 with validation and test evaluation",
            detail="Development adapters cannot support research personalization claims.",
        ),
        _check(
            "counterfactual-preparation-protocol",
            (
                preparation_spec.schema_version == "2.0"
                and preparation_spec.evidence_tier == "research"
                and preparation_spec.maximum_messages is None
                and preparation_spec.sampling_revision
                == "subject-profile-balanced-complete-message-v1"
                and preparation_spec.required_profile_ids == spec.required_profile_ids
                and preparation_spec.required_eeg_subject_ids == spec.required_eeg_subject_ids
                and preparation_spec.inference_scope == "study-p-dataset-specific-descriptive"
            ),
            observed=(
                f"sampling={preparation_spec.sampling_revision}, "
                f"planned_trials={planned_trial_count}"
            ),
            required="balanced v2 research sampling with exact profile and EEG strata",
            detail=(
                "Counterfactual preparation must sample complete messages without reusing "
                "recorded selections."
            ),
        ),
        _check(
            "counterfactual-fusion-protocol",
            tuple(condition.value for condition in fusion_spec.conditions)
            == PRIMARY_RESEARCH_CONDITIONS,
            observed=",".join(condition.value for condition in fusion_spec.conditions),
            required=",".join(PRIMARY_RESEARCH_CONDITIONS),
            detail="The primary research matrix is A-F; unsupported snapshots are not synthesized.",
        ),
        _adapter_check(spec, training_digest=training_spec.digest()),
        _language_artifact_check(
            spec,
            language_config_sha256=language_spec.digest(),
            required_trial_count=required_trial_count,
        ),
        _check(
            "decoder-provenance",
            decoder.clean or not spec.require_clean_sources,
            observed=f"clean={decoder.clean}",
            required="checksum-verified clean decoder evaluation and metadata",
            detail=decoder.detail,
        ),
        _check(
            "decoder-training-subjects",
            set(spec.required_training_eeg_subject_ids) == decoder.training_subject_ids,
            observed=",".join(sorted(decoder.training_subject_ids)) or "none",
            required=",".join(spec.required_training_eeg_subject_ids),
            detail="The decoder must use the complete preregistered training-subject split.",
        ),
        _check(
            "decoder-validation-subjects",
            set(spec.required_validation_eeg_subject_ids) == decoder.validation_subject_ids,
            observed=",".join(sorted(decoder.validation_subject_ids)) or "none",
            required=",".join(spec.required_validation_eeg_subject_ids),
            detail="Calibration must use the complete preregistered validation-subject split.",
        ),
        _check(
            "held-out-eeg-subjects",
            set(spec.required_eeg_subject_ids) == p300_subjects,
            observed=",".join(sorted(p300_subjects)) or "none",
            required=",".join(spec.required_eeg_subject_ids),
            detail="Every held-out EEG subject must contribute labeled timed replay trials.",
        ),
        _check(
            "balanced-p300-capacity",
            not capacity_shortfalls,
            observed=",".join(
                f"{subject_id}:{decoder.trial_counts_by_subject.get(subject_id, 0)}"
                for subject_id in spec.required_eeg_subject_ids
            ),
            required=(
                f"at least {planned_trials_per_subject} distinct trials per held-out subject "
                f"({planned_trial_count} total)"
            ),
            detail=(
                "Every profile/subject cell can receive its preregistered complete messages."
                if not capacity_shortfalls
                else "Subject shortfalls: "
                + ", ".join(
                    f"{subject_id}={shortfall}"
                    for subject_id, shortfall in capacity_shortfalls.items()
                )
            ),
        ),
    ]
    return ResearchExpansionReadiness(
        expansion_id=spec.expansion_id,
        evaluated_at=spec.evaluated_at,
        config_sha256=spec.digest(),
        benchmark_source_sha256=benchmark.source_sha256,
        required_message_count=len(messages),
        required_language_trial_count=required_trial_count,
        planned_counterfactual_trial_count=planned_trial_count,
        planned_trials_per_eeg_subject=planned_trials_per_subject,
        available_p300_trial_count=decoder.trial_count,
        ready=all(check.ready for check in checks),
        checks=tuple(checks),
    )
