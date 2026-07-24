"""Fail-closed completeness checks for full-split Study P research evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.decoding.classical import load_classical_decoder_config
from neuroselect.decoding.models import DecoderCheckpointMetadata, DecoderEvaluation
from neuroselect.eeg import EpochMetadata, P300Label, sha256_file
from neuroselect.evaluation.counterfactual import flash_trials_from_decoder_evaluation
from neuroselect.provenance import RunKind, RunManifest

DEFAULT_STUDY_P_RESEARCH_CONFIG = Path("configs/experiments/study_p_research_evidence.yaml")


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class StudyPResearchEvidenceSpec(BaseModel):
    """Exact data and decoder coverage required by Step 9."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    protocol_revision: Literal["study-p-full-subject-split-research-v1"]
    evidence_id: str = Field(min_length=1, max_length=160)
    evaluated_at: datetime
    processed_root: Path
    decoder_artifacts: Path
    decoder_config: Path
    required_sessions: tuple[str, ...] = Field(min_length=1)
    required_recordings_per_subject_per_session: int = Field(ge=1)
    required_training_subject_ids: tuple[str, ...] = Field(min_length=1)
    required_validation_subject_ids: tuple[str, ...] = Field(min_length=1)
    required_test_subject_ids: tuple[str, ...] = Field(min_length=1)
    minimum_usable_test_trials_per_subject: int = Field(ge=1)
    require_clean_decoder: bool = True

    @model_validator(mode="after")
    def validate_protocol(self) -> StudyPResearchEvidenceSpec:
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise ValueError("Study P research evaluation time must include a timezone")
        groups = (
            self.required_training_subject_ids,
            self.required_validation_subject_ids,
            self.required_test_subject_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("Study P research subject IDs must be unique within each split")
        sets = tuple(set(group) for group in groups)
        if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
            raise ValueError("Study P research subject splits must be disjoint")
        if len(self.required_sessions) != len(set(self.required_sessions)):
            raise ValueError("Study P research session IDs must be unique")
        return self

    @property
    def required_subject_ids(self) -> tuple[str, ...]:
        return (
            *self.required_training_subject_ids,
            *self.required_validation_subject_ids,
            *self.required_test_subject_ids,
        )

    @property
    def required_recording_count(self) -> int:
        return (
            len(self.required_subject_ids)
            * len(self.required_sessions)
            * self.required_recordings_per_subject_per_session
        )

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json")).encode()).hexdigest()


class P300ResearchCheck(BaseModel):
    """One auditable Step 9 requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1, max_length=160)
    ready: bool
    observed: str = Field(min_length=1, max_length=1_000)
    required: str = Field(min_length=1, max_length=1_000)
    detail: str = Field(min_length=1, max_length=2_000)


class StudyPResearchEvidenceAudit(BaseModel):
    """Canonical data and decoder completeness result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    evidence_id: str
    evaluated_at: datetime
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    include_decoder: bool
    prepared_recording_count: int = Field(ge=0)
    usable_test_trials_by_subject: dict[str, int]
    data_ready: bool
    decoder_ready: bool | None
    ready: bool
    checks: tuple[P300ResearchCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> StudyPResearchEvidenceAudit:
        expected_data = all(
            check.ready for check in self.checks if check.check_id.startswith("data-")
        )
        if self.data_ready != expected_data:
            raise ValueError("Study P data status must agree with data checks")
        expected_decoder = (
            all(check.ready for check in self.checks if check.check_id.startswith("decoder-"))
            if self.include_decoder
            else None
        )
        if self.decoder_ready != expected_decoder:
            raise ValueError("Study P decoder status must agree with decoder checks")
        if self.ready != (
            self.data_ready and (self.decoder_ready if self.include_decoder else True)
        ):
            raise ValueError("Study P readiness must agree with all required checks")
        return self

    def canonical_json(self) -> str:
        return _canonical_json(self.model_dump(mode="json"))


def load_study_p_research_evidence_spec(
    path: str | Path = DEFAULT_STUDY_P_RESEARCH_CONFIG,
) -> StudyPResearchEvidenceSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("Study P research evidence config must contain a YAML mapping")
    return StudyPResearchEvidenceSpec.model_validate(payload)


def _check(
    check_id: str,
    ready: bool,
    *,
    observed: object,
    required: object,
    detail: str,
) -> P300ResearchCheck:
    return P300ResearchCheck(
        check_id=check_id,
        ready=ready,
        observed=str(observed),
        required=str(required),
        detail=detail,
    )


def _verified_epoch_metadata(directory: Path) -> tuple[EpochMetadata, ...]:
    checksum_payload = json.loads((directory / "epoch-checksums.json").read_text(encoding="utf-8"))
    files = checksum_payload.get("files")
    if (
        checksum_payload.get("schema_version") != "1.0"
        or not isinstance(files, dict)
        or set(files) != {"epochs.npz", "epochs.json"}
    ):
        raise ValueError(f"invalid epoch checksum sidecar: {directory}")
    for name, expected in files.items():
        if not isinstance(expected, str) or sha256_file(directory / name) != expected:
            raise ValueError(f"epoch artifact checksum mismatch: {directory / name}")
    payload = json.loads((directory / "epochs.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.1":
        raise ValueError(f"invalid epoch metadata schema: {directory}")
    return tuple(EpochMetadata.model_validate(item) for item in payload["epochs"])


def _data_checks(
    spec: StudyPResearchEvidenceSpec,
) -> tuple[list[P300ResearchCheck], int, dict[str, int]]:
    checks: list[P300ResearchCheck] = []
    recording_counts: dict[tuple[str, str], int] = defaultdict(int)
    selection_metadata: dict[str, list[EpochMetadata]] = defaultdict(list)
    failures: list[str] = []
    directories = sorted(path.parent for path in spec.processed_root.rglob("epoch-checksums.json"))
    required_subjects = set(spec.required_subject_ids)
    required_sessions = set(spec.required_sessions)
    verified_recordings = 0
    for directory in directories:
        try:
            metadata = _verified_epoch_metadata(directory)
            if not metadata:
                raise ValueError("epoch artifact contains no accepted epochs")
            subjects = {item.subject_id for item in metadata}
            sessions = {item.session_id for item in metadata}
            recordings = {item.recording_id for item in metadata}
            if len(subjects) != 1 or len(sessions) != 1 or len(recordings) != 1:
                raise ValueError("epoch artifact mixes recording provenance")
            subject_id = next(iter(subjects))
            session_id = next(iter(sessions))
            recording_id = next(iter(recordings))
            if (
                subject_id not in required_subjects
                or session_id not in required_sessions
                or "_Train" not in recording_id
            ):
                raise ValueError("epoch artifact lies outside the research Train inventory")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{directory}: {error}")
            continue
        recording_counts[(subject_id, session_id)] += 1
        verified_recordings += 1
        if subject_id in spec.required_test_subject_ids:
            for item in metadata:
                selection_metadata[item.selection_trial_id].append(item)

    expected_per_cell = spec.required_recordings_per_subject_per_session
    cell_mismatches = {
        f"{subject_id}:{session_id}": recording_counts.get((subject_id, session_id), 0)
        for subject_id in spec.required_subject_ids
        for session_id in spec.required_sessions
        if recording_counts.get((subject_id, session_id), 0) != expected_per_cell
    }
    checks.append(
        _check(
            "data-recording-coverage",
            not failures
            and not cell_mismatches
            and verified_recordings == spec.required_recording_count,
            observed=f"{verified_recordings} checksum-verified recordings",
            required=(
                f"{spec.required_recording_count} recordings; "
                f"{expected_per_cell} per subject/session"
            ),
            detail=(
                "Every required Train recording is present and checksum-valid."
                if not failures and not cell_mismatches
                else "; ".join(
                    (
                        *failures[:5],
                        *(f"{cell}={count}" for cell, count in sorted(cell_mismatches.items())),
                    )
                )
            ),
        )
    )

    usable_by_subject = dict.fromkeys(spec.required_test_subject_ids, 0)
    for trial_metadata in selection_metadata.values():
        if (
            trial_metadata
            and all(
                item.label is not P300Label.UNKNOWN
                and item.stimulus_code is not None
                and item.onset_seconds is not None
                for item in trial_metadata
            )
            and any(item.label is P300Label.TARGET for item in trial_metadata)
        ):
            usable_by_subject[trial_metadata[0].subject_id] += 1
    capacity_ready = all(
        usable_by_subject[subject_id] >= spec.minimum_usable_test_trials_per_subject
        for subject_id in spec.required_test_subject_ids
    )
    checks.append(
        _check(
            "data-test-replay-capacity",
            capacity_ready,
            observed=",".join(
                f"{subject_id}:{usable_by_subject[subject_id]}"
                for subject_id in spec.required_test_subject_ids
            ),
            required=(
                f"at least {spec.minimum_usable_test_trials_per_subject} usable selections "
                "per held-out subject"
            ),
            detail=(
                "Every held-out subject can support the balanced Step 8 sample."
                if capacity_ready
                else "One or more held-out subjects lack usable labeled timed selections."
            ),
        )
    )
    return checks, verified_recordings, usable_by_subject


def _decoder_checks(
    spec: StudyPResearchEvidenceSpec,
) -> list[P300ResearchCheck]:
    try:
        directory = spec.decoder_artifacts
        manifest = RunManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        metadata_path = directory / "decoder.json"
        evaluation_path = directory / "evaluation.json"
        metadata_ref = next(
            item for item in manifest.outputs if item.uri == "artifact://decoder.json"
        )
        evaluation_ref = next(
            item for item in manifest.outputs if item.uri == "artifact://evaluation.json"
        )
        if sha256_file(metadata_path) != metadata_ref.sha256:
            raise ValueError("decoder metadata checksum mismatch")
        if sha256_file(evaluation_path) != evaluation_ref.sha256:
            raise ValueError("decoder evaluation checksum mismatch")
        if manifest.run_kind is not RunKind.EEG_ORIGINAL_TASK:
            raise ValueError("decoder artifact has the wrong run kind")
        metadata = DecoderCheckpointMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        evaluation = DecoderEvaluation.model_validate_json(
            evaluation_path.read_text(encoding="utf-8")
        )
        expected_config = load_classical_decoder_config(spec.decoder_config)
        if metadata.config != expected_config or manifest.config_sha256 != expected_config.digest():
            raise ValueError("decoder artifact uses a different tracked configuration")
        flash_trials = flash_trials_from_decoder_evaluation(evaluation)
    except (OSError, StopIteration, ValueError) as error:
        return [
            _check(
                "decoder-artifacts",
                False,
                observed="missing or invalid",
                required="checksum-verified research decoder metadata and evaluation",
                detail=str(error),
            )
        ]

    summary = metadata.training_summary
    trials_by_subject: dict[str, int] = defaultdict(int)
    for trial in flash_trials:
        trials_by_subject[trial.subject_id] += 1
    clean = manifest.metadata.get("working_tree_dirty") is False
    expected_test_subjects = set(spec.required_test_subject_ids)
    checks = [
        _check(
            "decoder-artifacts",
            True,
            observed="checksum-verified",
            required="checksum-verified research decoder metadata and evaluation",
            detail="Decoder JSON artifacts were verified without loading the joblib checkpoint.",
        ),
        _check(
            "decoder-training-subjects",
            set(summary.training_subject_ids) == set(spec.required_training_subject_ids),
            observed=",".join(summary.training_subject_ids),
            required=",".join(spec.required_training_subject_ids),
            detail="Training subjects must exactly match the fixed subject split.",
        ),
        _check(
            "decoder-validation-subjects",
            set(summary.calibration_subject_ids) == set(spec.required_validation_subject_ids),
            observed=",".join(summary.calibration_subject_ids),
            required=",".join(spec.required_validation_subject_ids),
            detail="Calibration subjects must exactly match the fixed subject split.",
        ),
        _check(
            "decoder-test-subjects",
            set(trials_by_subject) == expected_test_subjects,
            observed=",".join(sorted(trials_by_subject)) or "none",
            required=",".join(spec.required_test_subject_ids),
            detail="Evaluation subjects must exactly match the held-out subject split.",
        ),
        _check(
            "decoder-test-replay-capacity",
            all(
                trials_by_subject[subject_id] >= spec.minimum_usable_test_trials_per_subject
                for subject_id in spec.required_test_subject_ids
            ),
            observed=",".join(
                f"{subject_id}:{trials_by_subject[subject_id]}"
                for subject_id in spec.required_test_subject_ids
            ),
            required=(
                f"at least {spec.minimum_usable_test_trials_per_subject} replay trials "
                "per held-out subject"
            ),
            detail="Decoder predictions must retain the balanced replay capacity.",
        ),
        _check(
            "decoder-clean-source",
            clean or not spec.require_clean_decoder,
            observed=f"clean={clean}",
            required="clean tracked source state",
            detail="Research decoder evidence cannot originate from a dirty worktree.",
        ),
    ]
    return checks


def audit_study_p_research_evidence(
    spec: StudyPResearchEvidenceSpec,
    *,
    include_decoder: bool = True,
) -> StudyPResearchEvidenceAudit:
    """Verify complete prepared data and optional decoder evidence."""

    data_checks, recording_count, usable_by_subject = _data_checks(spec)
    decoder_checks = _decoder_checks(spec) if include_decoder else []
    checks = (*data_checks, *decoder_checks)
    data_ready = all(check.ready for check in data_checks)
    decoder_ready = all(check.ready for check in decoder_checks) if include_decoder else None
    ready = data_ready and (bool(decoder_ready) if include_decoder else True)
    return StudyPResearchEvidenceAudit(
        evidence_id=spec.evidence_id,
        evaluated_at=spec.evaluated_at,
        config_sha256=spec.digest(),
        include_decoder=include_decoder,
        prepared_recording_count=recording_count,
        usable_test_trials_by_subject=usable_by_subject,
        data_ready=data_ready,
        decoder_ready=decoder_ready,
        ready=ready,
        checks=checks,
    )
