"""Safe state-dict artifacts for EEGNet and chronological drift evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuroselect.decoding.models import (
    ChronologicalDriftReport,
    DecoderEvaluation,
    EEGNetCheckpointMetadata,
    EEGNetTrainingSummary,
)
from neuroselect.decoding.neural import EEGNetBinaryModel, EEGNetP300Decoder
from neuroselect.eeg import sha256_file, verify_sha256
from neuroselect.evaluation import capture_runtime_environment
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _artifact_by_uri(manifest: RunManifest, uri: str) -> ArtifactRef:
    try:
        return next(item for item in manifest.outputs if item.uri == uri)
    except StopIteration as error:
        raise ValueError(f"EEGNet manifest is missing {uri}") from error


def _validate_alignment(
    decoder: EEGNetP300Decoder,
    summary: EEGNetTrainingSummary,
) -> None:
    if (
        summary.config_sha256 != decoder.config.digest()
        or summary.model_revision != decoder.config.model_revision
        or summary.channel_names != decoder.channel_names
        or not np.isclose(summary.sampling_rate_hz, decoder.sampling_rate_hz)
        or summary.epoch_sample_count != decoder.epoch_sample_count
        or summary.preprocessing_config != decoder.preprocessing_config
        or not np.isclose(summary.temperature, decoder.temperature)
    ):
        raise ValueError("EEGNet checkpoint and training summary do not agree")


def write_eegnet_artifacts(
    decoder: EEGNetP300Decoder,
    summary: EEGNetTrainingSummary,
    evaluation: DecoderEvaluation,
    output_dir: str | Path,
    *,
    git_sha: str,
    run_time: datetime,
    drift_report: ChronologicalDriftReport | None = None,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
) -> RunManifest:
    """Write tensor-only EEGNet weights and checksum-addressed JSON results."""

    if run_time.tzinfo is None or run_time.utcoffset() is None:
        raise ValueError("EEGNet run time must include a timezone")
    _validate_alignment(decoder, summary)
    if drift_report is not None and drift_report.config_sha256 != decoder.config.digest():
        raise ValueError("drift report and EEGNet configuration do not agree")
    destination = Path(output_dir)
    weights_path = destination / "eegnet.pt"
    metadata_path = destination / "eegnet.json"
    evaluation_path = destination / "evaluation.json"
    drift_path = destination / "session-drift.json"
    manifest_path = destination / "manifest.json"
    paths = [weights_path, metadata_path, evaluation_path, manifest_path]
    if drift_report is not None:
        paths.append(drift_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite EEGNet artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    torch.save(decoder.model.state_dict(), weights_path)
    metadata = EEGNetCheckpointMetadata(
        config=decoder.config,
        training_summary=summary,
        development_groups={
            name: tuple(sorted(values)) for name, values in decoder.development_groups.items()
        },
    )
    metadata_path.write_text(_canonical_json(metadata.model_dump(mode="json")), encoding="utf-8")
    evaluation_path.write_text(
        _canonical_json(evaluation.model_dump(mode="json")), encoding="utf-8"
    )
    if drift_report is not None:
        drift_path.write_text(
            _canonical_json(drift_report.model_dump(mode="json")), encoding="utf-8"
        )
    package_versions, device = capture_runtime_environment()
    run_material = ":".join(
        (
            decoder.config.digest(),
            summary.training_dataset_sha256,
            summary.calibration_dataset_sha256,
            evaluation.dataset_sha256,
            drift_report.protocol_revision if drift_report is not None else "no-drift-report",
        )
    )
    output_refs = [
        ArtifactRef(
            artifact_id="eegnet-weights",
            uri="artifact://eegnet.pt",
            sha256=sha256_file(weights_path),
            revision=decoder.config.model_revision,
        ),
        ArtifactRef(
            artifact_id="eegnet-metadata",
            uri="artifact://eegnet.json",
            sha256=sha256_file(metadata_path),
            revision=decoder.config.model_revision,
        ),
        ArtifactRef(
            artifact_id="eegnet-evaluation",
            uri="artifact://evaluation.json",
            sha256=sha256_file(evaluation_path),
            revision=decoder.config.model_revision,
        ),
    ]
    if drift_report is not None:
        output_refs.append(
            ArtifactRef(
                artifact_id="chronological-session-drift",
                uri="artifact://session-drift.json",
                sha256=sha256_file(drift_path),
                revision=drift_report.protocol_revision,
            )
        )
    manifest = RunManifest(
        run_id=f"p300-eegnet-{hashlib.sha256(run_material.encode()).hexdigest()[:20]}",
        run_kind=RunKind.EEG_ORIGINAL_TASK,
        status=RunStatus.COMPLETED,
        started_at=run_time,
        completed_at=run_time,
        git_sha=git_sha,
        config_sha256=decoder.config.digest(),
        random_seeds={"global": decoder.config.random_seed},
        package_versions=package_versions,
        device=device,
        datasets=(
            ArtifactRef(
                artifact_id="study-p-training-epochs",
                uri="dataset://study-p/model-train",
                sha256=summary.training_dataset_sha256,
                revision="study-p-preprocessed-v2",
                license="CC-BY-4.0",
            ),
            ArtifactRef(
                artifact_id="study-p-calibration-epochs",
                uri="dataset://study-p/model-validation",
                sha256=summary.calibration_dataset_sha256,
                revision="study-p-preprocessed-v2",
                license="CC-BY-4.0",
            ),
            ArtifactRef(
                artifact_id="study-p-test-epochs",
                uri="dataset://study-p/model-test",
                sha256=evaluation.dataset_sha256,
                revision="study-p-preprocessed-v2",
                license="CC-BY-4.0",
            ),
        ),
        models=(
            ArtifactRef(
                artifact_id="eegnet-temperature",
                uri="model://eegnet/config",
                sha256=decoder.config.digest(),
                revision=decoder.config.model_revision,
            ),
        ),
        outputs=tuple(output_refs),
        metadata={
            "labeled_test_epochs": evaluation.labeled_epoch_count,
            "unknown_replay_epochs": evaluation.unknown_epoch_count,
            "chronological_drift_subjects": (
                len(drift_report.subjects) if drift_report is not None else 0
            ),
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_eegnet_artifacts(
    directory: str | Path,
) -> tuple[
    EEGNetP300Decoder,
    EEGNetCheckpointMetadata,
    DecoderEvaluation,
    ChronologicalDriftReport | None,
    RunManifest,
]:
    """Verify checksums and reconstruct EEGNet from tensor-only state data."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "artifact://eegnet.pt": source / "eegnet.pt",
        "artifact://eegnet.json": source / "eegnet.json",
        "artifact://evaluation.json": source / "evaluation.json",
    }
    for uri, path in required.items():
        verify_sha256(path, _artifact_by_uri(manifest, uri).sha256)
    drift_ref = next(
        (item for item in manifest.outputs if item.uri == "artifact://session-drift.json"), None
    )
    drift = None
    if drift_ref is not None:
        drift_path = source / "session-drift.json"
        verify_sha256(drift_path, drift_ref.sha256)
        drift = ChronologicalDriftReport.model_validate_json(drift_path.read_text(encoding="utf-8"))
    metadata = EEGNetCheckpointMetadata.model_validate_json(
        required["artifact://eegnet.json"].read_text(encoding="utf-8")
    )
    evaluation = DecoderEvaluation.model_validate_json(
        required["artifact://evaluation.json"].read_text(encoding="utf-8")
    )
    summary = metadata.training_summary
    expected_datasets = {
        "dataset://study-p/model-train": summary.training_dataset_sha256,
        "dataset://study-p/model-validation": summary.calibration_dataset_sha256,
        "dataset://study-p/model-test": evaluation.dataset_sha256,
    }
    if (
        manifest.run_kind is not RunKind.EEG_ORIGINAL_TASK
        or manifest.config_sha256 != metadata.config.digest()
        or {item.uri: item.sha256 for item in manifest.datasets} != expected_datasets
        or {item.uri: item.sha256 for item in manifest.models}
        != {"model://eegnet/config": metadata.config.digest()}
        or (drift is not None and drift.config_sha256 != metadata.config.digest())
    ):
        raise ValueError("EEGNet manifest does not agree with checkpoint metadata")
    model = EEGNetBinaryModel(
        len(summary.channel_names),
        summary.epoch_sample_count,
        metadata.config,
        np.zeros(len(summary.channel_names), dtype=np.float32),
        np.ones(len(summary.channel_names), dtype=np.float32),
    )
    state: Any = torch.load(required["artifact://eegnet.pt"], map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise ValueError("EEGNet checkpoint must contain only a tensor state dictionary")
    model.load_state_dict(state, strict=True)
    decoder = EEGNetP300Decoder(
        config=metadata.config,
        model=model,
        temperature=summary.temperature,
        channel_names=summary.channel_names,
        sampling_rate_hz=summary.sampling_rate_hz,
        epoch_sample_count=summary.epoch_sample_count,
        preprocessing_config=summary.preprocessing_config,
        development_groups={
            name: frozenset(values) for name, values in metadata.development_groups.items()
        },
        require_subject_disjoint_evaluation=metadata.config.require_subject_disjoint,
    )
    _validate_alignment(decoder, summary)
    return decoder, metadata, evaluation, drift, manifest
