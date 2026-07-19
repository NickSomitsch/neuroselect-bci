"""Checksum-addressed artifacts for trusted local classical-decoder checkpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from neuroselect.decoding.classical import CalibratedP300Decoder
from neuroselect.decoding.models import (
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    DecoderTrainingSummary,
)
from neuroselect.eeg import sha256_file, verify_sha256
from neuroselect.evaluation import capture_runtime_environment
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _artifact_by_uri(manifest: RunManifest, uri: str) -> ArtifactRef:
    try:
        return next(item for item in manifest.outputs if item.uri == uri)
    except StopIteration as error:
        raise ValueError(f"decoder manifest is missing {uri}") from error


def write_decoder_artifacts(
    decoder: CalibratedP300Decoder,
    summary: DecoderTrainingSummary,
    evaluation: DecoderEvaluation,
    output_dir: str | Path,
    *,
    git_sha: str,
    run_time: datetime,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
) -> RunManifest:
    """Write a local joblib checkpoint, safe JSON results, and complete run manifest."""

    if run_time.tzinfo is None or run_time.utcoffset() is None:
        raise ValueError("decoder run time must include a timezone")
    if (
        summary.channel_names != decoder.channel_names
        or not decoder.sampling_rate_hz == summary.sampling_rate_hz
        or decoder.epoch_sample_count != summary.epoch_sample_count
        or decoder.preprocessing_config != summary.preprocessing_config
    ):
        raise ValueError("decoder checkpoint and training tensor summary do not agree")
    destination = Path(output_dir)
    model_path = destination / "decoder.joblib"
    metadata_path = destination / "decoder.json"
    evaluation_path = destination / "evaluation.json"
    manifest_path = destination / "manifest.json"
    paths = (model_path, metadata_path, evaluation_path, manifest_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite decoder artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    joblib.dump(decoder, model_path, compress=3)
    checkpoint_metadata = DecoderCheckpointMetadata(
        config=decoder.config,
        training_summary=summary,
    )
    metadata_path.write_text(
        _canonical_json(checkpoint_metadata.model_dump(mode="json")), encoding="utf-8"
    )
    evaluation_path.write_text(
        _canonical_json(evaluation.model_dump(mode="json")), encoding="utf-8"
    )
    package_versions, device = capture_runtime_environment()
    run_material = ":".join(
        (
            decoder.config.digest(),
            summary.training_dataset_sha256,
            summary.calibration_dataset_sha256,
            evaluation.dataset_sha256,
        )
    )
    run_id = f"p300-baseline-{hashlib.sha256(run_material.encode()).hexdigest()[:20]}"
    manifest = RunManifest(
        run_id=run_id,
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
                artifact_id="xdawn-shrinkage-lda-platt",
                uri="model://classical-p300/config",
                sha256=decoder.config.digest(),
                revision=decoder.config.model_revision,
            ),
        ),
        outputs=(
            ArtifactRef(
                artifact_id="classical-p300-checkpoint",
                uri="artifact://decoder.joblib",
                sha256=sha256_file(model_path),
                revision=decoder.config.model_revision,
            ),
            ArtifactRef(
                artifact_id="classical-p300-metadata",
                uri="artifact://decoder.json",
                sha256=sha256_file(metadata_path),
                revision=decoder.config.model_revision,
            ),
            ArtifactRef(
                artifact_id="classical-p300-evaluation",
                uri="artifact://evaluation.json",
                sha256=sha256_file(evaluation_path),
                revision=decoder.config.model_revision,
            ),
        ),
        metadata={
            "labeled_test_epochs": evaluation.labeled_epoch_count,
            "unknown_replay_epochs": evaluation.unknown_epoch_count,
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_decoder_artifacts(
    directory: str | Path,
) -> tuple[CalibratedP300Decoder, DecoderCheckpointMetadata, DecoderEvaluation, RunManifest]:
    """Verify all checksums before loading a checkpoint from a trusted local run directory."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    paths = {
        "artifact://decoder.joblib": source / "decoder.joblib",
        "artifact://decoder.json": source / "decoder.json",
        "artifact://evaluation.json": source / "evaluation.json",
    }
    for uri, path in paths.items():
        verify_sha256(path, _artifact_by_uri(manifest, uri).sha256)
    metadata = DecoderCheckpointMetadata.model_validate_json(
        paths["artifact://decoder.json"].read_text(encoding="utf-8")
    )
    evaluation = DecoderEvaluation.model_validate_json(
        paths["artifact://evaluation.json"].read_text(encoding="utf-8")
    )
    summary = metadata.training_summary
    expected_datasets = {
        "dataset://study-p/model-train": summary.training_dataset_sha256,
        "dataset://study-p/model-validation": summary.calibration_dataset_sha256,
        "dataset://study-p/model-test": evaluation.dataset_sha256,
    }
    manifest_datasets = {item.uri: item.sha256 for item in manifest.datasets}
    if (
        manifest.run_kind is not RunKind.EEG_ORIGINAL_TASK
        or manifest.config_sha256 != metadata.config.digest()
        or manifest_datasets != expected_datasets
        or {item.uri: item.sha256 for item in manifest.models}
        != {"model://classical-p300/config": metadata.config.digest()}
    ):
        raise ValueError("decoder manifest does not agree with checkpoint metadata")
    loaded: Any = joblib.load(paths["artifact://decoder.joblib"])
    if not isinstance(loaded, CalibratedP300Decoder):
        raise ValueError("decoder checkpoint has an unexpected object type")
    if loaded.config != metadata.config or metadata.training_summary.config_sha256 != (
        loaded.config.digest()
    ):
        raise ValueError("decoder checkpoint and JSON metadata do not agree")
    if (
        loaded.channel_names != summary.channel_names
        or loaded.sampling_rate_hz != summary.sampling_rate_hz
        or loaded.epoch_sample_count != summary.epoch_sample_count
        or loaded.preprocessing_config != summary.preprocessing_config
    ):
        raise ValueError("decoder checkpoint and training tensor metadata do not agree")
    return loaded, metadata, evaluation, manifest
