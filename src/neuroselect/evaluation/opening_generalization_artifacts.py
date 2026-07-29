"""Checksum-addressed artifacts for the opening-generalization experiment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.opening_generalization import (
    OpeningGeneralizationResult,
    OpeningGeneralizationSpec,
    OpeningRecord,
    OpeningSplit,
    OpeningTrainingBank,
)
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _csv_content(rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_opening_generalization_artifacts(
    result: OpeningGeneralizationResult,
    bank: OpeningTrainingBank,
    records: tuple[OpeningRecord, ...],
    spec: OpeningGeneralizationSpec,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
    package_versions: dict[str, str] | None = None,
    device: dict[str, str] | None = None,
) -> RunManifest:
    destination = Path(output_dir)
    paths = {
        "result.json": _canonical_json(result.model_dump(mode="json")) + "\n",
        "trials.jsonl": (
            "\n".join(_canonical_json(trial.model_dump(mode="json")) for trial in result.trials)
            + "\n"
        ),
        "metrics.csv": _csv_content([metric.model_dump(mode="json") for metric in result.metrics]),
        "contrasts.csv": _csv_content(
            [contrast.model_dump(mode="json") for contrast in result.contrasts]
        ),
        "training-bank.json": _canonical_json(bank.model_dump(mode="json")) + "\n",
        **{
            f"benchmark-{split.value}.jsonl": (
                "\n".join(
                    _canonical_json(record.model_dump(mode="json"))
                    for record in records
                    if record.split is split
                )
                + "\n"
            )
            for split in OpeningSplit
        },
    }
    existing = [
        str(destination / name)
        for name in (*paths, "manifest.json")
        if (destination / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite opening-generalization artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in paths.items():
        (destination / name).write_text(content, encoding="utf-8")

    captured_packages, captured_device = capture_runtime_environment()
    manifest = RunManifest(
        run_id=result.run_id,
        run_kind=RunKind.EXPLORATORY_EVALUATION,
        status=RunStatus.COMPLETED,
        started_at=result.generated_at,
        completed_at=result.generated_at,
        git_sha=git_sha,
        config_sha256=result.config_sha256,
        random_seeds={
            "benchmark": 20260728,
            "bootstrap": spec.bootstrap_seed,
        },
        package_versions=captured_packages if package_versions is None else package_versions,
        device=captured_device if device is None else device,
        datasets=(
            ArtifactRef(
                artifact_id="offline-publication-protocol",
                uri="config://publication/offline-methods-v1",
                sha256=result.protocol_sha256,
                revision="offline-methods-v1",
            ),
            ArtifactRef(
                artifact_id="candidate-generation-step4-reference",
                uri="artifact://candidate-generation-step4-v1",
                sha256=result.step4_manifest_sha256,
                revision="candidate-generation-step4-v1",
            ),
            ArtifactRef(
                artifact_id="opening-generalization-source",
                uri="synthetic://opening-generalization-v1",
                sha256=result.benchmark_source_sha256,
                revision="opening-generalization-v1",
                license="MIT",
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"opening-generalization-{Path(name).stem}",
                uri=f"artifact://{name}",
                sha256=_sha256(content),
                revision=spec.protocol_revision,
            )
            for name, content in paths.items()
        ),
        metadata={
            "evidence_role": "exploratory_supplement",
            "design_status": result.design_status,
            "intended_opening_exposed_to_generators": False,
            "downstream_conditioning": result.downstream_conditioning,
            "training_bank_sha256": result.training_bank_sha256,
            **result.holdout_counts,
            "trial_count": len(result.trials),
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    (destination / "manifest.json").write_text(
        manifest.canonical_json() + "\n",
        encoding="utf-8",
    )
    return manifest


def read_opening_generalization_artifacts(
    directory: str | Path,
) -> tuple[OpeningGeneralizationResult, OpeningTrainingBank, RunManifest]:
    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for output in manifest.outputs:
        name = output.uri.removeprefix("artifact://")
        content = (source / name).read_text(encoding="utf-8")
        if _sha256(content) != output.sha256:
            raise ValueError(f"opening-generalization SHA-256 mismatch: {name}")
    result = OpeningGeneralizationResult.model_validate_json(
        (source / "result.json").read_text(encoding="utf-8")
    )
    bank = OpeningTrainingBank.model_validate_json(
        (source / "training-bank.json").read_text(encoding="utf-8")
    )
    datasets = {item.uri: item.sha256 for item in manifest.datasets}
    expected = {
        "config://publication/offline-methods-v1": result.protocol_sha256,
        "artifact://candidate-generation-step4-v1": result.step4_manifest_sha256,
        "synthetic://opening-generalization-v1": result.benchmark_source_sha256,
    }
    if (
        manifest.run_kind is not RunKind.EXPLORATORY_EVALUATION
        or manifest.config_sha256 != result.config_sha256
        or bank.digest() != result.training_bank_sha256
        or datasets != expected
    ):
        raise ValueError("opening-generalization manifest disagrees with its result")
    return result, bank, manifest
