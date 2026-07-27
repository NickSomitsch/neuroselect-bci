"""Checksum-addressed artifacts for the exploratory candidate-generation v2 analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.candidate_generation_v2 import (
    CandidateBank,
    CandidateGenerationV2Result,
    CandidateGenerationV2Spec,
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


def write_candidate_generation_v2_artifacts(
    result: CandidateGenerationV2Result,
    bank: CandidateBank,
    spec: CandidateGenerationV2Spec,
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
        "candidate-bank.json": _canonical_json(bank.model_dump(mode="json")) + "\n",
        "trials.jsonl": (
            "\n".join(_canonical_json(trial.model_dump(mode="json")) for trial in result.trials)
            + "\n"
        ),
        "metrics.csv": _csv_content([metric.model_dump(mode="json") for metric in result.metrics]),
        "intervals.csv": _csv_content(
            [interval.model_dump(mode="json") for interval in result.intervals]
        ),
    }
    manifest_path = destination / "manifest.json"
    existing = [
        str(destination / name)
        for name in (*paths, "manifest.json")
        if (destination / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite candidate-generation v2 artifacts: {existing}"
        )
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
                artifact_id="frozen-primary-language-result",
                uri="artifact://held-out-language-personalization-research-v1",
                sha256=result.primary_language_manifest_sha256,
                revision="held-out-language-personalization-v1",
            ),
            ArtifactRef(
                artifact_id="synthetic-language-benchmark",
                uri="synthetic://benchmark/all-splits",
                sha256=result.benchmark_source_sha256,
                revision="synthetic-benchmark-v1",
                license="MIT",
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"candidate-generation-v2-{Path(name).stem}",
                uri=f"artifact://{name}",
                sha256=_sha256(content),
                revision=spec.protocol_revision,
            )
            for name, content in paths.items()
        ),
        metadata={
            "evidence_role": "exploratory_supplement",
            "design_status": result.design_status,
            "intended_target_exposed_to_generator": False,
            "fitting_source_splits": list(result.fitting_source_splits),
            "trial_count": len(result.trials),
            "candidate_bank_sha256": result.candidate_bank_sha256,
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_candidate_generation_v2_artifacts(
    directory: str | Path,
) -> tuple[CandidateGenerationV2Result, CandidateBank, RunManifest]:
    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for name in (
        "result.json",
        "candidate-bank.json",
        "trials.jsonl",
        "metrics.csv",
        "intervals.csv",
    ):
        content = (source / name).read_text(encoding="utf-8")
        expected = next(
            item.sha256 for item in manifest.outputs if item.uri == f"artifact://{name}"
        )
        if _sha256(content) != expected:
            raise ValueError(f"candidate-generation v2 SHA-256 mismatch: {name}")
    result = CandidateGenerationV2Result.model_validate_json(
        (source / "result.json").read_text(encoding="utf-8")
    )
    bank = CandidateBank.model_validate_json(
        (source / "candidate-bank.json").read_text(encoding="utf-8")
    )
    datasets = {item.uri: item.sha256 for item in manifest.datasets}
    expected_datasets = {
        "config://publication/offline-methods-v1": result.protocol_sha256,
        "artifact://held-out-language-personalization-research-v1": (
            result.primary_language_manifest_sha256
        ),
        "synthetic://benchmark/all-splits": result.benchmark_source_sha256,
    }
    if (
        manifest.run_kind is not RunKind.EXPLORATORY_EVALUATION
        or manifest.config_sha256 != result.config_sha256
        or bank.digest() != result.candidate_bank_sha256
        or datasets != expected_datasets
    ):
        raise ValueError("candidate-generation v2 manifest does not agree with its result")
    return result, bank, manifest
