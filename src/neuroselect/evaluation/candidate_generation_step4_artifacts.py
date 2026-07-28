"""Checksum-addressed artifacts for the locked candidate-generation Step 4."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.candidate_generation_step4 import (
    CandidateGenerationStep4Result,
    CandidateGenerationStep4Spec,
)
from neuroselect.evaluation.candidate_generation_v2 import CandidateBank
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.synthetic import BenchmarkSplit, GeneratedBenchmark


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


def write_candidate_generation_step4_artifacts(
    result: CandidateGenerationStep4Result,
    existing_bank: CandidateBank,
    robustness_bank: CandidateBank,
    robustness_benchmark: GeneratedBenchmark,
    spec: CandidateGenerationStep4Spec,
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
        "existing-candidate-bank.json": (
            _canonical_json(existing_bank.model_dump(mode="json")) + "\n"
        ),
        "robustness-candidate-bank.json": (
            _canonical_json(robustness_bank.model_dump(mode="json")) + "\n"
        ),
        **{
            f"robustness-{split.value}.jsonl": (
                "\n".join(
                    _canonical_json(message.model_dump(mode="json"))
                    for message in robustness_benchmark.messages[split]
                )
                + "\n"
            )
            for split in BenchmarkSplit
        },
    }
    manifest_path = destination / "manifest.json"
    existing = [
        str(destination / name)
        for name in (*paths, "manifest.json")
        if (destination / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite Step 4 artifacts: {existing}")
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
            "robustness_benchmark": 20260728,
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
                artifact_id="candidate-generation-step3-reference",
                uri="artifact://candidate-generation-v2-exploratory-v1",
                sha256=result.step3_manifest_sha256,
                revision="candidate-generation-v2-exploratory-v1",
            ),
            ArtifactRef(
                artifact_id="candidate-generation-robustness-benchmark",
                uri="synthetic://candidate-robustness-v1/all-splits",
                sha256=result.robustness_source_sha256,
                revision="candidate-robustness-v1",
                license="MIT",
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"candidate-generation-step4-{Path(name).stem}",
                uri=f"artifact://{name}",
                sha256=_sha256(content),
                revision=spec.protocol_revision,
            )
            for name, content in paths.items()
        ),
        metadata={
            "evidence_role": "exploratory_supplement",
            "design_status": result.design_status,
            "intended_target_exposed_to_generators": False,
            "action_stage_conditioning": result.action_stage_conditioning,
            "robustness_opening_overlap_count": result.robustness_opening_overlap_count,
            "trial_count": len(result.trials),
            "existing_candidate_bank_sha256": result.existing_candidate_bank_sha256,
            "robustness_candidate_bank_sha256": result.robustness_candidate_bank_sha256,
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_candidate_generation_step4_artifacts(
    directory: str | Path,
) -> tuple[CandidateGenerationStep4Result, CandidateBank, CandidateBank, RunManifest]:
    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    output_names = tuple(item.uri.removeprefix("artifact://") for item in manifest.outputs)
    for name in output_names:
        content = (source / name).read_text(encoding="utf-8")
        expected = next(
            item.sha256 for item in manifest.outputs if item.uri == f"artifact://{name}"
        )
        if _sha256(content) != expected:
            raise ValueError(f"candidate-generation Step 4 SHA-256 mismatch: {name}")
    result = CandidateGenerationStep4Result.model_validate_json(
        (source / "result.json").read_text(encoding="utf-8")
    )
    existing_bank = CandidateBank.model_validate_json(
        (source / "existing-candidate-bank.json").read_text(encoding="utf-8")
    )
    robustness_bank = CandidateBank.model_validate_json(
        (source / "robustness-candidate-bank.json").read_text(encoding="utf-8")
    )
    datasets = {item.uri: item.sha256 for item in manifest.datasets}
    expected_datasets = {
        "config://publication/offline-methods-v1": result.protocol_sha256,
        "artifact://candidate-generation-v2-exploratory-v1": result.step3_manifest_sha256,
        "synthetic://candidate-robustness-v1/all-splits": result.robustness_source_sha256,
    }
    if (
        manifest.run_kind is not RunKind.EXPLORATORY_EVALUATION
        or manifest.config_sha256 != result.config_sha256
        or existing_bank.digest() != result.existing_candidate_bank_sha256
        or robustness_bank.digest() != result.robustness_candidate_bank_sha256
        or datasets != expected_datasets
    ):
        raise ValueError("candidate-generation Step 4 manifest disagrees with its result")
    return result, existing_bank, robustness_bank, manifest
