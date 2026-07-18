"""Deterministic JSON/JSONL evaluation artifacts with checksum provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neuroselect.evaluation.models import ExperimentResult
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def write_experiment_artifacts(
    result: ExperimentResult,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None = None,
) -> RunManifest:
    """Write byte-stable trials, summary, and manifest files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    trials_content = (
        "\n".join(
            _canonical_json(record.model_dump(mode="json")) for record in result.trial_records
        )
        + "\n"
    )
    summary_payload = result.model_dump(mode="json", exclude={"trial_records"})
    summary_payload["result_sha256"] = result.digest()
    summary_payload["trial_record_count"] = len(result.trial_records)
    summary_content = _canonical_json(summary_payload) + "\n"

    trials_path = destination / "trials.jsonl"
    summary_path = destination / "metrics.json"
    trials_path.write_text(trials_content, encoding="utf-8")
    summary_path.write_text(summary_content, encoding="utf-8")

    simulator_config = _canonical_json(result.spec.simulator.model_dump(mode="json"))
    manifest = RunManifest(
        run_id=result.run_id,
        run_kind=RunKind.SIMULATION,
        status=RunStatus.COMPLETED,
        started_at=result.generated_at,
        completed_at=result.generated_at,
        git_sha=git_sha,
        config_sha256=result.config_sha256,
        random_seeds={"global": result.spec.seed, "neural_simulator": result.spec.seed},
        datasets=(
            ArtifactRef(
                artifact_id="synthetic-held-out-benchmark",
                uri=f"synthetic://benchmark/{result.spec.split.value}",
                sha256=result.benchmark_source_sha256,
                revision="synthetic-benchmark-v1",
                license="MIT",
            ),
        ),
        models=(
            ArtifactRef(
                artifact_id="seeded-neural-simulator",
                uri="config://simulation",
                sha256=_sha256(simulator_config),
                revision="seeded-simulator-v1",
            ),
            ArtifactRef(
                artifact_id="controlled-candidate-protocol",
                uri="protocol://controlled-target-presence",
                sha256=_sha256("controlled-target-presence-v1"),
                revision="controlled-proposals-v1",
            ),
        ),
        outputs=(
            ArtifactRef(
                artifact_id="simulated-evaluation-trials",
                uri="artifact://trials.jsonl",
                sha256=_sha256(trials_content),
                revision=result.spec.protocol_revision,
            ),
            ArtifactRef(
                artifact_id="simulated-evaluation-metrics",
                uri="artifact://metrics.json",
                sha256=_sha256(summary_content),
                revision=result.spec.protocol_revision,
            ),
        ),
        metadata={
            "result_sha256": result.digest(),
            "conditions": [condition.value for condition in result.spec.conditions],
            "trial_record_count": len(result.trial_records),
            "latency_kind": "modeled_interaction_time",
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    (destination / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest
