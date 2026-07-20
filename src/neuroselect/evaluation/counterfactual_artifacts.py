"""Checksum-addressed counterfactual fusion result tables and provenance."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.counterfactual_models import CounterfactualFusionResult
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.ranking import load_ranking_policy


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _csv(rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _output_ref(manifest: RunManifest, uri: str) -> ArtifactRef:
    try:
        return next(item for item in manifest.outputs if item.uri == uri)
    except StopIteration as error:
        raise ValueError(f"counterfactual manifest is missing {uri}") from error


def write_counterfactual_artifacts(
    result: CounterfactualFusionResult,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
) -> RunManifest:
    """Write complete JSON, JSONL, CSV tables, and a cross-checked run manifest."""

    destination = Path(output_dir)
    result_path = destination / "result.json"
    trials_path = destination / "trials.jsonl"
    mappings_path = destination / "mapping-provenance.jsonl"
    metrics_path = destination / "condition-metrics.csv"
    intervals_path = destination / "paired-intervals.csv"
    manifest_path = destination / "manifest.json"
    paths = (
        result_path,
        trials_path,
        mappings_path,
        metrics_path,
        intervals_path,
        manifest_path,
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite counterfactual artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    result_content = _canonical_json(result.model_dump(mode="json")) + "\n"
    trials_content = (
        "\n".join(_canonical_json(item.model_dump(mode="json")) for item in result.trial_records)
        + "\n"
    )
    mappings_content = (
        "\n".join(
            _canonical_json(item.model_dump(mode="json")) for item in result.mapping_provenance
        )
        + "\n"
    )
    overall_metrics = [item for item in result.metrics if item.profile_id is None]
    metric_fields = (
        "condition",
        "trial_count",
        "top_1_candidate_recall",
        "top_3_candidate_recall",
        "selection_completion_rate",
        "correction_rate",
        "abstention_rate",
        "repeat_request_rate",
        "neural_expected_calibration_error",
        "neural_multiclass_brier_score",
        "mean_modeled_latency_seconds",
    )
    metrics_content = _csv(
        [
            {
                field: (item.condition.value if field == "condition" else getattr(item, field))
                for field in metric_fields
            }
            for item in overall_metrics
        ],
        metric_fields,
    )
    interval_fields = (
        "condition",
        "reference_condition",
        "metric",
        "observed_delta",
        "lower_bound",
        "upper_bound",
        "confidence_level",
        "resamples",
    )
    intervals_content = _csv(
        [
            {
                field: (
                    getattr(item, field).value
                    if field in {"condition", "reference_condition"}
                    else getattr(item, field)
                )
                for field in interval_fields
            }
            for item in result.paired_intervals
        ],
        interval_fields,
    )
    contents = {
        result_path: result_content,
        trials_path: trials_content,
        mappings_path: mappings_content,
        metrics_path: metrics_content,
        intervals_path: intervals_content,
    }
    for path, content in contents.items():
        path.write_text(content, encoding="utf-8")

    package_versions, device = capture_runtime_environment()
    ranking_policy = load_ranking_policy()
    ranking_payload = _canonical_json(ranking_policy.model_dump(mode="json"))
    manifest = RunManifest(
        run_id=result.run_id,
        run_kind=RunKind.COUNTERFACTUAL_REPLAY,
        status=RunStatus.COMPLETED,
        started_at=result.generated_at,
        completed_at=result.generated_at,
        git_sha=git_sha,
        config_sha256=result.config_sha256,
        random_seeds={"global": result.spec.seed, "hierarchical_bootstrap": result.spec.seed},
        package_versions=package_versions,
        device=device,
        datasets=(
            ArtifactRef(
                artifact_id="counterfactual-prepared-input",
                uri="artifact://counterfactual-input",
                sha256=result.input_sha256,
                revision=result.spec.protocol_revision,
            ),
            ArtifactRef(
                artifact_id="source-decoder-manifest",
                uri="artifact://source-decoder-manifest",
                sha256=result.source_decoder_manifest_sha256,
            ),
            ArtifactRef(
                artifact_id="original-task-evaluation",
                uri="artifact://original-task-evaluation",
                sha256=result.original_task_evaluation_sha256,
            ),
        ),
        models=(
            ArtifactRef(
                artifact_id="flash-tile-aggregation",
                uri="config://bci/flash-aggregation",
                sha256=result.spec.aggregation.digest(),
                revision=result.spec.aggregation.aggregation_revision,
            ),
            ArtifactRef(
                artifact_id="transparent-ranking-policy",
                uri="config://ranking",
                sha256=_sha256(ranking_payload),
                revision=ranking_policy.policy_revision,
            ),
            *(
                ArtifactRef(
                    artifact_id=f"personalization-{adapter_id}",
                    uri=f"model://personalization/{adapter_id}",
                    sha256=digest,
                    revision=adapter_id,
                )
                for adapter_id, digest in sorted(result.personalization_adapters.items())
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"counterfactual-{path.stem}",
                uri=f"artifact://{path.name}",
                sha256=_sha256(content),
                revision=result.spec.protocol_revision,
            )
            for path, content in contents.items()
        ),
        metadata={
            "evidence_kind": "offline_counterfactual_replay",
            "claim_eligible": result.claim_eligible,
            "conditions": [condition.value for condition in result.spec.conditions],
            "source_trial_count": len(result.mapping_provenance),
            "condition_trial_count": len(result.trial_records),
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_counterfactual_artifacts(
    directory: str | Path,
) -> tuple[CounterfactualFusionResult, RunManifest]:
    """Verify every output checksum and cross-check result/manifest provenance."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    names = (
        "result.json",
        "trials.jsonl",
        "mapping-provenance.jsonl",
        "condition-metrics.csv",
        "paired-intervals.csv",
    )
    for name in names:
        content = (source / name).read_text(encoding="utf-8")
        expected = _output_ref(manifest, f"artifact://{name}").sha256
        if _sha256(content) != expected:
            raise ValueError(f"counterfactual artifact SHA-256 mismatch: {name}")
    result = CounterfactualFusionResult.model_validate_json(
        (source / "result.json").read_text(encoding="utf-8")
    )
    expected_datasets = {
        "artifact://counterfactual-input": result.input_sha256,
        "artifact://source-decoder-manifest": result.source_decoder_manifest_sha256,
        "artifact://original-task-evaluation": result.original_task_evaluation_sha256,
    }
    ranking_policy = load_ranking_policy()
    ranking_payload = _canonical_json(ranking_policy.model_dump(mode="json"))
    expected_models = {
        "config://bci/flash-aggregation": result.spec.aggregation.digest(),
        "config://ranking": _sha256(ranking_payload),
        **{
            f"model://personalization/{adapter_id}": digest
            for adapter_id, digest in result.personalization_adapters.items()
        },
    }
    if (
        manifest.run_kind is not RunKind.COUNTERFACTUAL_REPLAY
        or manifest.config_sha256 != result.config_sha256
        or {item.uri: item.sha256 for item in manifest.datasets} != expected_datasets
        or {item.uri: item.sha256 for item in manifest.models} != expected_models
    ):
        raise ValueError("counterfactual manifest does not agree with the result")
    return result, manifest
