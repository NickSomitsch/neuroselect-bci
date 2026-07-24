"""Checksum-verified artifacts for held-out language evaluation."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from neuroselect.evaluation.artifacts import capture_runtime_environment
from neuroselect.evaluation.language_benchmark import HeldOutLanguageResult
from neuroselect.language import load_candidate_risk_policy
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.retrieval import load_retrieval_policy


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_held_out_language_artifacts(
    result: HeldOutLanguageResult,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
    package_versions: dict[str, str] | None = None,
    device: dict[str, str] | None = None,
) -> RunManifest:
    """Write result, trial, metric, and cross-checked manifest artifacts."""

    destination = Path(output_dir)
    result_path = destination / "result.json"
    trials_path = destination / "trials.jsonl"
    metrics_path = destination / "metrics.json"
    manifest_path = destination / "manifest.json"
    paths = (result_path, trials_path, metrics_path, manifest_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite held-out language artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    result_content = _canonical_json(result.model_dump(mode="json")) + "\n"
    trials_content = (
        "\n".join(_canonical_json(trial.model_dump(mode="json")) for trial in result.trials) + "\n"
    )
    metrics_content = (
        _canonical_json([metric.model_dump(mode="json") for metric in result.metrics]) + "\n"
    )
    contents = {
        result_path: result_content,
        trials_path: trials_content,
        metrics_path: metrics_content,
    }
    for path, content in contents.items():
        path.write_text(content, encoding="utf-8")

    captured_packages, captured_device = capture_runtime_environment()
    if package_versions is None:
        package_versions = captured_packages
        mlx_lm_version: str | None = None
        with suppress(PackageNotFoundError):
            mlx_lm_version = version("mlx-lm")
        if mlx_lm_version is not None:
            package_versions = {**package_versions, "mlx-lm": mlx_lm_version}
    risk_policy = load_candidate_risk_policy()
    retrieval_policy = load_retrieval_policy()
    manifest = RunManifest(
        run_id=result.run_id,
        run_kind=RunKind.COMPONENT_EVALUATION,
        status=RunStatus.COMPLETED,
        started_at=result.generated_at,
        completed_at=result.generated_at,
        git_sha=git_sha,
        config_sha256=result.config_sha256,
        random_seeds={"message_selection": result.spec.seed},
        package_versions=package_versions,
        device=captured_device if device is None else device,
        datasets=(
            ArtifactRef(
                artifact_id="synthetic-held-out-language-benchmark",
                uri="synthetic://benchmark/test",
                sha256=result.benchmark_source_sha256,
                revision="synthetic-benchmark-v1",
                license="MIT",
            ),
            *(
                ArtifactRef(
                    artifact_id=f"personalization-corpus-{profile_id}",
                    uri=f"artifact://language-corpus/{profile_id}",
                    sha256=digest,
                    revision="personalization-corpus-v1",
                )
                for profile_id, digest in sorted(result.corpus_manifest_sha256.items())
            ),
        ),
        models=(
            ArtifactRef(
                artifact_id="generic-language-backend",
                uri=f"model://{result.backend.model_id}",
                sha256=_sha256(_canonical_json(result.backend.model_dump(mode="json"))),
                revision=result.backend.model_revision,
            ),
            *(
                ArtifactRef(
                    artifact_id=f"personalization-{manifest.adapter_id}",
                    uri=f"model://personalization/{manifest.adapter_id}",
                    sha256=manifest.adapter_sha256,
                    revision=manifest.adapter_id,
                )
                for manifest in sorted(result.adapters.values(), key=lambda item: item.profile_id)
            ),
            ArtifactRef(
                artifact_id="candidate-risk-policy",
                uri="config://language/risk",
                sha256=_sha256(_canonical_json(risk_policy.model_dump(mode="json"))),
                revision=risk_policy.policy_revision,
            ),
            ArtifactRef(
                artifact_id="retrieval-policy",
                uri="config://retrieval",
                sha256=_sha256(_canonical_json(retrieval_policy.model_dump(mode="json"))),
                revision=retrieval_policy.tokenizer_revision,
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=f"held-out-language-{path.stem}",
                uri=f"artifact://{path.name}",
                sha256=_sha256(content),
                revision=result.spec.protocol_revision,
            )
            for path, content in contents.items()
        ),
        metadata={
            "evidence_kind": "held_out_language_component_evaluation",
            "evidence_tier": result.spec.evidence_tier,
            "claim_eligible": result.claim_eligible,
            "trial_count": len(result.trials),
            "profile_ids": sorted(result.adapters),
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_held_out_language_artifacts(
    directory: str | Path,
) -> tuple[HeldOutLanguageResult, RunManifest]:
    """Verify output checksums and cross-check result provenance."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for name in ("result.json", "trials.jsonl", "metrics.json"):
        content = (source / name).read_text(encoding="utf-8")
        expected = next(
            item.sha256 for item in manifest.outputs if item.uri == f"artifact://{name}"
        )
        if _sha256(content) != expected:
            raise ValueError(f"held-out language artifact SHA-256 mismatch: {name}")
    result = HeldOutLanguageResult.model_validate_json(
        (source / "result.json").read_text(encoding="utf-8")
    )
    adapter_refs = {
        item.uri: item.sha256
        for item in manifest.models
        if item.uri.startswith("model://personalization/")
    }
    expected_adapters = {
        f"model://personalization/{adapter.adapter_id}": adapter.adapter_sha256
        for adapter in result.adapters.values()
    }
    if (
        manifest.run_kind is not RunKind.COMPONENT_EVALUATION
        or manifest.config_sha256 != result.config_sha256
        or adapter_refs != expected_adapters
    ):
        raise ValueError("held-out language manifest does not agree with the result")
    return result, manifest
