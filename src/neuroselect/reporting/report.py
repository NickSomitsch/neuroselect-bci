"""Verify experiment artifacts and render evidence-separated release reports."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import yaml

from neuroselect.decoding.models import (
    ChronologicalDriftReport,
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    EEGNetCheckpointMetadata,
)
from neuroselect.evaluation import (
    ConditionMetrics,
    EvaluationCondition,
    ExperimentResult,
    TrialRecord,
    capture_runtime_environment,
    read_counterfactual_artifacts,
)
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.reporting.models import (
    EvidenceKind,
    EvidenceTable,
    MissingReportSource,
    ReportInterval,
    ReportMetricRow,
    ReportSourceSpec,
    ResearchReport,
    ResearchReportSpec,
)


class ResearchReportInputError(ValueError):
    """Raised when a present report source fails provenance or schema checks."""


@dataclass(frozen=True)
class _VerifiedSource:
    spec: ReportSourceSpec
    directory: Path
    manifest: RunManifest

    @property
    def dirty(self) -> bool:
        return self.manifest.metadata.get("working_tree_dirty") is True


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_text(content: str) -> str:
    return _sha256_bytes(content.encode())


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_research_report_spec(path: str | Path) -> ResearchReportSpec:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("research report configuration must contain a YAML mapping")
    return ResearchReportSpec.model_validate(payload)


def _artifact_path(directory: Path, uri: str) -> Path:
    prefix = "artifact://"
    if not uri.startswith(prefix):
        raise ResearchReportInputError(f"report source output has an unsupported URI: {uri}")
    relative = PurePosixPath(uri.removeprefix(prefix))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ResearchReportInputError(f"report source output has an unsafe URI: {uri}")
    return directory.joinpath(*relative.parts)


def _verify_source(source: ReportSourceSpec) -> _VerifiedSource | MissingReportSource:
    directory = source.path
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return MissingReportSource(
            source_id=source.source_id,
            label=source.label,
            path=str(directory),
            required=source.required,
            reason="manifest.json is absent",
        )
    try:
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResearchReportInputError(
            f"invalid manifest for report source {source.source_id}: {error}"
        ) from error
    if manifest.status is not RunStatus.COMPLETED:
        raise ResearchReportInputError(f"report source {source.source_id} is not completed")
    if manifest.run_kind is not source.expected_run_kind:
        raise ResearchReportInputError(
            f"report source {source.source_id} has run kind {manifest.run_kind.value}, "
            f"expected {source.expected_run_kind.value}"
        )
    dirty_state = manifest.metadata.get("working_tree_dirty")
    source_tree_sha256 = manifest.metadata.get("source_tree_sha256")
    if not isinstance(dirty_state, bool) or dirty_state != (source_tree_sha256 is not None):
        raise ResearchReportInputError(
            f"report source {source.source_id} has inconsistent dirty-tree provenance"
        )
    if source_tree_sha256 is not None and (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_tree_sha256)
    ):
        raise ResearchReportInputError(
            f"report source {source.source_id} has an invalid source-tree checksum"
        )
    for field_name in ("datasets", "models", "outputs"):
        artifacts = getattr(manifest, field_name)
        uris = [artifact.uri for artifact in artifacts]
        identifiers = [artifact.artifact_id for artifact in artifacts]
        if len(uris) != len(set(uris)) or len(identifiers) != len(set(identifiers)):
            raise ResearchReportInputError(
                f"report source {source.source_id} has duplicate {field_name}"
            )
    for artifact in manifest.outputs:
        output_path = _artifact_path(directory, artifact.uri)
        if not output_path.is_file():
            raise ResearchReportInputError(
                f"report source {source.source_id} is missing {artifact.uri}"
            )
        if _sha256_bytes(output_path.read_bytes()) != artifact.sha256:
            raise ResearchReportInputError(
                f"report source {source.source_id} failed checksum verification: {artifact.uri}"
            )
    return _VerifiedSource(spec=source, directory=directory, manifest=manifest)


def _metric_values(metric: ConditionMetrics) -> dict[str, float | int | str | bool | None]:
    payload = metric.model_dump(mode="json", exclude={"condition", "profile_id"})
    return dict(payload)


def _condition_rows(metrics: tuple[ConditionMetrics, ...]) -> tuple[ReportMetricRow, ...]:
    return tuple(
        ReportMetricRow(
            row_id=metric.condition.value,
            label=metric.condition.value,
            values=_metric_values(metric),
        )
        for metric in metrics
        if metric.profile_id is None
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchReportInputError(f"invalid JSONL report source {path}: {error}") from error


def _paired_simulation_intervals(
    *,
    records: tuple[TrialRecord, ...],
    conditions: tuple[EvaluationCondition, ...],
    reference: EvaluationCondition | None,
    source_id: str,
    report_spec: ResearchReportSpec,
) -> tuple[ReportInterval, ...]:
    if reference is None:
        return ()
    if reference not in conditions:
        raise ResearchReportInputError(
            f"reference condition {reference.value} is absent from {source_id}"
        )

    def trial_key(record: TrialRecord) -> tuple[str, str, int]:
        return record.profile_id, record.message_id, record.span_index

    def metric_value(
        record: TrialRecord,
        metric_name: Literal["top_1_candidate_recall", "selection_completion_rate"],
    ) -> float:
        if metric_name == "top_1_candidate_recall":
            return float(record.top_1_correct)
        return float(record.explicit_selection_completed)

    by_key: dict[tuple[EvaluationCondition, tuple[str, str, int]], TrialRecord] = {}
    condition_keys: dict[EvaluationCondition, set[tuple[str, str, int]]] = defaultdict(set)
    for record in records:
        key = trial_key(record)
        indexed_key = record.condition, key
        if indexed_key in by_key:
            raise ResearchReportInputError(
                f"paired condition {record.condition.value} has duplicate trials in {source_id}"
            )
        by_key[indexed_key] = record
        condition_keys[record.condition].add(key)
    reference_records = tuple(record for record in records if record.condition is reference)
    reference_keys = condition_keys[reference]
    for condition in conditions:
        if condition_keys[condition] != reference_keys:
            raise ResearchReportInputError(
                f"paired condition {condition.value} is incomplete in {source_id}"
            )
    profile_trials: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for record in reference_records:
        profile_trials[record.profile_id].append(trial_key(record))
    profiles = tuple(sorted(profile_trials))
    output: list[ReportInterval] = []
    for condition in conditions:
        if condition is reference:
            continue
        for metric_name in (
            "top_1_candidate_recall",
            "selection_completion_rate",
        ):
            observed: list[float] = []
            for reference_record in reference_records:
                key = trial_key(reference_record)
                comparison = by_key.get((condition, key))
                if comparison is None:
                    raise ResearchReportInputError(
                        f"paired condition {condition.value} is incomplete in {source_id}"
                    )
                observed.append(
                    metric_value(comparison, metric_name)
                    - metric_value(reference_record, metric_name)
                )
            material = f"{report_spec.bootstrap_seed}:{source_id}:{condition.value}:{metric_name}"
            seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
            generator = np.random.default_rng(seed)
            samples = np.empty(report_spec.bootstrap_resamples, dtype=np.float64)
            for iteration in range(report_spec.bootstrap_resamples):
                deltas: list[float] = []
                for profile_index in generator.integers(0, len(profiles), size=len(profiles)):
                    profile = profiles[int(profile_index)]
                    keys = profile_trials[profile]
                    for trial_index in generator.integers(0, len(keys), size=len(keys)):
                        key = keys[int(trial_index)]
                        deltas.append(
                            metric_value(by_key[(condition, key)], metric_name)
                            - metric_value(by_key[(reference, key)], metric_name)
                        )
                samples[iteration] = float(np.mean(deltas))
            alpha = (1.0 - report_spec.confidence_level) / 2.0
            output.append(
                ReportInterval(
                    condition=condition.value,
                    reference_condition=reference.value,
                    metric=metric_name,
                    observed_delta=float(np.mean(observed)),
                    lower_bound=float(np.quantile(samples, alpha)),
                    upper_bound=float(np.quantile(samples, 1.0 - alpha)),
                    confidence_level=report_spec.confidence_level,
                    resamples=report_spec.bootstrap_resamples,
                    sampling_unit="profile_then_trial",
                )
            )
    return tuple(output)


def _simulation_table(source: _VerifiedSource, report_spec: ResearchReportSpec) -> EvidenceTable:
    output_uris = {item.uri for item in source.manifest.outputs}
    required_outputs = {"artifact://metrics.json", "artifact://trials.jsonl"}
    if not required_outputs.issubset(output_uris):
        raise ResearchReportInputError("simulation source is missing required output identities")
    metrics_path = source.directory / "metrics.json"
    trials_path = source.directory / "trials.jsonl"
    try:
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchReportInputError(f"invalid simulation summary: {error}") from error
    if not isinstance(summary, dict):
        raise ResearchReportInputError("simulation metrics.json must contain a JSON object")
    result_sha256 = summary.pop("result_sha256", None)
    declared_count = summary.pop("trial_record_count", None)
    trial_payloads = _read_jsonl(trials_path)
    original_result_payload = {**summary, "trial_records": trial_payloads}
    original_result_sha256 = _sha256_text(_canonical_json(original_result_payload))
    try:
        result = ExperimentResult.model_validate(original_result_payload)
    except ValueError as error:
        raise ResearchReportInputError(f"invalid simulation result: {error}") from error
    if declared_count != len(result.trial_records) or result_sha256 != original_result_sha256:
        raise ResearchReportInputError("simulation result count or digest does not agree")
    if (
        source.manifest.run_id != result.run_id
        or source.manifest.config_sha256 != result.config_sha256
        or {item.uri: item.sha256 for item in source.manifest.datasets}
        != {f"synthetic://benchmark/{result.spec.split.value}": result.benchmark_source_sha256}
    ):
        raise ResearchReportInputError("simulation manifest does not agree with the result")
    intervals = _paired_simulation_intervals(
        records=result.trial_records,
        conditions=result.spec.conditions,
        reference=source.spec.reference_condition,
        source_id=source.spec.source_id,
        report_spec=report_spec,
    )
    return EvidenceTable(
        table_id=source.spec.source_id,
        title=source.spec.label,
        evidence_kind=EvidenceKind.CONTROLLED_SIMULATION,
        scope_statement=(
            "Controlled synthetic target-present fusion evaluation; these values are engineering "
            "checks and are not recorded EEG or participant communication performance."
        ),
        source_run_id=result.run_id,
        source_manifest_sha256=source.manifest.digest(),
        source_git_sha=source.manifest.git_sha,
        source_tree_dirty=source.dirty,
        claim_eligible=False,
        metric_rows=_condition_rows(result.metrics),
        intervals=intervals,
        limitations=result.limitations,
    )


def _counterfactual_table(source: _VerifiedSource) -> EvidenceTable:
    try:
        result, manifest = read_counterfactual_artifacts(source.directory)
    except (OSError, ValueError) as error:
        raise ResearchReportInputError(f"invalid counterfactual source: {error}") from error
    if manifest != source.manifest:
        raise ResearchReportInputError("counterfactual manifest changed during report loading")
    intervals = tuple(
        ReportInterval(
            condition=interval.condition.value,
            reference_condition=interval.reference_condition.value,
            metric=interval.metric,
            observed_delta=interval.observed_delta,
            lower_bound=interval.lower_bound,
            upper_bound=interval.upper_bound,
            confidence_level=interval.confidence_level,
            resamples=interval.resamples,
            sampling_unit="subject_then_trial",
        )
        for interval in result.paired_intervals
    )
    return EvidenceTable(
        table_id=source.spec.source_id,
        title=source.spec.label,
        evidence_kind=EvidenceKind.COUNTERFACTUAL_REPLAY,
        scope_statement=(
            "Offline counterfactual candidate replay over recorded flash probabilities; displayed "
            "candidate text was not selected by the source participant."
        ),
        source_run_id=result.run_id,
        source_manifest_sha256=manifest.digest(),
        source_git_sha=manifest.git_sha,
        source_tree_dirty=source.dirty,
        claim_eligible=result.claim_eligible and not source.dirty,
        metric_rows=_condition_rows(result.metrics),
        intervals=intervals,
        limitations=result.limitations,
    )


def _original_task_table(source: _VerifiedSource) -> EvidenceTable:
    evaluation_path = source.directory / "evaluation.json"
    try:
        evaluation = DecoderEvaluation.model_validate_json(
            evaluation_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ResearchReportInputError(f"invalid original-task evaluation: {error}") from error
    models = {item.uri: item.sha256 for item in source.manifest.models}
    output_uris = {item.uri for item in source.manifest.outputs}
    if set(models) == {"model://classical-p300/config"}:
        metadata_path = source.directory / "decoder.json"
        metadata_type: type[DecoderCheckpointMetadata] | type[EEGNetCheckpointMetadata] = (
            DecoderCheckpointMetadata
        )
        expected_model_uri = "model://classical-p300/config"
        model_label = "xDAWN/shrinkage-LDA/Platt"
        required_outputs = {
            "artifact://decoder.joblib",
            "artifact://decoder.json",
            "artifact://evaluation.json",
        }
    elif set(models) == {"model://eegnet/config"}:
        metadata_path = source.directory / "eegnet.json"
        metadata_type = EEGNetCheckpointMetadata
        expected_model_uri = "model://eegnet/config"
        model_label = "EEGNet/temperature"
        required_outputs = {
            "artifact://eegnet.pt",
            "artifact://eegnet.json",
            "artifact://evaluation.json",
        }
    else:
        raise ResearchReportInputError("original-task source has an unsupported model identity")
    if not required_outputs.issubset(output_uris):
        raise ResearchReportInputError("original-task source is missing required output identities")
    try:
        metadata = metadata_type.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResearchReportInputError(f"invalid original-task metadata: {error}") from error
    config = metadata.config
    summary = metadata.training_summary
    expected_datasets = {
        "dataset://study-p/model-train": summary.training_dataset_sha256,
        "dataset://study-p/model-validation": summary.calibration_dataset_sha256,
        "dataset://study-p/model-test": evaluation.dataset_sha256,
    }
    if (
        source.manifest.config_sha256 != config.digest()
        or models[expected_model_uri] != config.digest()
        or {item.uri: item.sha256 for item in source.manifest.datasets} != expected_datasets
    ):
        raise ResearchReportInputError("original-task manifest does not agree with evaluation")
    values: dict[str, float | int | str | bool | None] = {
        "model": model_label,
        "labeled_epoch_count": evaluation.labeled_epoch_count,
        "unknown_epoch_count": evaluation.unknown_epoch_count,
        "selection_trial_count": evaluation.selection_trial_count,
        "selection_code_set_accuracy": evaluation.selection_code_set_accuracy,
    }
    if evaluation.metrics is not None:
        values.update(evaluation.metrics.model_dump(mode="json"))
    rows = [ReportMetricRow(row_id="overall", label="Overall held-out test", values=values)]
    limitations = [
        "Metrics describe the original Study P target/non-target event task, not word decoding.",
        "No confidence interval is generated from the aggregate evaluation artifact alone.",
    ]
    drift_uri = "artifact://session-drift.json"
    if drift_uri in output_uris:
        drift_path = source.directory / "session-drift.json"
        try:
            drift = ChronologicalDriftReport.model_validate_json(
                drift_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ResearchReportInputError(f"invalid session-drift report: {error}") from error
        if drift.config_sha256 != config.digest():
            raise ResearchReportInputError("session-drift report has a different model config")
        rows.append(
            ReportMetricRow(
                row_id="chronological_session_drift",
                label="SE001-to-SE002 chronological drift",
                values={
                    "subject_count": len(drift.subjects),
                    "adapted_subject_count": drift.adapted_subject_count,
                    "fallback_subject_count": drift.fallback_subject_count,
                    "mean_auroc_delta": drift.mean_auroc_delta,
                    "mean_brier_delta": drift.mean_brier_delta,
                },
            )
        )
        limitations.append(
            "Chronological session deltas are descriptive and may be confounded by condition order."
        )
    return EvidenceTable(
        table_id=source.spec.source_id,
        title=source.spec.label,
        evidence_kind=EvidenceKind.EEG_ORIGINAL_TASK,
        scope_statement=(
            "Held-out original-task P300 target/non-target decoding; stimulus-code accuracy is not "
            "NeuroSelect candidate accuracy."
        ),
        source_run_id=source.manifest.run_id,
        source_manifest_sha256=source.manifest.digest(),
        source_git_sha=source.manifest.git_sha,
        source_tree_dirty=source.dirty,
        claim_eligible=evaluation.metrics is not None and not source.dirty,
        metric_rows=tuple(rows),
        limitations=tuple(limitations),
    )


class ResearchReportBuilder:
    """Build a report without loading or executing model checkpoint payloads."""

    def __init__(self, spec: ResearchReportSpec) -> None:
        self.spec = spec

    def build(self) -> ResearchReport:
        tables: list[EvidenceTable] = []
        missing: list[MissingReportSource] = []
        for source_spec in self.spec.sources:
            verified = _verify_source(source_spec)
            if isinstance(verified, MissingReportSource):
                missing.append(verified)
                continue
            if verified.manifest.run_kind is RunKind.SIMULATION:
                tables.append(_simulation_table(verified, self.spec))
            elif verified.manifest.run_kind is RunKind.COUNTERFACTUAL_REPLAY:
                tables.append(_counterfactual_table(verified))
            elif verified.manifest.run_kind is RunKind.EEG_ORIGINAL_TASK:
                tables.append(_original_task_table(verified))
            else:
                raise ResearchReportInputError(
                    f"unsupported report source run kind: {verified.manifest.run_kind.value}"
                )
        required_missing = any(source.required for source in missing)
        disallowed_dirty = self.spec.reject_dirty_sources and any(
            table.source_tree_dirty for table in tables
        )
        release_ready = not required_missing and not disallowed_dirty
        source_material = (
            ":".join(table.source_manifest_sha256 for table in tables)
            + ":"
            + ":".join(source.source_id for source in missing)
        )
        run_material = f"{self.spec.digest()}:{source_material}"
        limitations = (
            "Controlled simulation, original-task EEG, and counterfactual replay are separate "
            "evidence tiers and must not be pooled into one performance estimate.",
            "Bootstrap intervals are descriptive and do not establish clinical utility, "
            "non-inferiority, or population-level efficacy.",
            "A release-ready report requires every configured required source and, under the "
            "tracked policy, clean source worktrees.",
        )
        return ResearchReport(
            run_id=f"research-report-{hashlib.sha256(run_material.encode()).hexdigest()[:20]}",
            generated_at=self.spec.generated_at,
            config_sha256=self.spec.digest(),
            spec=self.spec,
            release_ready=release_ready,
            tables=tuple(tables),
            missing_sources=tuple(missing),
            limitations=limitations,
        )


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|")


MARKDOWN_METRIC_PRIORITY = (
    "model",
    "trial_count",
    "available_trial_count",
    "labeled_epoch_count",
    "unknown_epoch_count",
    "selection_trial_count",
    "target_availability_rate",
    "top_1_candidate_recall",
    "top_3_candidate_recall",
    "top_1_recall_given_available",
    "top_3_recall_given_available",
    "other_fallback_success_rate",
    "selection_completion_rate",
    "final_message_exact_accuracy",
    "words_per_minute",
    "correction_rate",
    "abstention_rate",
    "repeat_request_rate",
    "display_accuracy",
    "incorrect_display_rate",
    "candidate_generation_failure_rate",
    "unintended_word_rate",
    "auroc",
    "balanced_accuracy",
    "brier_score",
    "expected_calibration_error",
    "selection_code_set_accuracy",
    "neural_expected_calibration_error",
    "automatic_selection_violation_count",
    "subject_count",
    "adapted_subject_count",
    "fallback_subject_count",
    "mean_auroc_delta",
    "mean_brier_delta",
)


def render_research_report_markdown(report: ResearchReport) -> str:
    """Render deterministic Markdown with an explicit scope statement for every table."""

    lines = [
        f"# {report.spec.title}",
        "",
        "> NeuroSelect does not decode unrestricted thoughts. No generated message is user "
        "confirmed by an offline experiment.",
        "",
        f"- Report ID: `{report.run_id}`",
        f"- Generated: `{report.generated_at.isoformat()}`",
        f"- Release ready: **{'yes' if report.release_ready else 'no'}**",
        f"- Configuration SHA-256: `{report.config_sha256}`",
        "",
        "## Source status",
        "",
        "| Source | Evidence kind | Status | Clean source | Claim scope eligible |",
        "|---|---|---|---:|---:|",
    ]
    tables = {table.table_id: table for table in report.tables}
    missing = {source.source_id: source for source in report.missing_sources}
    for source in report.spec.sources:
        if source.source_id in tables:
            table = tables[source.source_id]
            lines.append(
                f"| {source.label} | {table.evidence_kind.value} | available | "
                f"{'no' if table.source_tree_dirty else 'yes'} | "
                f"{'yes' if table.claim_eligible else 'no'} |"
            )
        else:
            absent = missing[source.source_id]
            requirement = "required" if absent.required else "optional"
            lines.append(
                f"| {source.label} | {source.expected_run_kind.value} | "
                f"missing ({requirement}) | — | no |"
            )
    for table in report.tables:
        lines.extend(
            [
                "",
                f"## {table.title}",
                "",
                f"**Evidence scope:** {table.scope_statement}",
                "",
                f"Source run: `{table.source_run_id}`  ",
                f"Source Git revision: `{table.source_git_sha}`  ",
                f"Source manifest SHA-256: `{table.source_manifest_sha256}`",
                "",
            ]
        )
        available_columns = {name for row in table.metric_rows for name in row.values}
        columns = tuple(name for name in MARKDOWN_METRIC_PRIORITY if name in available_columns)
        lines.append("| Row | " + " | ".join(columns) + " |")
        lines.append("|---|" + "---:|" * len(columns))
        for row in table.metric_rows:
            values = " | ".join(_format_value(row.values.get(column)) for column in columns)
            lines.append(f"| {row.label} | {values} |")
        if table.intervals:
            lines.extend(
                [
                    "",
                    "### Descriptive paired intervals",
                    "",
                    "| Condition | Reference | Metric | Delta | Lower | Upper | Resamples | Unit |",
                    "|---|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for interval in table.intervals:
                lines.append(
                    f"| {interval.condition} | {interval.reference_condition} | "
                    f"{interval.metric} | {_format_value(interval.observed_delta)} | "
                    f"{_format_value(interval.lower_bound)} | "
                    f"{_format_value(interval.upper_bound)} | {interval.resamples} | "
                    f"{interval.sampling_unit} |"
                )
        lines.extend(["", "### Table limitations", ""])
        lines.extend(f"- {limitation}" for limitation in table.limitations)
    if report.missing_sources:
        lines.extend(["", "## Missing sources", ""])
        lines.extend(
            f"- **{source.label}** (`{source.path}`): {source.reason} "
            f"({'required' if source.required else 'optional'})."
            for source in report.missing_sources
        )
    lines.extend(["", "## Global limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The adjacent `report.json` is the canonical machine-readable report. The run manifest "
            "records every source-manifest digest plus checksums for both report files.",
            "",
        ]
    )
    return "\n".join(lines)


def write_research_report_artifacts(
    report: ResearchReport,
    output_dir: str | Path,
    *,
    git_sha: str,
    source_tree_sha256: str | None = None,
    overwrite: bool = False,
) -> RunManifest:
    """Write canonical JSON/Markdown plus a checksum-addressed report manifest."""

    destination = Path(output_dir)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    manifest_path = destination / "manifest.json"
    paths = (json_path, markdown_path, manifest_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite research report artifacts: {existing}")
    destination.mkdir(parents=True, exist_ok=True)
    json_content = report.canonical_json() + "\n"
    markdown_content = render_research_report_markdown(report)
    json_path.write_text(json_content, encoding="utf-8")
    markdown_path.write_text(markdown_content, encoding="utf-8")
    package_versions, device = capture_runtime_environment()
    source_tables = {table.table_id: table for table in report.tables}
    manifest = RunManifest(
        run_id=report.run_id,
        run_kind=RunKind.RESEARCH_REPORT,
        status=RunStatus.COMPLETED,
        started_at=report.generated_at,
        completed_at=report.generated_at,
        git_sha=git_sha,
        config_sha256=report.config_sha256,
        random_seeds={
            "hierarchical_bootstrap": report.spec.bootstrap_seed,
        },
        package_versions=package_versions,
        device=device,
        datasets=tuple(
            ArtifactRef(
                artifact_id=f"report-source-{source.source_id}",
                uri=f"run://{source.source_id}/manifest",
                sha256=source_tables[source.source_id].source_manifest_sha256,
                revision=source_tables[source.source_id].source_run_id,
            )
            for source in report.spec.sources
            if source.source_id in source_tables
        ),
        outputs=(
            ArtifactRef(
                artifact_id="research-report-json",
                uri="artifact://report.json",
                sha256=_sha256_text(json_content),
                revision=report.schema_version,
            ),
            ArtifactRef(
                artifact_id="research-report-markdown",
                uri="artifact://report.md",
                sha256=_sha256_text(markdown_content),
                revision=report.schema_version,
            ),
        ),
        metadata={
            "report_sha256": report.digest(),
            "release_ready": report.release_ready,
            "evidence_kinds": [table.evidence_kind.value for table in report.tables],
            "missing_sources": [source.source_id for source in report.missing_sources],
            "working_tree_dirty": source_tree_sha256 is not None,
            **(
                {"source_tree_sha256": source_tree_sha256} if source_tree_sha256 is not None else {}
            ),
        },
    )
    manifest_path.write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    return manifest


def read_research_report_artifacts(
    directory: str | Path,
) -> tuple[ResearchReport, RunManifest]:
    """Verify report outputs and cross-check all embedded source-manifest identities."""

    source = Path(directory)
    manifest = RunManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    output_uris = [item.uri for item in manifest.outputs]
    if len(output_uris) != len(set(output_uris)):
        raise ValueError("research report manifest has duplicate outputs")
    outputs = {item.uri: item.sha256 for item in manifest.outputs}
    if set(outputs) != {"artifact://report.json", "artifact://report.md"}:
        raise ValueError("research report manifest has unexpected output identities")
    for name in ("report.json", "report.md"):
        content = (source / name).read_bytes()
        if _sha256_bytes(content) != outputs.get(f"artifact://{name}"):
            raise ValueError(f"research report artifact SHA-256 mismatch: {name}")
    report = ResearchReport.model_validate_json(
        (source / "report.json").read_text(encoding="utf-8")
    )
    if (source / "report.json").read_text(encoding="utf-8") != report.canonical_json() + "\n":
        raise ValueError("research report JSON is not canonical")
    if (source / "report.md").read_text(encoding="utf-8") != render_research_report_markdown(
        report
    ):
        raise ValueError("research report Markdown does not agree with the report")
    expected_inputs = {
        f"run://{table.table_id}/manifest": (
            table.source_manifest_sha256,
            table.source_run_id,
        )
        for table in report.tables
    }
    actual_input_uris = [item.uri for item in manifest.datasets]
    actual_inputs = {item.uri: (item.sha256, item.revision) for item in manifest.datasets}
    dirty_state = manifest.metadata.get("working_tree_dirty")
    source_tree_sha256 = manifest.metadata.get("source_tree_sha256")
    if (
        manifest.status is not RunStatus.COMPLETED
        or manifest.run_kind is not RunKind.RESEARCH_REPORT
        or manifest.run_id != report.run_id
        or manifest.config_sha256 != report.config_sha256
        or manifest.started_at != report.generated_at
        or manifest.completed_at != report.generated_at
        or manifest.random_seeds != {"hierarchical_bootstrap": report.spec.bootstrap_seed}
        or manifest.metadata.get("report_sha256") != report.digest()
        or manifest.metadata.get("release_ready") is not report.release_ready
        or manifest.metadata.get("evidence_kinds")
        != [table.evidence_kind.value for table in report.tables]
        or manifest.metadata.get("missing_sources")
        != [item.source_id for item in report.missing_sources]
        or not isinstance(dirty_state, bool)
        or dirty_state != (source_tree_sha256 is not None)
        or (
            source_tree_sha256 is not None
            and (
                not isinstance(source_tree_sha256, str)
                or len(source_tree_sha256) != 64
                or any(character not in "0123456789abcdef" for character in source_tree_sha256)
            )
        )
        or len(actual_input_uris) != len(set(actual_input_uris))
        or actual_inputs != expected_inputs
    ):
        raise ValueError("research report manifest does not agree with the report")
    return report, manifest
