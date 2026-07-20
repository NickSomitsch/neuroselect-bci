from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.bci import SimulationConfig
from neuroselect.decoding import (
    BinaryDecoderMetrics,
    ClassicalDecoderConfig,
    DecoderCheckpointMetadata,
    DecoderEvaluation,
    DecoderTrainingSummary,
    EpochPrediction,
)
from neuroselect.eeg import P300Label, PreprocessingConfig
from neuroselect.evaluation import (
    EvaluationCondition,
    SimulatedExperimentRunner,
    load_experiment_spec,
    write_experiment_artifacts,
)
from neuroselect.provenance import ArtifactRef, RunKind, RunManifest, RunStatus
from neuroselect.reporting import (
    EvidenceKind,
    ReportSourceSpec,
    ResearchReportBuilder,
    ResearchReportInputError,
    ResearchReportSpec,
    check_generated_release_report,
    check_tracked_release_files,
    load_research_report_spec,
    read_research_report_artifacts,
    render_research_report_markdown,
    write_research_report_artifacts,
)
from neuroselect.reporting.release import REQUIRED_FILES
from neuroselect.synthetic import generate_from_sources, load_profiles

ROOT = Path(__file__).parents[2]
GENERATED_AT = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
PROFILES = load_profiles(ROOT / "synthetic_data/profiles")
BENCHMARK = generate_from_sources(
    ROOT / "synthetic_data/benchmark.yaml",
    ROOT / "synthetic_data/profiles",
)


def write_simulation_source(directory: Path, *, dirty: bool = False) -> None:
    default = load_experiment_spec(ROOT / "configs/experiments/simulated_vertical_slice.yaml")
    spec = default.model_copy(
        update={
            "experiment_id": "reporting-unit-fixture",
            "profile_ids": ("synthetic-concise",),
            "message_limit_per_profile": 1,
            "conditions": (
                EvaluationCondition.A_BCI_ONLY,
                EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
                EvaluationCondition.CURRENT_SAFE_FUSION,
            ),
            "simulator": SimulationConfig(
                seed=default.seed,
                target_concentration=100.0,
                distractor_concentration=1.0,
                ambiguous_concentration=10.0,
                lapse_probability=0.0,
                ambiguous_probability=0.0,
                timeline_origin=default.simulator.timeline_origin,
            ),
        }
    )
    result = SimulatedExperimentRunner(spec).run(benchmark=BENCHMARK, profiles=PROFILES)
    write_experiment_artifacts(
        result,
        directory,
        git_sha="b4321f7",
        source_tree_sha256="a" * 64 if dirty else None,
    )


def report_spec(
    source: Path,
    *,
    required_missing: Path | None = None,
) -> ResearchReportSpec:
    sources = [
        ReportSourceSpec(
            source_id="controlled-simulation",
            label="Controlled simulation",
            path=source,
            expected_run_kind=RunKind.SIMULATION,
            required=True,
            reference_condition=EvaluationCondition.CURRENT_SAFE_FUSION,
        ),
        ReportSourceSpec(
            source_id="optional-counterfactual",
            label="Optional counterfactual",
            path=source.parent / "missing-counterfactual",
            expected_run_kind=RunKind.COUNTERFACTUAL_REPLAY,
            required=False,
        ),
    ]
    if required_missing is not None:
        sources.append(
            ReportSourceSpec(
                source_id="required-eeg",
                label="Required EEG",
                path=required_missing,
                expected_run_kind=RunKind.EEG_ORIGINAL_TASK,
                required=True,
            )
        )
    return ResearchReportSpec(
        report_id="reporting-unit-fixture",
        title="Reporting unit fixture",
        generated_at=GENERATED_AT,
        bootstrap_resamples=100,
        sources=tuple(sources),
    )


def write_original_task_source(directory: Path) -> None:
    config = ClassicalDecoderConfig()
    metadata = DecoderCheckpointMetadata(
        config=config,
        training_summary=DecoderTrainingSummary(
            model_revision=config.model_revision,
            config_sha256=config.digest(),
            training_dataset_sha256="1" * 64,
            calibration_dataset_sha256="2" * 64,
            training_epoch_count=10,
            calibration_epoch_count=5,
            excluded_unknown_training_count=0,
            excluded_unknown_calibration_count=0,
            training_subject_ids=("P_01",),
            calibration_subject_ids=("P_02",),
            channel_names=("Cz",),
            sampling_rate_hz=128.0,
            epoch_sample_count=16,
            preprocessing_config=PreprocessingConfig(),
        ),
    )
    evaluation = DecoderEvaluation(
        dataset_sha256="3" * 64,
        predictions=(
            EpochPrediction(
                epoch_id="epoch-1",
                event_id="event-1",
                selection_trial_id="trial-1",
                recording_id="P_03:SE002:recording",
                subject_id="P_03",
                session_id="SE002",
                true_label=P300Label.TARGET,
                target_probability=0.9,
                predicted_target=True,
                onset_seconds=0.0,
                stimulus_code=1,
            ),
        ),
        labeled_epoch_count=1,
        unknown_epoch_count=0,
        metrics=BinaryDecoderMetrics(
            auroc=1.0,
            balanced_accuracy=1.0,
            brier_score=0.01,
            negative_log_likelihood=0.1,
            expected_calibration_error=0.1,
        ),
        selection_trial_count=1,
        selection_code_set_accuracy=1.0,
    )
    directory.mkdir(parents=True)
    contents = {
        "decoder.joblib": b"not-loaded-by-reporting",
        "decoder.json": metadata.model_dump_json().encode(),
        "evaluation.json": evaluation.model_dump_json().encode(),
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    manifest = RunManifest(
        run_id="original-task-report-fixture",
        run_kind=RunKind.EEG_ORIGINAL_TASK,
        status=RunStatus.COMPLETED,
        started_at=GENERATED_AT,
        completed_at=GENERATED_AT,
        git_sha="b4321f7",
        config_sha256=config.digest(),
        random_seeds={"global": config.random_seed},
        package_versions={"python": "3.12"},
        device={"system": "test"},
        datasets=(
            ArtifactRef(
                artifact_id="train",
                uri="dataset://study-p/model-train",
                sha256="1" * 64,
            ),
            ArtifactRef(
                artifact_id="validation",
                uri="dataset://study-p/model-validation",
                sha256="2" * 64,
            ),
            ArtifactRef(
                artifact_id="test",
                uri="dataset://study-p/model-test",
                sha256="3" * 64,
            ),
        ),
        models=(
            ArtifactRef(
                artifact_id="classical",
                uri="model://classical-p300/config",
                sha256=config.digest(),
            ),
        ),
        outputs=tuple(
            ArtifactRef(
                artifact_id=name,
                uri=f"artifact://{name}",
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for name, content in contents.items()
        ),
        metadata={"working_tree_dirty": False},
    )
    (directory / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")


def test_report_builds_verified_separate_table_and_paired_intervals(tmp_path: Path) -> None:
    source = tmp_path / "simulation"
    write_simulation_source(source)

    report = ResearchReportBuilder(report_spec(source)).build()

    assert report.release_ready is True
    assert len(report.tables) == 1
    table = report.tables[0]
    assert table.evidence_kind is EvidenceKind.CONTROLLED_SIMULATION
    assert table.claim_eligible is False
    assert len(table.metric_rows) == 3
    assert len(table.intervals) == 4
    assert all(interval.sampling_unit == "profile_then_trial" for interval in table.intervals)
    assert report.missing_sources[0].source_id == "optional-counterfactual"
    markdown = render_research_report_markdown(report)
    assert "Controlled synthetic target-present" in markdown
    assert "Source Git revision: `b4321f7`" in markdown
    assert "Release ready: **yes**" in markdown
    assert "missing (optional)" in markdown


def test_report_artifacts_round_trip_and_detect_tampering(tmp_path: Path) -> None:
    source = tmp_path / "simulation"
    output = tmp_path / "report"
    write_simulation_source(source)
    report = ResearchReportBuilder(report_spec(source)).build()

    manifest = write_research_report_artifacts(report, output, git_sha="b4321f7")
    restored, restored_manifest = read_research_report_artifacts(output)

    assert restored == report
    assert restored_manifest == manifest
    assert manifest.run_kind is RunKind.RESEARCH_REPORT
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_research_report_artifacts(report, output, git_sha="b4321f7")
    invalid_manifest = manifest.model_copy(
        update={"metadata": {**manifest.metadata, "release_ready": False}}
    )
    (output / "manifest.json").write_text(
        invalid_manifest.canonical_json() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest does not agree"):
        read_research_report_artifacts(output)
    (output / "manifest.json").write_text(manifest.canonical_json() + "\n", encoding="utf-8")
    (output / "report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_research_report_artifacts(output)


def test_release_gate_rejects_a_report_built_from_dirty_code(tmp_path: Path) -> None:
    source = tmp_path / "simulation"
    output = tmp_path / "report"
    write_simulation_source(source)
    report = ResearchReportBuilder(report_spec(source)).build()
    write_research_report_artifacts(
        report,
        output,
        git_sha="b4321f7",
        source_tree_sha256="f" * 64,
    )

    assert check_generated_release_report(output) == (
        "generated research report was built from a dirty source tree",
    )


def test_report_marks_required_missing_and_dirty_sources_not_ready(tmp_path: Path) -> None:
    clean_source = tmp_path / "clean"
    dirty_source = tmp_path / "dirty"
    write_simulation_source(clean_source)
    write_simulation_source(dirty_source, dirty=True)

    missing_report = ResearchReportBuilder(
        report_spec(clean_source, required_missing=tmp_path / "missing-eeg")
    ).build()
    dirty_report = ResearchReportBuilder(report_spec(dirty_source)).build()

    assert missing_report.release_ready is False
    assert any(source.required for source in missing_report.missing_sources)
    assert dirty_report.release_ready is False
    assert dirty_report.tables[0].source_tree_dirty is True


def test_report_rejects_source_tampering_and_wrong_kind(tmp_path: Path) -> None:
    source = tmp_path / "simulation"
    write_simulation_source(source)
    (source / "trials.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ResearchReportInputError, match="checksum verification"):
        ResearchReportBuilder(report_spec(source)).build()

    with pytest.raises(ValidationError, match="reference conditions"):
        ReportSourceSpec(
            source_id="invalid",
            label="Invalid",
            path=source,
            expected_run_kind=RunKind.EEG_ORIGINAL_TASK,
            reference_condition=EvaluationCondition.A_BCI_ONLY,
        )
    with pytest.raises(ValidationError, match="cannot consume"):
        ReportSourceSpec(
            source_id="recursive",
            label="Recursive",
            path=source,
            expected_run_kind=RunKind.RESEARCH_REPORT,
        )


def test_report_requires_manifested_simulation_inputs(tmp_path: Path) -> None:
    source = tmp_path / "simulation"
    write_simulation_source(source)
    manifest_path = source / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    invalid_manifest = manifest.model_copy(
        update={
            "outputs": tuple(
                item for item in manifest.outputs if item.uri != "artifact://metrics.json"
            )
        }
    )
    manifest_path.write_text(invalid_manifest.canonical_json() + "\n", encoding="utf-8")

    with pytest.raises(ResearchReportInputError, match="required output identities"):
        ResearchReportBuilder(report_spec(source)).build()


def test_report_reads_original_task_json_without_executing_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "original-task"
    write_original_task_source(source)
    spec = ResearchReportSpec(
        report_id="original-task-report-fixture",
        title="Original-task report fixture",
        generated_at=GENERATED_AT,
        bootstrap_resamples=100,
        sources=(
            ReportSourceSpec(
                source_id="original-task",
                label="Original task",
                path=source,
                expected_run_kind=RunKind.EEG_ORIGINAL_TASK,
                required=True,
            ),
        ),
    )

    report = ResearchReportBuilder(spec).build()

    assert report.release_ready is True
    assert report.tables[0].evidence_kind is EvidenceKind.EEG_ORIGINAL_TASK
    assert report.tables[0].claim_eligible is True
    assert report.tables[0].metric_rows[0].values["auroc"] == 1.0
    assert (source / "decoder.joblib").read_bytes() == b"not-loaded-by-reporting"

    # A stale, undeclared result must not silently enter the report.
    (source / "session-drift.json").write_text("unverified stale data\n", encoding="utf-8")
    rebuilt = ResearchReportBuilder(spec).build()
    assert [row.row_id for row in rebuilt.tables[0].metric_rows] == ["overall"]

    manifest_path = source / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    invalid_training = manifest.datasets[0].model_copy(update={"sha256": "f" * 64})
    invalid_manifest = manifest.model_copy(
        update={"datasets": (invalid_training, *manifest.datasets[1:])}
    )
    manifest_path.write_text(invalid_manifest.canonical_json() + "\n", encoding="utf-8")
    with pytest.raises(ResearchReportInputError, match="manifest does not agree"):
        ResearchReportBuilder(spec).build()


def test_tracked_report_config_is_strict_and_release_metadata_is_complete(
    tmp_path: Path,
) -> None:
    spec = load_research_report_spec(ROOT / "configs/release/research_report.yaml")
    assert spec.sources[0].required is True
    assert spec.reject_dirty_sources is True
    assert len(spec.digest()) == 64

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_research_report_spec(invalid)

    assert check_tracked_release_files() == ()


def test_release_file_check_reports_missing_boundaries_without_crashing(tmp_path: Path) -> None:
    errors = check_tracked_release_files(tmp_path)

    assert "missing or empty release file: pyproject.toml" in errors
    assert "missing or empty release file: ui/package.json" in errors
    assert "missing or empty release file: README.md" in errors
    assert "missing or empty release file: docs/model-card.md" in errors


def test_release_file_check_rejects_ui_version_drift(tmp_path: Path) -> None:
    for relative in REQUIRED_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    ui_package = tmp_path / "ui/package.json"
    ui_package.write_text(
        ui_package.read_text(encoding="utf-8").replace("0.1.0-dev.0", "0.2.0"),
        encoding="utf-8",
    )

    assert "UI package version '0.2.0' does not match project version '0.1.0.dev0'" in (
        check_tracked_release_files(tmp_path)
    )
