"""Verified, evidence-separated research report generation."""

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
from neuroselect.reporting.release import (
    check_generated_release_report,
    check_tracked_release_files,
    project_version,
)
from neuroselect.reporting.report import (
    ResearchReportBuilder,
    ResearchReportInputError,
    load_research_report_spec,
    read_research_report_artifacts,
    render_research_report_markdown,
    write_research_report_artifacts,
)

__all__ = [
    "EvidenceKind",
    "EvidenceTable",
    "MissingReportSource",
    "ReportInterval",
    "ReportMetricRow",
    "ReportSourceSpec",
    "ResearchReport",
    "ResearchReportBuilder",
    "ResearchReportInputError",
    "ResearchReportSpec",
    "check_generated_release_report",
    "check_tracked_release_files",
    "load_research_report_spec",
    "project_version",
    "read_research_report_artifacts",
    "render_research_report_markdown",
    "write_research_report_artifacts",
]
