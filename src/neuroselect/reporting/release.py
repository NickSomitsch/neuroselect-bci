"""Static source-tree checks required before a public research release."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
REQUIRED_FILES = (
    "pyproject.toml",
    "ui/package.json",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/dataset-card.md",
    "docs/model-card.md",
    "docs/privacy.md",
    "docs/limitations.md",
    "docs/responsible-use.md",
    "docs/threat-model.md",
    "docs/reproducibility.md",
)
FORBIDDEN_CLAIM_PATTERNS = (
    r"\breads? (?:the )?user'?s? thoughts?\b",
    r"\bclinically proven\b",
)
REQUIRED_BOUNDARIES = {
    "README.md": ("not a mind-reading system", "explicit confirmation"),
    "docs/model-card.md": ("not a medical device", "original-task"),
    "docs/limitations.md": ("does not decode unrestricted thoughts", "not a medical device"),
}


def project_version(root: Path = ROOT) -> str:
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def check_tracked_release_files(root: Path = ROOT) -> tuple[str, ...]:
    """Return every release-source violation without mutating the tree."""

    errors: list[str] = []
    contents: dict[str, str] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing or empty release file: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"unreadable release file {relative}: {error}")
            continue
        if not content.strip():
            errors.append(f"missing or empty release file: {relative}")
            continue
        contents[relative] = content
    version: str | None = None
    if "pyproject.toml" in contents:
        try:
            project = tomllib.loads(contents["pyproject.toml"])
            version = str(project["project"]["version"])
        except (tomllib.TOMLDecodeError, KeyError, TypeError) as error:
            errors.append(f"invalid project version metadata: {error}")
    if "CITATION.cff" in contents:
        try:
            citation = yaml.safe_load(contents["CITATION.cff"])
        except yaml.YAMLError as error:
            errors.append(f"invalid CITATION.cff metadata: {error}")
        else:
            citation_version = citation.get("version") if isinstance(citation, dict) else None
            if version is not None and citation_version != version:
                errors.append(
                    f"CITATION.cff version {citation_version!r} does not match "
                    f"project version {version!r}"
                )
    if "ui/package.json" in contents:
        try:
            ui_package = json.loads(contents["ui/package.json"])
            ui_version = ui_package["version"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            errors.append(f"invalid UI version metadata: {error}")
        else:
            expected_ui_version = version.replace(".dev", "-dev.") if version is not None else None
            if expected_ui_version is not None and ui_version != expected_ui_version:
                errors.append(
                    f"UI package version {ui_version!r} does not match project version {version!r}"
                )
    for relative, content in contents.items():
        if not relative.endswith(".md"):
            continue
        content = content.casefold()
        for pattern in FORBIDDEN_CLAIM_PATTERNS:
            if re.search(pattern, content):
                errors.append(f"unsupported claim pattern {pattern!r} in {relative}")
    for relative, boundaries in REQUIRED_BOUNDARIES.items():
        if relative not in contents:
            continue
        content = contents[relative].casefold()
        for boundary in boundaries:
            if boundary not in content:
                errors.append(f"missing research boundary {boundary!r} in {relative}")
    return tuple(errors)


def check_generated_release_report(directory: Path) -> tuple[str, ...]:
    """Validate a generated report and its own source-tree provenance for release use."""

    from neuroselect.reporting.report import read_research_report_artifacts

    try:
        report, manifest = read_research_report_artifacts(directory)
    except (OSError, ValueError) as error:
        return (f"invalid release report: {error}",)
    errors: list[str] = []
    if not report.release_ready:
        errors.append("generated research report is not release-ready")
    if manifest.metadata.get("working_tree_dirty") is not False:
        errors.append("generated research report was built from a dirty source tree")
    return tuple(errors)
