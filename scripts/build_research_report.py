"""Build a checksum-verified, evidence-separated NeuroSelect research report."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from neuroselect.reporting import (
    ResearchReportBuilder,
    load_research_report_spec,
    write_research_report_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/release/research_report.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/neuroselect-research-release-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_state() -> tuple[str, str | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return revision, None
    digest = hashlib.sha256(
        subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        path = Path(raw_path.decode())
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return revision, digest.hexdigest()


def main() -> None:
    args = parse_args()
    spec = load_research_report_spec(args.config)
    report = ResearchReportBuilder(spec).build()
    revision, source_tree_sha256 = git_state()
    manifest = write_research_report_artifacts(
        report,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    print(f"Report: {report.run_id}")
    print(f"Release ready: {report.release_ready}")
    print(f"Available evidence tables: {len(report.tables)}")
    print(f"Missing sources: {len(report.missing_sources)}")
    for missing in report.missing_sources:
        requirement = "required" if missing.required else "optional"
        print(f"- {missing.source_id}: {missing.reason} ({requirement})")
    print(f"Markdown: {args.output / 'report.md'}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
