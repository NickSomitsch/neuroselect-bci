"""Validate tracked files and version metadata required for a NeuroSelect research release."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.reporting import (
    check_generated_release_report,
    check_tracked_release_files,
    project_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional generated report directory that must be release-ready.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = list(check_tracked_release_files())
    if args.report is not None:
        errors.extend(check_generated_release_report(args.report))
    if errors:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
    print(f"Release metadata valid for NeuroSelect {project_version()}.")


if __name__ == "__main__":
    main()
