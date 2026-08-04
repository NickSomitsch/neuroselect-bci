"""Build a fail-closed RBET or Neuroinformatics journal package."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.publication.submission import (
    DEFAULT_SUBMISSION_CONFIG,
    build_journal_submission,
    load_submission_spec,
    selected_journal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", choices=("rbet", "neuroinformatics", "auto"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_SUBMISSION_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    spec = load_submission_spec(args.config)
    journal = selected_journal(spec) if args.journal == "auto" else args.journal
    output = args.output or Path(f"artifacts/submission/{journal}-v1")
    inventory = build_journal_submission(
        spec,
        journal=journal,
        repository=Path.cwd(),
        output=output,
        allow_pending=args.allow_pending,
        overwrite=args.overwrite,
    )
    print(f"Journal: {inventory.journal}")
    print(f"Article type: {inventory.article_type}")
    print(f"Route: {inventory.route}")
    print(f"Submission ready: {inventory.submission_ready}")
    print(f"Package: {output / inventory.archive_filename}")
    print(f"Package SHA-256: {inventory.archive_sha256}")
    for gate in inventory.pending_gates:
        print(f"BLOCKED: {gate}")


if __name__ == "__main__":
    main()
