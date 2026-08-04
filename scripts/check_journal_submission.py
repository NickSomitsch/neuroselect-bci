"""Verify a generated RBET or Neuroinformatics submission package."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.publication.submission import verify_journal_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    inventory = verify_journal_submission(
        args.output,
        require_ready=not args.allow_pending,
        repository=Path.cwd(),
    )
    print("Journal submission package verified")
    print(f"Journal: {inventory.journal}")
    print(f"Submission ready: {inventory.submission_ready}")
    print(f"Archive SHA-256: {inventory.archive_sha256}")


if __name__ == "__main__":
    main()
