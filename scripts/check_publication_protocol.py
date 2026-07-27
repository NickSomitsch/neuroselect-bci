"""Validate the locked offline-methods publication protocol and source evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.publication import (
    DEFAULT_PUBLICATION_PROTOCOL,
    assess_publication_protocol,
    load_publication_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PUBLICATION_PROTOCOL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/publication-protocol-readiness-v1.json"),
    )
    parser.add_argument(
        "--require-submission-ready",
        action="store_true",
        help="Also fail while external ethics, review, authorship, or funding gates are pending.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assessment = assess_publication_protocol(load_publication_protocol(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(assessment.canonical_json() + "\n", encoding="utf-8")
    print(f"Protocol: {assessment.protocol_id}")
    print(f"Protocol SHA-256: {assessment.protocol_sha256}")
    print(f"Protocol ready: {assessment.protocol_ready}")
    print(f"Submission ready: {assessment.submission_ready}")
    for check in assessment.checks:
        status = "READY" if check.ready else "PENDING"
        scope = "protocol" if check.required_for_protocol else "submission"
        print(f"[{status}] {scope}/{check.check_id}: {check.observed}")
        if not check.ready:
            print(f"  Required: {check.required}")
            print(f"  {check.detail}")
    print(f"Assessment JSON: {args.output}")
    if not assessment.protocol_ready:
        raise SystemExit(2)
    if args.require_submission_ready and not assessment.submission_ready:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
