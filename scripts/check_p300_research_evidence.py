"""Audit full-split Study P data and decoder evidence for Step 9."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.decoding.research_evidence import (
    DEFAULT_STUDY_P_RESEARCH_CONFIG,
    audit_study_p_research_evidence,
    load_study_p_research_evidence_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STUDY_P_RESEARCH_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/study-p-research-evidence-v1.json"),
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Audit prepared recordings before decoder training.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write and report blockers without returning an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_study_p_research_evidence(
        load_study_p_research_evidence_spec(args.config),
        include_decoder=not args.data_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(audit.canonical_json() + "\n", encoding="utf-8")
    print(f"Step 9 ready: {audit.ready}")
    print(f"Prepared recordings: {audit.prepared_recording_count}")
    print(
        "Usable held-out selections: "
        + ",".join(
            f"{subject_id}:{count}"
            for subject_id, count in audit.usable_test_trials_by_subject.items()
        )
    )
    for check in audit.checks:
        status = "READY" if check.ready else "BLOCKED"
        print(f"[{status}] {check.check_id}: {check.observed}")
        if not check.ready:
            print(f"  Required: {check.required}")
            print(f"  {check.detail}")
    print(f"Audit JSON: {args.output}")
    if not audit.ready and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
