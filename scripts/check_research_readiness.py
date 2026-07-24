"""Check research expansion prerequisites without starting expensive work."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.evaluation.research_readiness import (
    DEFAULT_RESEARCH_EXPANSION_CONFIG,
    assess_research_expansion,
    load_research_expansion_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RESEARCH_EXPANSION_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/research-expansion-readiness-v1.json"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success after reporting blockers; intended for development audits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    readiness = assess_research_expansion(load_research_expansion_spec(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(readiness.canonical_json() + "\n", encoding="utf-8")
    print(f"Research ready: {readiness.ready}")
    print(f"Held-out messages: {readiness.required_message_count}")
    print(f"Full held-out language trials: {readiness.required_language_trial_count}")
    print(
        "Planned balanced counterfactual trials: "
        f"{readiness.planned_counterfactual_trial_count} "
        f"({readiness.planned_trials_per_eeg_subject} per EEG subject)"
    )
    print(f"Available P300 trials: {readiness.available_p300_trial_count}")
    for check in readiness.checks:
        status = "READY" if check.ready else "BLOCKED"
        print(f"[{status}] {check.check_id}: {check.observed}")
        if not check.ready:
            print(f"  Required: {check.required}")
            print(f"  {check.detail}")
    print(f"Readiness JSON: {args.output}")
    if not readiness.ready and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
