"""Create a local SQLite knowledge store from the tracked synthetic profiles."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from neuroselect.retrieval import KnowledgeRecordInput, SQLiteKnowledgeStore
from neuroselect.synthetic import load_profiles

DEFAULT_TIMESTAMP = datetime.fromisoformat("2026-07-17T00:00:00+00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("synthetic_data/profiles"),
        help="Directory containing tracked synthetic persona YAML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic-knowledge.sqlite3"),
        help="Ignored destination for the generated local database.",
    )
    parser.add_argument(
        "--timestamp",
        type=datetime.fromisoformat,
        default=DEFAULT_TIMESTAMP,
        help="Timezone-aware deterministic import timestamp.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing generated database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.replace:
            raise SystemExit(f"output already exists: {args.output}; pass --replace to recreate it")
        args.output.unlink()

    profiles = load_profiles(args.profiles)
    record_count = 0
    with SQLiteKnowledgeStore(args.output) as store:
        for profile in profiles:
            for record in profile.knowledge:
                store.add(
                    profile_id=profile.profile_id,
                    record=KnowledgeRecordInput.model_validate(record.model_dump()),
                    at_time=args.timestamp,
                )
                record_count += 1
    print(f"Imported {record_count} records for {len(profiles)} synthetic profiles")
    print(f"Knowledge store: {args.output}")


if __name__ == "__main__":
    main()
