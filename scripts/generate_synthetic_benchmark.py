"""Generate the tracked-recipe synthetic benchmark into ignored artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.synthetic import generate_from_sources, write_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("synthetic_data/benchmark.yaml"),
        help="Path to the versioned benchmark recipe.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("synthetic_data/profiles"),
        help="Directory containing synthetic profile YAML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_benchmark"),
        help="Ignored artifact destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = generate_from_sources(args.spec, args.profiles)
    manifest = write_benchmark(benchmark, args.output)
    total = sum(artifact.message_count for artifact in manifest.artifacts)
    print(f"Generated {total} synthetic messages in {args.output}")
    print(f"Source SHA-256: {manifest.source_sha256}")


if __name__ == "__main__":
    main()
