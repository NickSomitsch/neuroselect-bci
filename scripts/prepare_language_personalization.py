"""Prepare per-profile MLX LoRA corpora from the tracked synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroselect.language import write_all_personalization_corpora
from neuroselect.synthetic import generate_from_sources, load_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("synthetic_data/benchmark.yaml"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("synthetic_data/profiles"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/language/personalization-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.profiles)
    benchmark = generate_from_sources(args.spec, args.profiles)
    manifests = write_all_personalization_corpora(benchmark, profiles, args.output)
    summary = {
        "schema_version": "1.0",
        "output": str(args.output),
        "source_benchmark_sha256": benchmark.source_sha256,
        "profiles": {
            manifest.profile_id: {
                "manifest_sha256": manifest.digest(),
                "example_counts": {
                    artifact.split.value: artifact.example_count for artifact in manifest.artifacts
                },
            }
            for manifest in manifests
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
