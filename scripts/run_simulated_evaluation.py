"""Run the controlled held-out simulation matrix and write reproducible artifacts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from neuroselect.evaluation import (
    SimulatedExperimentRunner,
    load_experiment_spec,
    write_experiment_artifacts,
)
from neuroselect.synthetic import generate_from_sources, load_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/simulated_vertical_slice.yaml"),
        help="Versioned simulated experiment configuration.",
    )
    parser.add_argument(
        "--benchmark-spec",
        type=Path,
        default=Path("synthetic_data/benchmark.yaml"),
        help="Tracked synthetic benchmark recipe.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("synthetic_data/profiles"),
        help="Tracked synthetic profile directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/simulated-vertical-slice-v1"),
        help="Ignored destination for JSON, JSONL, and checksum artifacts.",
    )
    return parser.parse_args()


def git_state() -> tuple[str, str | None]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return sha, None

    digest = hashlib.sha256()
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
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
    return sha, digest.hexdigest()


def main() -> None:
    args = parse_args()
    spec = load_experiment_spec(args.config)
    profiles = load_profiles(args.profiles)
    benchmark = generate_from_sources(args.benchmark_spec, args.profiles)
    result = SimulatedExperimentRunner(spec).run(
        benchmark=benchmark,
        profiles=profiles,
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_experiment_artifacts(
        result,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
    )

    print(f"Run: {result.run_id}")
    print(f"Trials: {len(result.trial_records)}")
    for metrics in (item for item in result.metrics if item.profile_id is None):
        print(
            f"{metrics.condition.value}: top1={metrics.top_1_candidate_recall:.3f} "
            f"top3={metrics.top_3_candidate_recall:.3f} "
            f"complete={metrics.selection_completion_rate:.3f} "
            f"abstain={metrics.abstention_rate:.3f} "
            f"repeat={metrics.repeat_request_rate:.3f}"
        )
    print(f"Manifest: {args.output / 'manifest.json'}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
