"""Run paired offline P300 counterfactual fusion from an explicit prepared input."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from neuroselect.evaluation import (
    CounterfactualFusionRunner,
    load_counterfactual_input,
    load_counterfactual_spec,
    write_counterfactual_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prepared JSON containing fixed candidate grids and recorded flash probabilities.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/counterfactual_fusion.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/counterfactual-fusion-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_state() -> tuple[str, str | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return revision, None
    digest = hashlib.sha256(
        subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
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
    return revision, digest.hexdigest()


def main() -> None:
    args = parse_args()
    experiment_input = load_counterfactual_input(args.input)
    spec = load_counterfactual_spec(args.config)
    experiment_input = experiment_input.model_copy(update={"spec": spec})
    result = CounterfactualFusionRunner(experiment_input).run()
    revision, source_tree_sha256 = git_state()
    manifest = write_counterfactual_artifacts(
        result,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    print(f"Run: {result.run_id}")
    print(f"Source trials: {len(result.mapping_provenance)}")
    print(f"Condition trials: {len(result.trial_records)}")
    print(f"Claim eligible: {result.claim_eligible}")
    for metrics in (item for item in result.metrics if item.profile_id is None):
        print(
            f"{metrics.condition.value}: top1={metrics.top_1_candidate_recall:.3f} "
            f"complete={metrics.selection_completion_rate:.3f} "
            f"repeat={metrics.repeat_request_rate:.3f}"
        )
    print(f"Manifest: {args.output / 'manifest.json'}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
