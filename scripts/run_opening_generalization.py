"""Run the locked hierarchical opening-generalization experiment."""

from __future__ import annotations

import argparse
import hashlib
import resource
import subprocess
import time
from pathlib import Path

from neuroselect.evaluation.candidate_generation_step4_artifacts import (
    read_candidate_generation_step4_artifacts,
)
from neuroselect.evaluation.opening_generalization import (
    DEFAULT_OPENING_GENERALIZATION_CONFIG,
    OpeningChallenge,
    OpeningMethod,
    build_opening_training_bank,
    evaluate_opening_generalization,
    generate_opening_records,
    load_opening_generalization_source,
    load_opening_generalization_spec,
    validate_opening_holdouts,
)
from neuroselect.evaluation.opening_generalization_artifacts import (
    read_opening_generalization_artifacts,
    write_opening_generalization_artifacts,
)
from neuroselect.publication import load_publication_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OPENING_GENERALIZATION_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/publication/opening-generalization-v1"),
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
        text=True,
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
    started = time.perf_counter()
    spec = load_opening_generalization_spec(args.config)
    protocol = load_publication_protocol(spec.publication_protocol)
    if protocol.digest() != spec.expected_protocol_sha256:
        raise ValueError("publication protocol differs from the opening experiment pin")
    if (
        protocol.analysis_commitments.candidate_v2_role != "exploratory_supplement"
        or not protocol.analysis_commitments.outcome_based_omission_forbidden
    ):
        raise ValueError("publication protocol does not permit this exploratory analysis")

    _, _, _, step4_manifest = read_candidate_generation_step4_artifacts(spec.step4_artifacts)
    if step4_manifest.digest() != spec.expected_step4_manifest_sha256:
        raise ValueError("Step 4 manifest differs from the opening experiment pin")
    if step4_manifest.metadata["working_tree_dirty"] is not False:
        raise ValueError("opening experiment requires a clean-source Step 4 reference")

    source = load_opening_generalization_source(spec.benchmark_source)
    if source.digest() != spec.expected_benchmark_source_sha256:
        raise ValueError("opening benchmark source differs from the locked digest")
    if source.candidate_budget != spec.candidate_budget:
        raise ValueError("opening source and experiment candidate budgets differ")
    records = generate_opening_records(source)
    validate_opening_holdouts(records)
    bank = build_opening_training_bank(records, source)
    result = evaluate_opening_generalization(
        spec=spec,
        source=source,
        records=records,
        bank=bank,
        protocol_sha256=protocol.digest(),
        step4_manifest_sha256=step4_manifest.digest(),
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_opening_generalization_artifacts(
        result,
        bank,
        records,
        spec,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    restored, restored_bank, restored_manifest = read_opening_generalization_artifacts(args.output)
    assert (restored, restored_bank, restored_manifest) == (result, bank, manifest)

    elapsed = time.perf_counter() - started
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    print(f"Run: {result.run_id}")
    print(f"Opening trials: {len(result.trials)}")
    print(
        "Holdouts: "
        f"{result.holdout_counts['combination_test_count']} combinations, "
        f"{result.holdout_counts['family_test_count']} paraphrase-family openings"
    )
    print("Fit/test exact opening overlap: 0")
    print("Intended openings exposed to generators: no")
    for challenge in (
        OpeningChallenge.HELDOUT_COMBINATION,
        OpeningChallenge.HELDOUT_PARAPHRASE_FAMILY,
    ):
        print(f"{challenge.value}:")
        for method in OpeningMethod:
            metric = next(
                row
                for row in result.metrics
                if row.challenge == challenge.value
                and row.method is method
                and row.scope == "overall"
            )
            print(
                f"  {method.value}: availability={metric.availability_rate:.3f} "
                f"selections={metric.planned_selections} "
                f"coverage/selection={metric.coverage_per_required_selection:.3f} "
                f"candidate_exposures={metric.mean_candidate_exposures:.1f}"
            )
    print(f"Performance: elapsed={elapsed:.1f}s, peak_rss={peak_rss_gib:.2f} GiB")
    print(f"Working tree clean: {manifest.metadata['working_tree_dirty'] is False}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
