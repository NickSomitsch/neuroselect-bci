"""Run the locked target-blind exploratory candidate-generation v2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import resource
import subprocess
import time
from pathlib import Path

from neuroselect.evaluation.candidate_generation_v2 import (
    DEFAULT_CANDIDATE_GENERATION_V2_CONFIG,
    build_candidate_bank_v2,
    evaluate_candidate_generation_v2,
    load_candidate_generation_v2_spec,
)
from neuroselect.evaluation.candidate_generation_v2_artifacts import (
    read_candidate_generation_v2_artifacts,
    write_candidate_generation_v2_artifacts,
)
from neuroselect.evaluation.language_artifacts import read_held_out_language_artifacts
from neuroselect.publication import load_publication_protocol
from neuroselect.synthetic import BenchmarkSplit, generate_from_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CANDIDATE_GENERATION_V2_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/publication/candidate-generation-v2-exploratory-v1"),
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
    spec = load_candidate_generation_v2_spec(args.config)
    protocol = load_publication_protocol(spec.publication_protocol)
    if protocol.digest() != spec.expected_protocol_sha256:
        raise ValueError("publication protocol differs from the candidate-generation v2 pin")
    if (
        protocol.analysis_commitments.candidate_v2_role != "exploratory_supplement"
        or not protocol.analysis_commitments.outcome_based_omission_forbidden
    ):
        raise ValueError("publication protocol does not permit the locked exploratory analysis")

    baseline, baseline_manifest = read_held_out_language_artifacts(spec.primary_language_artifacts)
    if baseline_manifest.digest() != spec.expected_primary_language_manifest_sha256:
        raise ValueError("primary language manifest differs from the candidate-generation v2 pin")
    benchmark = generate_from_sources(spec.benchmark_spec, spec.profiles_directory)
    if (
        benchmark.source_sha256 != spec.expected_benchmark_source_sha256
        or baseline.benchmark_source_sha256 != benchmark.source_sha256
    ):
        raise ValueError("candidate-generation v2 benchmark differs from the locked primary source")

    bank = build_candidate_bank_v2(benchmark, spec)
    test_ids = {message.message_id for message in benchmark.messages[BenchmarkSplit.TEST]}
    overlap = set(bank.source_message_ids) & test_ids
    if overlap:
        raise ValueError(f"candidate-bank fitting leaked test message IDs: {sorted(overlap)[:3]}")
    result = evaluate_candidate_generation_v2(
        benchmark=benchmark,
        baseline_trials=baseline.trials,
        bank=bank,
        spec=spec,
        primary_language_manifest_sha256=baseline_manifest.digest(),
        protocol_sha256=protocol.digest(),
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_candidate_generation_v2_artifacts(
        result,
        bank,
        spec,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    restored, restored_bank, restored_manifest = read_candidate_generation_v2_artifacts(args.output)
    assert (restored, restored_bank, restored_manifest) == (result, bank, manifest)

    elapsed = time.perf_counter() - started
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    overall = next(metric for metric in result.metrics if metric.scope == "overall")
    assert overall.baseline_message_availability_rate is not None
    assert overall.v2_message_availability_rate is not None
    assert overall.message_availability_delta is not None
    print(f"Run: {result.run_id}")
    print(f"Trials: {len(result.trials)}")
    print(f"Candidate-bank entries: {len(bank.entries)}")
    print("Fit partitions: train, validation")
    print("Test targets exposed to generator: no")
    print(
        f"Availability: v1={overall.baseline_target_availability_rate:.3f} "
        f"v2={overall.v2_target_availability_rate:.3f} "
        f"delta={overall.availability_delta:+.3f}"
    )
    print(
        f"Complete messages: v1={overall.baseline_message_availability_rate:.3f} "
        f"v2={overall.v2_message_availability_rate:.3f} "
        f"delta={overall.message_availability_delta:+.3f}"
    )
    for interval in result.intervals:
        if interval.scope == "overall":
            print(
                f"{interval.metric}: 95% interval "
                f"[{interval.lower_bound:+.3f}, {interval.upper_bound:+.3f}]"
            )
    print(f"Performance: elapsed={elapsed:.1f}s, peak_rss={peak_rss_gib:.2f} GiB")
    print(f"Working tree clean: {manifest.metadata['working_tree_dirty'] is False}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
