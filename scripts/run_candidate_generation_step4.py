"""Run locked candidate-generation ablations and the opening robustness benchmark."""

from __future__ import annotations

import argparse
import hashlib
import resource
import subprocess
import time
from pathlib import Path

from neuroselect.evaluation.candidate_generation_step4 import (
    DEFAULT_CANDIDATE_GENERATION_STEP4_CONFIG,
    CandidateGenerationDataset,
    CandidateGenerationMethod,
    evaluate_candidate_generation_step4,
    existing_evaluation_spans,
    load_candidate_generation_step4_spec,
    robustness_evaluation_spans,
    validate_robustness_opening_holdout,
)
from neuroselect.evaluation.candidate_generation_step4_artifacts import (
    read_candidate_generation_step4_artifacts,
    write_candidate_generation_step4_artifacts,
)
from neuroselect.evaluation.candidate_generation_v2 import (
    TargetBlindContextualGeneratorV2,
    build_candidate_bank_v2,
    load_candidate_generation_v2_spec,
)
from neuroselect.evaluation.candidate_generation_v2_artifacts import (
    read_candidate_generation_v2_artifacts,
)
from neuroselect.publication import load_publication_protocol
from neuroselect.synthetic import BenchmarkSplit, generate_from_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CANDIDATE_GENERATION_STEP4_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/publication/candidate-generation-step4-v1"),
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
    spec = load_candidate_generation_step4_spec(args.config)
    protocol = load_publication_protocol(spec.publication_protocol)
    if protocol.digest() != spec.expected_protocol_sha256:
        raise ValueError("publication protocol differs from the Step 4 pin")
    if (
        protocol.analysis_commitments.candidate_v2_role != "exploratory_supplement"
        or not protocol.analysis_commitments.outcome_based_omission_forbidden
    ):
        raise ValueError("publication protocol does not permit the locked Step 4 analysis")

    step3_result, existing_bank, step3_manifest = read_candidate_generation_v2_artifacts(
        spec.step3_artifacts
    )
    v2_spec = load_candidate_generation_v2_spec()
    if step3_manifest.digest() != spec.expected_step3_manifest_sha256:
        raise ValueError("Step 3 manifest differs from the Step 4 pin")
    if (
        step3_result.config_sha256 != spec.expected_step3_config_sha256
        or v2_spec.digest() != spec.expected_step3_config_sha256
    ):
        raise ValueError("Step 3 recipe differs from the Step 4 pin")

    # Reproduce every frozen Step 3 candidate set before applying any ablation.
    full_reference = TargetBlindContextualGeneratorV2(existing_bank, v2_spec)
    for trial in step3_result.trials:
        regenerated = full_reference.generate(
            profile_id=trial.profile_id,
            confirmed_context=trial.confirmed_context,
            span_index=trial.span_index,
        )
        if tuple(candidate.text for candidate in regenerated) != tuple(
            candidate.text for candidate in trial.candidates
        ):
            raise ValueError("full_v2 no longer reproduces the frozen Step 3 candidates")

    robustness_benchmark = generate_from_sources(
        spec.robustness_benchmark_spec,
        spec.profiles_directory,
    )
    if robustness_benchmark.source_sha256 != spec.expected_robustness_source_sha256:
        raise ValueError("robustness benchmark differs from the Step 4 pin")
    validate_robustness_opening_holdout(robustness_benchmark)
    robustness_bank = build_candidate_bank_v2(robustness_benchmark, v2_spec)
    robustness_test_ids = {
        message.message_id for message in robustness_benchmark.messages[BenchmarkSplit.TEST]
    }
    if set(robustness_bank.source_message_ids) & robustness_test_ids:
        raise ValueError("robustness candidate bank leaked test message IDs")

    result = evaluate_candidate_generation_step4(
        spec=spec,
        v2_spec=v2_spec,
        existing_spans=existing_evaluation_spans(step3_result.trials),
        existing_bank=existing_bank,
        robustness_spans=robustness_evaluation_spans(robustness_benchmark),
        robustness_bank=robustness_bank,
        protocol_sha256=protocol.digest(),
        step3_manifest_sha256=step3_manifest.digest(),
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_candidate_generation_step4_artifacts(
        result,
        existing_bank,
        robustness_bank,
        robustness_benchmark,
        spec,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    restored, restored_existing, restored_robustness, restored_manifest = (
        read_candidate_generation_step4_artifacts(args.output)
    )
    assert (restored, restored_existing, restored_robustness, restored_manifest) == (
        result,
        existing_bank,
        robustness_bank,
        manifest,
    )

    elapsed = time.perf_counter() - started
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    print(f"Run: {result.run_id}")
    print(f"Method trials: {len(result.trials)}")
    print("Methods: " + ", ".join(method.value for method in spec.methods))
    print("Intended targets exposed to generators: no")
    print("Robustness fit/test opening overlap: 0")
    for dataset_id in CandidateGenerationDataset:
        print(f"{dataset_id.value}:")
        for method in CandidateGenerationMethod:
            overall = next(
                metric
                for metric in result.metrics
                if metric.dataset_id is dataset_id
                and metric.method is method
                and metric.scope == "overall"
            )
            opening = next(
                metric
                for metric in result.metrics
                if metric.dataset_id is dataset_id
                and metric.method is method
                and metric.scope == "opening"
            )
            complete = next(
                metric
                for metric in result.metrics
                if metric.dataset_id is dataset_id
                and metric.method is method
                and metric.scope == "complete_messages"
            )
            print(
                f"  {method.value}: overall={overall.availability_rate:.3f} "
                f"opening={opening.availability_rate:.3f} "
                f"complete={complete.availability_rate:.3f}"
            )
    print(f"Performance: elapsed={elapsed:.1f}s, peak_rss={peak_rss_gib:.2f} GiB")
    print(f"Working tree clean: {manifest.metadata['working_tree_dirty'] is False}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
