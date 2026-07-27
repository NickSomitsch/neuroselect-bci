"""Train, calibrate, and evaluate EEGNet plus chronological subject adaptation."""

from __future__ import annotations

import argparse
import hashlib
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from neuroselect.decoding import (
    evaluate_chronological_session_drift,
    evaluate_decoder,
    fit_eegnet_decoder,
    load_eegnet_config,
    load_partitioned_epoch_batches,
    write_eegnet_artifacts,
)
from neuroselect.eeg import DataSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/bigp3bci-study-p/1.0.0"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/decoding/eegnet.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/p300-eegnet-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--batch-limit-per-partition",
        type=int,
        help="Development pilot only: use the first N recording batches in each partition.",
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help=(
            "Skip chronological subject adaptation when only the original-task comparator "
            "is needed."
        ),
    )
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
    started = time.monotonic()
    if args.batch_limit_per_partition is not None and args.batch_limit_per_partition < 1:
        raise ValueError("batch limit must be positive")
    config = load_eegnet_config(args.config)
    batches = load_partitioned_epoch_batches(args.processed_root)
    if args.batch_limit_per_partition is not None:
        batches = {
            partition: rows[: args.batch_limit_per_partition] for partition, rows in batches.items()
        }
    decoder, summary = fit_eegnet_decoder(
        batches[DataSplit.TRAIN], batches[DataSplit.VALIDATION], config
    )
    evaluation = evaluate_decoder(decoder, batches[DataSplit.TEST])
    drift = (
        None
        if args.skip_drift
        else evaluate_chronological_session_drift(decoder, batches[DataSplit.TEST])
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_eegnet_artifacts(
        decoder,
        summary,
        evaluation,
        args.output,
        git_sha=revision,
        run_time=datetime.now(UTC),
        drift_report=drift,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    assert evaluation.metrics is not None
    print(f"Run: {manifest.run_id}")
    print(f"Training device: {summary.training_device}")
    print(f"Selected epoch: {summary.selected_epoch}")
    print(f"Held-subject AUROC: {evaluation.metrics.auroc:.4f}")
    print(f"Held-subject Brier score: {evaluation.metrics.brier_score:.4f}")
    if evaluation.selection_metrics is not None:
        print(f"Target-event recall@K: {evaluation.selection_metrics.target_event_recall_at_k:.4f}")
        print(
            "Target-event average precision: "
            f"{evaluation.selection_metrics.target_event_average_precision:.4f}"
        )
    if drift is not None:
        print(f"Chronological subjects: {len(drift.subjects)}")
        print(f"Subject-independent fallbacks: {drift.fallback_subject_count}")
        print(f"Mean adapted AUROC delta: {drift.mean_auroc_delta:+.4f}")
        print(f"Mean adapted Brier delta: {drift.mean_brier_delta:+.4f}")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = peak_rss / 1024**3 if sys.platform == "darwin" else peak_rss / 1024**2
    print(f"Elapsed: {(time.monotonic() - started) / 60:.1f}m")
    print(f"Peak process RSS: {peak_gib:.2f} GiB")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
