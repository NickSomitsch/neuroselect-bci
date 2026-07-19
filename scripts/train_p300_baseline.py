"""Train and evaluate the classical P300 baseline from prepared Study P epochs."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from neuroselect.decoding import (
    evaluate_decoder,
    fit_calibrated_decoder,
    load_classical_decoder_config,
    load_partitioned_epoch_batches,
    write_decoder_artifacts,
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
        default=Path("configs/decoding/xdawn_lda.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/p300-xdawn-lda-v1"),
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
    config = load_classical_decoder_config(args.config)
    batches = load_partitioned_epoch_batches(args.processed_root)
    decoder, summary = fit_calibrated_decoder(
        batches[DataSplit.TRAIN], batches[DataSplit.VALIDATION], config
    )
    evaluation = evaluate_decoder(decoder, batches[DataSplit.TEST])
    revision, source_tree_sha256 = git_state()
    manifest = write_decoder_artifacts(
        decoder,
        summary,
        evaluation,
        args.output,
        git_sha=revision,
        run_time=datetime.now(UTC),
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    assert evaluation.metrics is not None
    print(f"Run: {manifest.run_id}")
    print(f"Labeled test epochs: {evaluation.labeled_epoch_count}")
    print(f"Unknown replay-only epochs: {evaluation.unknown_epoch_count}")
    print(f"AUROC: {evaluation.metrics.auroc:.4f}")
    print(f"Balanced accuracy: {evaluation.metrics.balanced_accuracy:.4f}")
    print(f"Brier score: {evaluation.metrics.brier_score:.4f}")
    print(f"ECE: {evaluation.metrics.expected_calibration_error:.4f}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
