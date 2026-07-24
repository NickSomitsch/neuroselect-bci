"""Download and preprocess pinned bigP3BCI Study P records outside Git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroselect.eeg import (
    EXPECTED_SUBJECT_IDS,
    SourcePartition,
    cross_session_folds,
    download_pinned_inventory,
    download_source_files,
    load_pinned_inventory,
    load_study_p_edf,
    make_subject_split,
    preprocess_recording,
    recording_artifact_directory,
    select_source_files,
    write_epoch_batch,
    write_standardized_recording,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/bigp3bci/1.0.0"),
        help="Ignored immutable source root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/bigp3bci-study-p/1.0.0"),
        help="Ignored standardized FIF and epoch artifact root.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=EXPECTED_SUBJECT_IDS,
        help="Study P subjects to process. Required with --download.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=("SE001", "SE002"),
        help="Optional session subset; defaults to both sessions.",
    )
    parser.add_argument(
        "--source-partitions",
        nargs="+",
        choices=tuple(item.value for item in SourcePartition),
        help="Optional author block subset (train/test); never used as a model split.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly fetch the pinned inventory and selected EDF files.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge the dataset's CC-BY-4.0 terms for downloads.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        help="Development smoke limit after deterministic inventory selection.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=1,
        help="Bounded parallel EDF downloads; checksum verification remains per file.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_split_manifests(output_root: Path) -> None:
    split = make_subject_split(EXPECTED_SUBJECT_IDS)
    (output_root / "subject-split.json").write_text(
        json.dumps(split.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "session-folds.json").write_text(
        json.dumps(
            [fold.model_dump(mode="json") for fold in cross_session_folds()],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.download and not args.subjects:
        raise SystemExit("--download requires an explicit --subjects selection")
    if args.limit_files is not None and args.limit_files < 1:
        raise SystemExit("--limit-files must be positive")
    if not 1 <= args.download_workers <= 16:
        raise SystemExit("--download-workers must lie in [1, 16]")

    inventory_path = args.raw_root / "SHA256SUMS.txt"
    if args.download:
        download_pinned_inventory(inventory_path, accept_license=args.accept_license)
    elif not inventory_path.exists():
        raise SystemExit(
            f"missing {inventory_path}; rerun with --download --accept-license --subjects P_XX"
        )

    inventory = load_pinned_inventory(inventory_path)
    selected = select_source_files(
        inventory,
        subject_ids=args.subjects,
        session_ids=args.sessions,
        source_partitions=(
            tuple(SourcePartition(value) for value in args.source_partitions)
            if args.source_partitions
            else None
        ),
    )
    if args.limit_files is not None:
        selected = selected[: args.limit_files]
    if args.download:
        download_source_files(
            selected,
            args.raw_root,
            accept_license=args.accept_license,
            workers=args.download_workers,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_split_manifests(args.output_root)
    for index, source in enumerate(selected, start=1):
        recording = load_study_p_edf(source, args.raw_root)
        artifact_directory = recording_artifact_directory(args.output_root, recording.metadata)
        write_standardized_recording(recording, args.output_root, overwrite=args.overwrite)
        batch = preprocess_recording(recording)
        write_epoch_batch(batch, artifact_directory, overwrite=args.overwrite)
        print(
            f"[{index}/{len(selected)}] {recording.metadata.key.recording_id}: "
            f"{batch.report.accepted_event_count}/{batch.report.input_event_count} epochs accepted"
        )

    print(f"Prepared {len(selected)} source recordings under {args.output_root}")


if __name__ == "__main__":
    main()
