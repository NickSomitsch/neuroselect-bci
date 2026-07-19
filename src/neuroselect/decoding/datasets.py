"""Load prepared epoch artifacts according to the tracked subject split."""

from __future__ import annotations

from pathlib import Path

from neuroselect.eeg import DataSplit, EpochBatch, SubjectSplit, read_epoch_batch, split_for_subject


def load_partitioned_epoch_batches(root: str | Path) -> dict[DataSplit, list[EpochBatch]]:
    source = Path(root)
    split_path = source / "subject-split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"missing split manifest: {split_path}")
    split = SubjectSplit.model_validate_json(split_path.read_text(encoding="utf-8"))
    directories = sorted(path.parent for path in source.rglob("epoch-checksums.json"))
    if not directories:
        raise FileNotFoundError(f"no prepared epoch artifacts found under {source}")
    partitions: dict[DataSplit, list[EpochBatch]] = {partition: [] for partition in DataSplit}
    for directory in directories:
        batch = read_epoch_batch(directory)
        subject_ids = {item.subject_id for item in batch.metadata}
        if len(subject_ids) != 1:
            raise ValueError(f"epoch artifact mixes subjects: {directory}")
        partition = split_for_subject(next(iter(subject_ids)), split)
        partitions[partition].append(batch)
    if any(not batches for batches in partitions.values()):
        raise ValueError("prepared epochs must cover train, validation, and test subjects")
    return partitions
