"""Deterministic subject/session splits with explicit leakage checks."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence

from neuroselect.eeg.models import DataSplit, EpochMetadata, SessionFold, SubjectSplit


def make_subject_split(
    subject_ids: Iterable[str],
    *,
    seed: int = 20260718,
    validation_count: int = 3,
    test_count: int = 3,
) -> SubjectSplit:
    """Create the primary held-subject split without reading labels or source folder names."""

    supplied_subjects = tuple(subject_ids)
    subjects = sorted(set(supplied_subjects))
    if len(subjects) != len(supplied_subjects):
        raise ValueError("subject IDs must be unique")
    if validation_count < 1 or test_count < 1:
        raise ValueError("validation and test partitions must be non-empty")
    train_count = len(subjects) - validation_count - test_count
    if train_count < 1:
        raise ValueError("subject split must leave at least one training subject")
    random.Random(seed).shuffle(subjects)
    return SubjectSplit(
        seed=seed,
        train_subjects=tuple(sorted(subjects[:train_count])),
        validation_subjects=tuple(sorted(subjects[train_count : train_count + validation_count])),
        test_subjects=tuple(sorted(subjects[train_count + validation_count :])),
    )


def split_for_subject(subject_id: str, split: SubjectSplit) -> DataSplit:
    memberships = {
        DataSplit.TRAIN: split.train_subjects,
        DataSplit.VALIDATION: split.validation_subjects,
        DataSplit.TEST: split.test_subjects,
    }
    for partition, subject_ids in memberships.items():
        if subject_id in subject_ids:
            return partition
    raise ValueError(f"subject {subject_id} is absent from the split manifest")


def assign_epochs_by_subject(
    epochs: Iterable[EpochMetadata], split: SubjectSplit
) -> dict[DataSplit, tuple[EpochMetadata, ...]]:
    assigned: dict[DataSplit, list[EpochMetadata]] = {partition: [] for partition in DataSplit}
    for epoch in epochs:
        assigned[split_for_subject(epoch.subject_id, split)].append(epoch)
    result = {partition: tuple(items) for partition, items in assigned.items()}
    validate_split_integrity(result, require_subject_disjoint=True)
    return result


def validate_split_integrity(
    partitions: Mapping[DataSplit, Sequence[EpochMetadata]],
    *,
    require_subject_disjoint: bool,
) -> None:
    """Reject duplicate epochs and any subject/session/trial group crossing a boundary."""

    if set(partitions) != set(DataSplit):
        raise ValueError("epoch partitions must define train, validation, and test")
    group_sets: dict[str, dict[DataSplit, set[str]]] = {
        "epoch": {},
        "selection trial": {},
        "recording": {},
        "subject": {},
    }
    for partition, epochs in partitions.items():
        group_sets["epoch"][partition] = {epoch.epoch_id for epoch in epochs}
        if len(group_sets["epoch"][partition]) != len(epochs):
            raise ValueError(f"duplicate epoch IDs inside {partition.value}")
        group_sets["selection trial"][partition] = {epoch.selection_trial_id for epoch in epochs}
        group_sets["recording"][partition] = {epoch.recording_id for epoch in epochs}
        group_sets["subject"][partition] = {epoch.subject_id for epoch in epochs}

    checked_groups: tuple[str, ...] = ("epoch", "selection trial", "recording")
    if require_subject_disjoint:
        checked_groups += ("subject",)
    for group_name in checked_groups:
        by_partition = group_sets[group_name]
        ordered = tuple(DataSplit)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                overlap = by_partition[left] & by_partition[right]
                if overlap:
                    raise ValueError(
                        f"{group_name} leakage between {left.value} and {right.value}: "
                        f"{sorted(overlap)[:3]}"
                    )


def cross_session_folds() -> tuple[SessionFold, SessionFold]:
    """Return both directions; Study P condition order is counterbalanced by subject."""

    return (
        SessionFold(
            fold_id="study-p-se001-to-se002",
            train_sessions=("SE001",),
            test_sessions=("SE002",),
        ),
        SessionFold(
            fold_id="study-p-se002-to-se001",
            train_sessions=("SE002",),
            test_sessions=("SE001",),
        ),
    )
