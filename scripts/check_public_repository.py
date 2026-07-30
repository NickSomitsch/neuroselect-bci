"""Audit public-candidate files and complete Git history before changing visibility."""

from __future__ import annotations

import subprocess
from pathlib import Path

from neuroselect.reporting.public_repository import (
    audit_public_commit_emails,
    audit_public_files,
    audit_public_history_blobs,
    audit_public_history_patch,
    audit_public_history_paths,
)

ROOT = Path(__file__).parents[1]


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _historical_blobs() -> tuple[tuple[str, int], ...]:
    objects = [
        line.partition(" ")
        for line in _git_bytes("rev-list", "--objects", "--all").decode().splitlines()
    ]
    object_ids = [object_id for object_id, _, _ in objects]
    metadata = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=ROOT,
        input=("\n".join(object_ids) + "\n").encode(),
        check=True,
        capture_output=True,
    ).stdout
    object_sizes = {}
    for line in metadata.decode().splitlines():
        object_id, object_type, size = line.split()
        if object_type == "blob":
            object_sizes[object_id] = int(size)
    return tuple(
        (path, object_sizes[object_id])
        for object_id, separator, path in objects
        if separator and object_id in object_sizes
    )


def main() -> None:
    public_candidate_paths = tuple(
        item.decode()
        for item in _git_bytes(
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ).split(b"\0")
        if item
    )
    history_paths = _git_bytes(
        "log",
        "--all",
        "--pretty=format:",
        "--name-only",
    ).decode()
    history_patch = _git_bytes(
        "log",
        "--all",
        "-p",
        "--no-ext-diff",
        "--no-textconv",
    ).decode(errors="replace")
    commit_emails = _git_bytes(
        "log",
        "--all",
        "--format=%ae%n%ce",
    ).decode()
    errors = [
        *audit_public_files(ROOT, public_candidate_paths),
        *audit_public_history_paths(history_paths.splitlines()),
        *audit_public_history_patch(history_patch),
        *audit_public_history_blobs(_historical_blobs()),
        *audit_public_commit_emails(commit_emails.splitlines()),
    ]
    if errors:
        raise SystemExit("Public repository audit failed:\n- " + "\n- ".join(sorted(set(errors))))
    print(
        "Public repository audit passed: "
        f"{len(public_candidate_paths)} public candidate files and complete Git history checked."
    )


if __name__ == "__main__":
    main()
