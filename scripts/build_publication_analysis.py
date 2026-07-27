"""Build and verify the frozen primary publication analysis."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from neuroselect.publication.analysis import (
    DEFAULT_PUBLICATION_ANALYSIS_CONFIG,
    build_publication_analysis,
    load_publication_analysis_spec,
    read_publication_analysis,
    write_publication_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PUBLICATION_ANALYSIS_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/publication/offline-methods-v1"),
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
    spec = load_publication_analysis_spec(args.config)
    result = build_publication_analysis(spec)
    revision, source_tree_sha256 = git_state()
    manifest = write_publication_analysis(
        result,
        spec,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    verified, verified_manifest = read_publication_analysis(args.output)
    assert verified == result and verified_manifest == manifest
    print(f"Run: {manifest.run_id}")
    print(f"Estimates: {len(result.estimates)}")
    print(f"Intervals: {len(result.intervals)}")
    print(f"EEGNet included: {result.eegnet_included}")
    print(f"Working tree clean: {manifest.metadata['working_tree_dirty'] is False}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
