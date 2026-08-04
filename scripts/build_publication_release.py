"""Build the deterministic NeuroSelect v0.1.0 archival release."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.publication.archival import (
    DEFAULT_RELEASE_CONFIG,
    build_publication_release,
    load_release_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release/v0.1.0"))
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    spec = load_release_spec(args.config)
    result = build_publication_release(
        spec,
        repository=Path.cwd(),
        output=args.output,
        allow_pending=args.allow_pending,
        overwrite=args.overwrite,
    )
    print(f"Release: {result.output}")
    print(f"Git revision: {result.inventory.git_revision}")
    print(f"Release ready: {result.inventory.release_ready}")
    for archive in result.inventory.archives:
        print(f"- {archive.filename}: {archive.sha256}")
    for gate in result.inventory.pending_gates:
        print(f"BLOCKED: {gate}")


if __name__ == "__main__":
    main()
