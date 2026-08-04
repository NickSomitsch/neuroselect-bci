"""Independently verify a generated NeuroSelect archival release."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.publication.archival import verify_publication_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release/v0.1.0"))
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    inventory = verify_publication_release(args.output, require_ready=not args.allow_pending)
    print("Publication release verified")
    print(f"Release ready: {inventory.release_ready}")
    print(f"Git revision: {inventory.git_revision}")
    for archive in inventory.archives:
        print(f"- {archive.filename}: {archive.sha256}")


if __name__ == "__main__":
    main()
