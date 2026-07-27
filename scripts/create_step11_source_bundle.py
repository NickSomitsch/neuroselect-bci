"""Create the exact clean Git source bundle consumed by the Step 11 cloud notebook."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cloud/neuroselect-step11-source.bundle"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("commit every intended source change before creating the cloud bundle")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite source bundle: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        temporary_path = Path(temporary) / args.output.name
        subprocess.run(
            ["git", "bundle", "create", str(temporary_path), "HEAD"],
            check=True,
        )
        heads = subprocess.run(
            ["git", "bundle", "list-heads", str(temporary_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if not any(line.startswith(revision + " ") for line in heads):
            raise ValueError("created source bundle does not contain the current commit")
        temporary_path.replace(args.output)

    print(f"Source revision: {revision}")
    print(f"Source bundle: {args.output}")


if __name__ == "__main__":
    main()
