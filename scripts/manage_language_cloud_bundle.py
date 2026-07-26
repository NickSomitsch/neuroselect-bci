"""Create, verify, or extract the portable research inputs for Step 11."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.evaluation.language_cloud import (
    create_language_cloud_bundle,
    extract_language_cloud_bundle,
    verify_language_cloud_bundle,
)
from neuroselect.language import load_local_model_config

DEFAULT_PROFILES = (
    "synthetic-casual",
    "synthetic-concise",
    "synthetic-formal",
    "synthetic-reflective",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cloud/step11-language-inputs-v1.tar.gz"),
    )
    create.add_argument("--repository-root", type=Path, default=Path("."))
    create.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    create.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("artifacts/models/language-lora"),
    )
    create.add_argument("--adapter-suffix", default="-research-v1")
    create.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("artifacts/language/personalization-v1"),
    )
    create.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_mlx.yaml"),
    )
    create.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("bundle", type=Path)

    extract = subparsers.add_parser("extract")
    extract.add_argument("bundle", type=Path)
    extract.add_argument("--destination", type=Path, default=Path("."))
    extract.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        manifest = create_language_cloud_bundle(
            args.output,
            repository_root=args.repository_root,
            profile_ids=tuple(args.profiles),
            adapter_root=args.adapter_root,
            adapter_suffix=args.adapter_suffix,
            corpus_root=args.corpus_root,
            model_config=load_local_model_config(args.model_config),
            overwrite=args.overwrite,
        )
        path = args.output
    elif args.command == "verify":
        manifest = verify_language_cloud_bundle(args.bundle)
        path = args.bundle
    else:
        manifest = extract_language_cloud_bundle(
            args.bundle,
            args.destination,
            overwrite=args.overwrite,
        )
        path = args.destination
    print(f"Bundle revision: {manifest.bundle_revision}")
    print(f"Profiles: {', '.join(manifest.profile_ids)}")
    print(f"Files: {len(manifest.files)}")
    print(f"Manifest SHA-256: {manifest.digest()}")
    print(f"Path: {path}")


if __name__ == "__main__":
    main()
