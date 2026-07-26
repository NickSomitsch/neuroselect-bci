"""Resolve or explicitly download the exact revision-pinned local language model."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroselect.language import load_local_model_config, resolve_local_model_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_mlx.yaml"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow Hugging Face network access when the pinned snapshot is not cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_local_model_config(args.model_config)
    source = resolve_local_model_source(config, allow_download=args.download)
    print(f"Model: {config.model_id}")
    print(f"Revision: {config.model_revision}")
    print(f"Resolved path: {source}")


if __name__ == "__main__":
    main()
