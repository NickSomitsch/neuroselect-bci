"""Train and checksum one synthetic-profile MLX QLoRA adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

from neuroselect.language import (
    build_mlx_lora_command,
    finalize_personalization_adapter,
    load_local_model_config,
    load_lora_training_config,
    load_personalization_corpus_manifest,
    resolve_local_model_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_mlx.yaml"),
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_lora.yaml"),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly allow downloading the exact pinned model revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"adapter output already exists; choose a new directory: {args.output}")
    corpus_manifest = load_personalization_corpus_manifest(args.corpus)
    model_config = load_local_model_config(args.model_config)
    training_config = load_lora_training_config(args.training_config)
    model_source = resolve_local_model_source(model_config, allow_download=args.download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_mlx_lora_command(
        model_source=model_source,
        corpus_dir=Path(args.corpus),
        adapter_dir=args.output,
        config=training_config,
    )
    subprocess.run(command, check=True)
    manifest = finalize_personalization_adapter(
        adapter_dir=args.output,
        corpus_manifest=corpus_manifest,
        model_config=model_config,
        training_config=training_config,
        mlx_lm_version=importlib.metadata.version("mlx-lm"),
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
