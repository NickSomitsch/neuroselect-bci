"""Run the loopback-only NeuroSelect research API."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from neuroselect.api import create_app
from neuroselect.core.config import load_app_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo/default.yaml"),
        help="Path to the loopback-only application configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)
    uvicorn.run(
        create_app(),
        host=config.service.host,
        port=config.service.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
