"""Print deterministic structured candidates without changing session state."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from neuroselect.language import (
    CandidateGenerationRequest,
    CandidateGenerator,
    FixtureCandidateBackend,
    load_fixture_backend_config,
)

CandidateCount = Literal[4, 6, 8, 12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmed_text", nargs="?", default="")
    parser.add_argument("--count", type=int, choices=(4, 6, 8, 12), default=8)
    parser.add_argument("--maximum-phrase-tokens", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/language/fixture.yaml"),
        help="Path to the deterministic fixture backend configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = FixtureCandidateBackend(load_fixture_backend_config(args.config))
    result = CandidateGenerator(backend).generate(
        CandidateGenerationRequest(
            confirmed_text=args.confirmed_text,
            candidate_count=cast(CandidateCount, args.count),
            maximum_phrase_tokens=args.maximum_phrase_tokens,
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
