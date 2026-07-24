"""Run candidate-aligned generic, style-personalized, and synthetic RAG scoring."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from neuroselect.language import (
    CandidateGenerationRequest,
    CandidateGenerator,
    ControlledStylePersonalizer,
    FixtureCandidateBackend,
    LocalModelCandidateBackend,
    MlxAdapterPersonalizer,
    PersonalizedLanguagePipeline,
    load_local_model_config,
    load_personalization_adapter,
)
from neuroselect.language.generation import CandidateBackend
from neuroselect.language.personalization import CandidatePersonalizer
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import load_profiles

CandidateCount = Literal[4, 6, 8, 12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmed_text", nargs="?", default="I would like")
    parser.add_argument("--profile", default="synthetic-concise")
    parser.add_argument("--count", type=int, choices=(4, 6, 8, 12), default=8)
    parser.add_argument("--maximum-phrase-tokens", type=int, default=4)
    parser.add_argument("--backend", choices=("fixture", "mlx"), default="fixture")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_mlx.yaml"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Verified MLX adapter directory. Omit to use the controlled style proxy.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("synthetic_data/profiles"),
    )
    parser.add_argument(
        "--at-time",
        type=datetime.fromisoformat,
        default=datetime.now(UTC),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly allow downloading the exact pinned model revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = next(
        (item for item in load_profiles(args.profiles) if item.profile_id == args.profile),
        None,
    )
    if profile is None:
        raise ValueError(f"unknown synthetic profile: {args.profile}")
    if args.adapter is not None and args.backend != "mlx":
        raise ValueError("a real adapter requires the MLX backend")

    base_backend: CandidateBackend
    personalizer: CandidatePersonalizer
    if args.backend == "mlx":
        model_config = load_local_model_config(args.model_config)
        base_backend = LocalModelCandidateBackend(model_config, allow_download=args.download)
        if args.adapter is not None:
            bundle = load_personalization_adapter(
                args.adapter,
                expected_profile_id=profile.profile_id,
                expected_model_id=model_config.model_id,
                expected_model_revision=model_config.model_revision,
            )
            adapter_backend = LocalModelCandidateBackend(
                model_config,
                adapter_path=bundle.directory,
                allow_download=args.download,
            )
            personalizer = MlxAdapterPersonalizer(adapter_backend, bundle)
        else:
            personalizer = ControlledStylePersonalizer(profile)
    else:
        base_backend = FixtureCandidateBackend()
        personalizer = ControlledStylePersonalizer(profile)

    at_time = args.at_time
    if at_time.tzinfo is None or at_time.utcoffset() is None:
        raise ValueError("--at-time must include a timezone")
    with SQLiteKnowledgeStore(":memory:") as store:
        for record in profile.knowledge:
            store.add(
                profile_id=profile.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=at_time,
            )
        result = PersonalizedLanguagePipeline(
            CandidateGenerator(base_backend),
            personalizer,
            LexicalRetriever(store),
        ).generate(
            CandidateGenerationRequest(
                confirmed_text=args.confirmed_text,
                candidate_count=cast(CandidateCount, args.count),
                maximum_phrase_tokens=args.maximum_phrase_tokens,
            ),
            profile_id=profile.profile_id,
            at_time=at_time,
        )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
