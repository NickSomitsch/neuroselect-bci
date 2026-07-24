"""Run held-out natural-candidate evaluation with verified local MLX adapters."""

from __future__ import annotations

import argparse
import gc
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from neuroselect.evaluation import (
    HeldOutLanguageBenchmarkRunner,
    LanguageProfileRuntime,
    build_held_out_candidate_vocabulary,
    load_held_out_language_spec,
    write_held_out_language_artifacts,
)
from neuroselect.language import (
    CandidateGenerator,
    LocalModelCandidateBackend,
    MlxAdapterPersonalizer,
    PersonalizedLanguagePipeline,
    load_local_model_config,
    load_personalization_adapter,
    load_personalization_corpus_manifest,
)
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import generate_from_sources, load_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/held_out_language_personalization.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3_4b_mlx.yaml"),
    )
    parser.add_argument(
        "--benchmark-spec",
        type=Path,
        default=Path("synthetic_data/benchmark.yaml"),
    )
    parser.add_argument("--profiles", type=Path, default=Path("synthetic_data/profiles"))
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("artifacts/language/personalization-v1"),
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("artifacts/models/language-lora"),
    )
    parser.add_argument("--adapter-suffix", default="-dev-v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/held-out-language-personalization-dev-v1"),
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


def clear_mlx_cache() -> None:
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        mx.clear_cache()
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    spec = load_held_out_language_spec(args.config)
    model_config = load_local_model_config(args.model_config)
    profiles = load_profiles(args.profiles)
    benchmark = generate_from_sources(args.benchmark_spec, args.profiles)
    candidate_vocabulary = build_held_out_candidate_vocabulary(benchmark)
    base_backend = LocalModelCandidateBackend(
        model_config,
        candidate_vocabulary=candidate_vocabulary,
    )
    generator = CandidateGenerator(base_backend)

    with SQLiteKnowledgeStore(":memory:") as store:
        for profile in profiles:
            for record in profile.knowledge:
                store.add(
                    profile_id=profile.profile_id,
                    record=KnowledgeRecordInput.model_validate(record.model_dump()),
                    at_time=spec.retrieval_at,
                )
        retriever = LexicalRetriever(store)
        runtimes: list[LanguageProfileRuntime] = []
        for profile in profiles:
            adapter = load_personalization_adapter(
                args.adapter_root / f"{profile.profile_id}{args.adapter_suffix}",
                expected_profile_id=profile.profile_id,
                expected_model_id=model_config.model_id,
                expected_model_revision=model_config.model_revision,
            )
            corpus = load_personalization_corpus_manifest(args.corpus_root / profile.profile_id)

            def pipeline_factory(
                *,
                adapter: Any = adapter,
            ) -> PersonalizedLanguagePipeline:
                adapter_backend = LocalModelCandidateBackend(
                    model_config,
                    adapter_path=adapter.directory,
                )
                return PersonalizedLanguagePipeline(
                    generator,
                    MlxAdapterPersonalizer(adapter_backend, adapter),
                    retriever,
                )

            runtimes.append(
                LanguageProfileRuntime(
                    profile=profile,
                    adapter=adapter,
                    corpus_manifest=corpus,
                    pipeline_factory=pipeline_factory,
                    cleanup=clear_mlx_cache,
                )
            )
        result = HeldOutLanguageBenchmarkRunner(spec).run(
            benchmark=benchmark,
            runtimes=tuple(runtimes),
            candidate_vocabulary_sha256=candidate_vocabulary.digest(),
        )

    revision, source_tree_sha256 = git_state()
    manifest = write_held_out_language_artifacts(
        result,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    print(f"Run: {result.run_id}")
    print(f"Trials: {len(result.trials)}")
    print(f"Claim eligible: {result.claim_eligible}")
    for metrics in result.metrics:
        scope = metrics.profile_id or "overall"
        print(
            f"{scope}: availability={metrics.target_availability_rate:.3f} "
            f"repaired={metrics.repaired_generation_rate:.3f} "
            f"generic_top1={metrics.generic_top_1_candidate_recall:.3f} "
            f"personalized_top1={metrics.personalized_top_1_candidate_recall:.3f}"
        )
    print(f"Manifest: {args.output / 'manifest.json'}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
