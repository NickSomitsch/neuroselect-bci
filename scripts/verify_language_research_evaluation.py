"""Strictly verify a complete, clean, protocol-locked Step 11 research result."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from neuroselect.evaluation import (
    build_held_out_candidate_vocabulary,
    expected_language_trial_count,
    held_out_language_run_id,
    load_held_out_language_spec,
    read_held_out_language_artifacts,
    select_held_out_messages,
)
from neuroselect.language import (
    load_local_model_config,
    load_personalization_adapter,
    load_personalization_corpus_manifest,
)
from neuroselect.synthetic import generate_from_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/evaluation/held-out-language-personalization-research-v1"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/held_out_language_personalization_research.yaml"),
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
    parser.add_argument("--adapter-suffix", default="-research-v1")
    parser.add_argument(
        "--allow-different-git",
        action="store_true",
        help="Allow verification from a checkout other than the producing commit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, manifest = read_held_out_language_artifacts(args.artifacts)
    spec = load_held_out_language_spec(args.config)
    model = load_local_model_config(args.model_config)
    benchmark = generate_from_sources(args.benchmark_spec, args.profiles)
    vocabulary = build_held_out_candidate_vocabulary(benchmark)
    if result.spec != spec or spec.evidence_tier != "research":
        raise ValueError("result does not use the locked research language protocol")
    if result.backend != model.metadata:
        raise ValueError("result language backend does not match the pinned model configuration")
    if result.benchmark_source_sha256 != benchmark.source_sha256:
        raise ValueError("result benchmark checksum does not match the generated benchmark")
    if result.candidate_vocabulary_sha256 != vocabulary.digest():
        raise ValueError("result candidate vocabulary does not match non-test benchmark data")
    if len(result.trials) != expected_language_trial_count(spec, benchmark):
        raise ValueError("result does not contain every selected research trial")

    expected_trials: list[tuple[object, ...]] = []
    for profile_id in sorted(benchmark.profile_ids):
        adapter = load_personalization_adapter(
            args.adapter_root / f"{profile_id}{args.adapter_suffix}",
            expected_profile_id=profile_id,
            expected_model_id=model.model_id,
            expected_model_revision=model.model_revision,
        )
        corpus = load_personalization_corpus_manifest(args.corpus_root / profile_id)
        if result.adapters.get(profile_id) != adapter.manifest:
            raise ValueError(f"result adapter provenance does not match {profile_id}")
        if result.corpus_manifest_sha256.get(profile_id) != corpus.digest():
            raise ValueError(f"result corpus provenance does not match {profile_id}")
        messages = tuple(
            message
            for message in benchmark.messages[spec.split]
            if message.profile_id == profile_id
        )
        for message in select_held_out_messages(spec, messages):
            confirmed: list[str] = []
            for span_index, intended_text in enumerate(message.target_spans):
                expected_trials.append(
                    (
                        f"language-{message.message_id}-{span_index:02d}",
                        profile_id,
                        message.message_id,
                        span_index,
                        len(message.target_spans),
                        " ".join(confirmed),
                        intended_text,
                        adapter.manifest.adapter_id,
                        adapter.manifest.adapter_sha256,
                    )
                )
                confirmed.append(intended_text)

    actual_trials = [
        (
            trial.trial_id,
            trial.profile_id,
            trial.message_id,
            trial.span_index,
            trial.message_span_count,
            trial.confirmed_context,
            trial.intended_text,
            trial.adapter_id,
            trial.adapter_sha256,
        )
        for trial in result.trials
    ]
    if actual_trials != expected_trials:
        raise ValueError("result trial order or teacher-forced protocol fields do not match")
    expected_run_id = held_out_language_run_id(
        spec=spec,
        benchmark_source_sha256=benchmark.source_sha256,
        backend=model.metadata,
        adapters=result.adapters,
        candidate_vocabulary_sha256=vocabulary.digest(),
    )
    if result.run_id != expected_run_id or manifest.run_id != expected_run_id:
        raise ValueError("result run identity does not match its research inputs")
    if not result.claim_eligible:
        raise ValueError("result is not eligible for held-out personalization claims")
    if manifest.metadata.get("working_tree_dirty") is not False:
        raise ValueError("research result was produced from a dirty working tree")
    current_git = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if not args.allow_different_git and manifest.git_sha != current_git:
        raise ValueError("research result was produced by a different Git revision")

    print("Step 11 research evaluation verified")
    print(f"Run: {result.run_id}")
    print(f"Trials: {len(result.trials)}")
    print(f"Profiles: {', '.join(sorted(result.adapters))}")
    print(f"Claim eligible: {result.claim_eligible}")
    print(f"Producing Git SHA: {manifest.git_sha}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
