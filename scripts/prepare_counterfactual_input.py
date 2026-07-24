"""Pair verified development language candidates with recorded P300 flash trials."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from neuroselect.decoding import read_decoder_artifacts
from neuroselect.evaluation import (
    load_counterfactual_spec,
    read_held_out_language_artifacts,
)
from neuroselect.evaluation.counterfactual_preparation import (
    CounterfactualInputBuilder,
    load_counterfactual_preparation_spec,
    write_counterfactual_input_artifacts,
)
from neuroselect.provenance import RunManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language-artifacts",
        type=Path,
        default=Path("artifacts/evaluation/held-out-language-personalization-dev-v1"),
    )
    parser.add_argument(
        "--decoder-artifacts",
        type=Path,
        default=Path("artifacts/models/p300-xdawn-lda-v1"),
    )
    parser.add_argument(
        "--preparation-config",
        type=Path,
        default=Path("configs/experiments/counterfactual_input_development.yaml"),
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=Path("configs/experiments/counterfactual_fusion_development.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/counterfactual-input-development-v1"),
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


def artifact_sha256(manifest: RunManifest, uri: str) -> str:
    matches = [item.sha256 for item in manifest.outputs if item.uri == uri]
    if len(matches) != 1:
        raise ValueError(f"source manifest must contain exactly one {uri} output")
    return matches[0]


def main() -> None:
    args = parse_args()
    preparation_spec = load_counterfactual_preparation_spec(args.preparation_config)
    fusion_spec = load_counterfactual_spec(args.fusion_config)
    language_result, language_manifest = read_held_out_language_artifacts(args.language_artifacts)
    _, _, decoder_evaluation, decoder_manifest = read_decoder_artifacts(args.decoder_artifacts)
    experiment_input = CounterfactualInputBuilder(
        preparation_spec,
        fusion_spec,
    ).build(
        language_result=language_result,
        decoder_evaluation=decoder_evaluation,
        source_decoder_manifest_sha256=decoder_manifest.digest(),
        original_task_evaluation_sha256=artifact_sha256(
            decoder_manifest, "artifact://evaluation.json"
        ),
        source_language_manifest_sha256=language_manifest.digest(),
        source_language_result_sha256=artifact_sha256(language_manifest, "artifact://result.json"),
        prepared_at=datetime.now(UTC),
    )
    revision, source_tree_sha256 = git_state()
    manifest = write_counterfactual_input_artifacts(
        experiment_input,
        preparation_spec,
        args.output,
        git_sha=revision,
        source_tree_sha256=source_tree_sha256,
        overwrite=args.overwrite,
    )
    selected_messages = {
        (trial.synthetic_profile_id, trial.message_id) for trial in experiment_input.trials
    }
    available = sum(trial.target_available for trial in experiment_input.trials)
    print(f"Run: {manifest.run_id}")
    print(f"Complete messages: {len(selected_messages)}")
    print(f"Paired trials: {len(experiment_input.trials)}")
    print(f"Intended-target availability: {available / len(experiment_input.trials):.3f}")
    print(f"Claim eligible: {experiment_input.source_evidence_claim_eligible}")
    print(f"Input: {args.output / 'input.json'}")
    print(f"Manifest SHA-256: {manifest.digest()}")


if __name__ == "__main__":
    main()
