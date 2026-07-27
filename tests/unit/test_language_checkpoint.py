from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.evaluation import (
    LanguageBenchmarkTrial,
    LanguageCheckpointIdentity,
    LanguageCheckpointStore,
)
from neuroselect.language import BackendMetadata

NOW = datetime(2026, 7, 26, 12, tzinfo=UTC)


def identity() -> LanguageCheckpointIdentity:
    return LanguageCheckpointIdentity(
        schema_version="1.0",
        run_id="held-out-language-test",
        git_sha="a" * 40,
        config_sha256="b" * 64,
        model_config_sha256="c" * 64,
        benchmark_source_sha256="d" * 64,
        candidate_vocabulary_sha256="e" * 64,
        backend=BackendMetadata(
            backend_id="test-language",
            model_id="test/model",
            model_revision="f" * 40,
            generator_revision="test-generator-v1",
            prompt_revision="test-prompt-v1",
            deterministic=True,
        ),
        adapter_manifest_sha256={"synthetic-concise": "1" * 64},
        corpus_manifest_sha256={"synthetic-concise": "2" * 64},
        expected_trial_count=2,
    )


def failed_trial(index: int) -> LanguageBenchmarkTrial:
    return LanguageBenchmarkTrial(
        trial_id=f"language-msg-{'1' * 20}-{index:02d}",
        profile_id="synthetic-concise",
        message_id=f"msg-{'1' * 20}",
        span_index=index,
        message_span_count=2,
        confirmed_context="" if index == 0 else "alpha",
        intended_text="alpha" if index == 0 else "beta",
        candidate_generation_failed=True,
        failure_reason="controlled failure",
        adapter_id="lora-synthetic-concise-test",
        adapter_sha256="3" * 64,
    )


def test_checkpoint_round_trip_repairs_partial_tail_and_marks_complete(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    with LanguageCheckpointStore.open(
        checkpoint_dir,
        identity(),
        flush_every=1,
        started_at=NOW,
    ) as store:
        store.append(failed_trial(0))
        with pytest.raises(ValueError, match="already contains"):
            store.append(failed_trial(0))
        with pytest.raises(ValueError, match="before every expected"):
            store.mark_complete(result_manifest_sha256="4" * 64)

    trials_path = checkpoint_dir / "trials.jsonl"
    with trials_path.open("ab") as stream:
        stream.write(b'{"interrupted":')

    with LanguageCheckpointStore.open(
        checkpoint_dir,
        identity(),
        resume=True,
        flush_every=1,
    ) as resumed:
        assert resumed.metadata.started_at == NOW
        assert resumed.trials == [failed_trial(0)]
        resumed.append(failed_trial(1))
        resumed.mark_complete(result_manifest_sha256="4" * 64)

    assert trials_path.read_bytes().endswith(b"\n")
    assert (checkpoint_dir / "complete.json").is_file()


def test_checkpoint_refuses_unapproved_or_mismatched_resume(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    LanguageCheckpointStore.open(checkpoint_dir, identity()).close()

    with pytest.raises(FileExistsError, match="pass --resume"):
        LanguageCheckpointStore.open(checkpoint_dir, identity())
    changed = identity().model_copy(update={"git_sha": "9" * 40})
    with pytest.raises(ValueError, match="identity does not match"):
        LanguageCheckpointStore.open(checkpoint_dir, changed, resume=True)


def test_checkpoint_rejects_invalid_storage_and_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frequency must be positive"):
        LanguageCheckpointStore.open(tmp_path / "frequency", identity(), flush_every=0)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "unrelated").write_text("test", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        LanguageCheckpointStore.open(nonempty, identity())

    bad_profiles = identity().model_dump()
    bad_profiles["corpus_manifest_sha256"] = {"synthetic-formal": "2" * 64}
    with pytest.raises(ValidationError, match="profiles must agree"):
        LanguageCheckpointIdentity.model_validate(bad_profiles)
    bad_digest = identity().model_dump()
    bad_digest["corpus_manifest_sha256"] = {"synthetic-concise": "G" * 64}
    with pytest.raises(ValidationError, match="SHA-256"):
        LanguageCheckpointIdentity.model_validate(bad_digest)


def test_checkpoint_rejects_invalid_duplicate_and_excess_trials(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    LanguageCheckpointStore.open(invalid, identity()).close()
    (invalid / "trials.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid checkpoint trial"):
        LanguageCheckpointStore.open(invalid, identity(), resume=True)

    duplicate = tmp_path / "duplicate"
    LanguageCheckpointStore.open(duplicate, identity()).close()
    line = failed_trial(0).model_dump_json() + "\n"
    (duplicate / "trials.jsonl").write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate checkpoint trial"):
        LanguageCheckpointStore.open(duplicate, identity(), resume=True)

    full = tmp_path / "full"
    with LanguageCheckpointStore.open(full, identity()) as store:
        store.append(failed_trial(0))
        store.append(failed_trial(1))
        with pytest.raises(ValueError, match="expected number"):
            store.append(failed_trial(0).model_copy(update={"trial_id": "language-extra"}))


def test_checkpoint_uses_fast_active_storage_with_atomic_durable_mirror(
    tmp_path: Path,
) -> None:
    active = tmp_path / "active"
    mirror = tmp_path / "mirror"
    with LanguageCheckpointStore.open(
        active,
        identity(),
        flush_every=2,
        mirror_directory=mirror,
    ) as store:
        store.append(failed_trial(0))
        assert (mirror / "trials.jsonl").read_text(encoding="utf-8") == ""
        store.append(failed_trial(1))

    mirrored_trials = (mirror / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(mirrored_trials) == 2

    restored_active = tmp_path / "restored-active"
    with LanguageCheckpointStore.open(
        restored_active,
        identity(),
        resume=True,
        mirror_directory=mirror,
    ) as restored:
        assert restored.trials == [failed_trial(0), failed_trial(1)]
        restored.mark_complete(result_manifest_sha256="4" * 64)

    assert (mirror / "complete.json").is_file()
