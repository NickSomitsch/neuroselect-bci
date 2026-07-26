"""Durable, identity-locked checkpoints for long held-out language runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.evaluation.language_benchmark import LanguageBenchmarkTrial
from neuroselect.language import BackendMetadata


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


class LanguageCheckpointIdentity(BaseModel):
    """Every input that must agree before completed trials may be reused."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    run_id: str = Field(min_length=1, max_length=160)
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: BackendMetadata
    adapter_manifest_sha256: dict[str, str]
    corpus_manifest_sha256: dict[str, str]
    expected_trial_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_profiles(self) -> LanguageCheckpointIdentity:
        if set(self.adapter_manifest_sha256) != set(self.corpus_manifest_sha256):
            raise ValueError("checkpoint adapter and corpus profiles must agree")
        for values in (self.adapter_manifest_sha256, self.corpus_manifest_sha256):
            if any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in values.values()
            ):
                raise ValueError("checkpoint manifest digests must be lowercase SHA-256 values")
        return self

    def digest(self) -> str:
        return _sha256_text(_canonical_json(self.model_dump(mode="json")))


class LanguageCheckpointMetadata(BaseModel):
    """Persistent checkpoint header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    identity: LanguageCheckpointIdentity
    started_at: datetime

    @model_validator(mode="after")
    def require_aware_time(self) -> LanguageCheckpointMetadata:
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("checkpoint started_at must include a timezone")
        return self


class LanguageCheckpointCompletion(BaseModel):
    """Marker linking a complete checkpoint to its canonical result manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_count: int = Field(ge=1)
    result_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime


class LanguageCheckpointStore:
    """Append-only JSONL storage with bounded-loss fsync and safe resume."""

    def __init__(
        self,
        directory: Path,
        metadata: LanguageCheckpointMetadata,
        trials: tuple[LanguageBenchmarkTrial, ...],
        *,
        flush_every: int,
    ) -> None:
        self.directory = directory
        self.metadata = metadata
        self.trials = list(trials)
        self._trial_ids = {trial.trial_id for trial in trials}
        self._flush_every = flush_every
        self._pending = 0
        self._stream: IO[str] = (directory / "trials.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )

    @classmethod
    def open(
        cls,
        directory: str | Path,
        identity: LanguageCheckpointIdentity,
        *,
        resume: bool = False,
        flush_every: int = 5,
        started_at: datetime | None = None,
    ) -> Self:
        """Create a checkpoint or resume only when its complete identity agrees."""

        if flush_every < 1:
            raise ValueError("checkpoint flush frequency must be positive")
        destination = Path(directory)
        metadata_path = destination / "checkpoint.json"
        trials_path = destination / "trials.jsonl"
        if metadata_path.exists():
            if not resume:
                raise FileExistsError(
                    f"checkpoint already exists at {destination}; pass --resume to reuse it"
                )
            metadata = LanguageCheckpointMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
            if metadata.identity != identity:
                raise ValueError(
                    "checkpoint identity does not match the code, protocol, model, adapters, "
                    "corpora, or candidate vocabulary"
                )
        else:
            if trials_path.exists() or (destination.exists() and any(destination.iterdir())):
                raise ValueError(
                    f"checkpoint directory {destination} is non-empty but has no checkpoint header"
                )
            destination.mkdir(parents=True, exist_ok=True)
            metadata = LanguageCheckpointMetadata(
                schema_version="1.0",
                identity=identity,
                started_at=started_at or datetime.now(UTC),
            )
            _atomic_write(
                metadata_path,
                (_canonical_json(metadata.model_dump(mode="json")) + "\n").encode(),
            )

        trials = cls._read_trials(trials_path)
        if len(trials) > identity.expected_trial_count:
            raise ValueError("checkpoint contains more trials than the selected protocol")
        return cls(destination, metadata, trials, flush_every=flush_every)

    @staticmethod
    def _read_trials(path: Path) -> tuple[LanguageBenchmarkTrial, ...]:
        if not path.exists():
            return ()
        content = path.read_bytes()
        complete_length = content.rfind(b"\n") + 1
        if complete_length != len(content):
            _atomic_write(path, content[:complete_length])
            content = content[:complete_length]
        trials: list[LanguageBenchmarkTrial] = []
        trial_ids: set[str] = set()
        for line_number, line in enumerate(content.splitlines(), start=1):
            try:
                trial = LanguageBenchmarkTrial.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"invalid checkpoint trial on line {line_number} of {path}"
                ) from error
            if trial.trial_id in trial_ids:
                raise ValueError(f"duplicate checkpoint trial ID: {trial.trial_id}")
            trial_ids.add(trial.trial_id)
            trials.append(trial)
        return tuple(trials)

    def append(self, trial: LanguageBenchmarkTrial) -> None:
        """Append one completed trial and periodically force it to durable storage."""

        if trial.trial_id in self._trial_ids:
            raise ValueError(f"checkpoint already contains trial {trial.trial_id}")
        if len(self.trials) >= self.metadata.identity.expected_trial_count:
            raise ValueError("checkpoint already contains the expected number of trials")
        self._stream.write(_canonical_json(trial.model_dump(mode="json")) + "\n")
        self.trials.append(trial)
        self._trial_ids.add(trial.trial_id)
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush and fsync all pending trials."""

        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._pending = 0

    def mark_complete(self, *, result_manifest_sha256: str) -> None:
        """Persist the link to the canonical final artifact after all trials exist."""

        self.flush()
        if len(self.trials) != self.metadata.identity.expected_trial_count:
            raise ValueError("cannot complete a checkpoint before every expected trial exists")
        completion = LanguageCheckpointCompletion(
            schema_version="1.0",
            identity_sha256=self.metadata.identity.digest(),
            trial_count=len(self.trials),
            result_manifest_sha256=result_manifest_sha256,
            completed_at=datetime.now(UTC),
        )
        _atomic_write(
            self.directory / "complete.json",
            (_canonical_json(completion.model_dump(mode="json")) + "\n").encode(),
        )

    def close(self) -> None:
        if not self._stream.closed:
            self.flush()
            self._stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
