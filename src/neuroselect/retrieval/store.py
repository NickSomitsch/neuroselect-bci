"""SQLite persistence for profile-scoped personal knowledge records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any, Self

from pydantic import TypeAdapter

from neuroselect.core.models import Identifier
from neuroselect.retrieval.models import (
    KnowledgeRecordInput,
    KnowledgeRecordPatch,
    StoredKnowledgeRecord,
    require_timezone,
)
from neuroselect.retrieval.safety import detect_prompt_injection

SCHEMA_VERSION = 1
IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class KnowledgeStoreError(RuntimeError):
    """Base class for expected personal-knowledge storage failures."""


class KnowledgeRecordNotFoundError(KnowledgeStoreError):
    pass


class KnowledgeRecordConflictError(KnowledgeStoreError):
    pass


class KnowledgeStoreSchemaError(KnowledgeStoreError):
    pass


class SQLiteKnowledgeStore(AbstractContextManager["SQLiteKnowledgeStore"]):
    """Small local store with parameterized SQL and optimistic revisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, SCHEMA_VERSION}:
            self._connection.close()
            raise KnowledgeStoreSchemaError(
                f"unsupported personal-knowledge schema version {version}"
            )
        if version == 0:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE knowledge_records (
                        profile_id TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        permissions_json TEXT NOT NULL,
                        enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                        valid_from TEXT,
                        valid_until TEXT,
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        injection_risk INTEGER NOT NULL CHECK (injection_risk IN (0, 1)),
                        risk_reasons_json TEXT NOT NULL,
                        PRIMARY KEY (profile_id, record_id)
                    )
                    """
                )
                self._connection.execute(
                    "CREATE INDEX knowledge_profile_enabled "
                    "ON knowledge_records (profile_id, enabled)"
                )
                self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def add(
        self,
        *,
        profile_id: str,
        record: KnowledgeRecordInput,
        at_time: datetime | None = None,
    ) -> StoredKnowledgeRecord:
        with self._lock:
            profile_id = IDENTIFIER_ADAPTER.validate_python(profile_id)
            timestamp = at_time or datetime.now(UTC)
            require_timezone(timestamp, "at_time")
            risk_reasons = detect_prompt_injection(record.content)
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO knowledge_records (
                            profile_id, record_id, kind, content, source, permissions_json,
                            enabled, valid_from, valid_until, revision, created_at, updated_at,
                            injection_risk, risk_reasons_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            profile_id,
                            record.record_id,
                            record.kind.value,
                            record.content,
                            record.source,
                            self._permissions_json(record),
                            int(record.enabled),
                            self._datetime_text(record.valid_from),
                            self._datetime_text(record.valid_until),
                            timestamp.isoformat(),
                            timestamp.isoformat(),
                            int(bool(risk_reasons)),
                            json.dumps([reason.value for reason in risk_reasons]),
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise KnowledgeRecordConflictError(
                    f"knowledge record already exists: {profile_id}/{record.record_id}"
                ) from error
            return self.get(profile_id=profile_id, record_id=record.record_id)

    def get(self, *, profile_id: str, record_id: str) -> StoredKnowledgeRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM knowledge_records WHERE profile_id = ? AND record_id = ?",
                (profile_id, record_id),
            ).fetchone()
            if row is None:
                raise KnowledgeRecordNotFoundError(
                    f"knowledge record not found: {profile_id}/{record_id}"
                )
            return self._record_from_row(row)

    def list_records(
        self, *, profile_id: str, include_disabled: bool = False
    ) -> tuple[StoredKnowledgeRecord, ...]:
        with self._lock:
            query = "SELECT * FROM knowledge_records WHERE profile_id = ?"
            parameters: tuple[Any, ...] = (profile_id,)
            if not include_disabled:
                query += " AND enabled = 1"
            query += " ORDER BY record_id"
            rows = self._connection.execute(query, parameters).fetchall()
            return tuple(self._record_from_row(row) for row in rows)

    def update(
        self,
        *,
        profile_id: str,
        record_id: str,
        expected_revision: int,
        patch: KnowledgeRecordPatch,
        at_time: datetime | None = None,
    ) -> StoredKnowledgeRecord:
        with self._lock:
            timestamp = at_time or datetime.now(UTC)
            require_timezone(timestamp, "at_time")
            current = self.get(profile_id=profile_id, record_id=record_id)
            if timestamp < current.updated_at:
                raise ValueError("update time cannot precede the current record revision")
            input_fields = KnowledgeRecordInput.model_fields
            merged_payload = {
                field_name: getattr(current, field_name) for field_name in input_fields
            }
            merged_payload.update(patch.model_dump(exclude_unset=True))
            merged = KnowledgeRecordInput.model_validate(merged_payload)
            risk_reasons = detect_prompt_injection(merged.content)
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE knowledge_records SET
                        kind = ?, content = ?, source = ?, permissions_json = ?, enabled = ?,
                        valid_from = ?, valid_until = ?, revision = revision + 1, updated_at = ?,
                        injection_risk = ?, risk_reasons_json = ?
                    WHERE profile_id = ? AND record_id = ? AND revision = ?
                    """,
                    (
                        merged.kind.value,
                        merged.content,
                        merged.source,
                        self._permissions_json(merged),
                        int(merged.enabled),
                        self._datetime_text(merged.valid_from),
                        self._datetime_text(merged.valid_until),
                        timestamp.isoformat(),
                        int(bool(risk_reasons)),
                        json.dumps([reason.value for reason in risk_reasons]),
                        profile_id,
                        record_id,
                        expected_revision,
                    ),
                )
            if cursor.rowcount != 1:
                raise KnowledgeRecordConflictError(
                    f"knowledge record revision conflict: {profile_id}/{record_id}"
                )
            return self.get(profile_id=profile_id, record_id=record_id)

    def disable(
        self,
        *,
        profile_id: str,
        record_id: str,
        expected_revision: int,
        at_time: datetime | None = None,
    ) -> StoredKnowledgeRecord:
        return self.update(
            profile_id=profile_id,
            record_id=record_id,
            expected_revision=expected_revision,
            patch=KnowledgeRecordPatch(enabled=False),
            at_time=at_time,
        )

    def delete(self, *, profile_id: str, record_id: str, expected_revision: int) -> None:
        with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM knowledge_records "
                    "WHERE profile_id = ? AND record_id = ? AND revision = ?",
                    (profile_id, record_id, expected_revision),
                )
            if cursor.rowcount == 1:
                return
            try:
                self.get(profile_id=profile_id, record_id=record_id)
            except KnowledgeRecordNotFoundError:
                raise
            raise KnowledgeRecordConflictError(
                f"knowledge record revision conflict: {profile_id}/{record_id}"
            )

    @staticmethod
    def _permissions_json(record: KnowledgeRecordInput) -> str:
        return json.dumps(sorted(permission.value for permission in record.permissions))

    @staticmethod
    def _datetime_text(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    @staticmethod
    def _optional_datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value is not None else None

    @classmethod
    def _record_from_row(cls, row: sqlite3.Row) -> StoredKnowledgeRecord:
        return StoredKnowledgeRecord.model_validate(
            {
                "profile_id": row["profile_id"],
                "record_id": row["record_id"],
                "kind": row["kind"],
                "content": row["content"],
                "source": row["source"],
                "permissions": json.loads(row["permissions_json"]),
                "enabled": bool(row["enabled"]),
                "valid_from": cls._optional_datetime(row["valid_from"]),
                "valid_until": cls._optional_datetime(row["valid_until"]),
                "revision": row["revision"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
                "injection_risk": bool(row["injection_risk"]),
                "risk_reasons": json.loads(row["risk_reasons_json"]),
            }
        )
