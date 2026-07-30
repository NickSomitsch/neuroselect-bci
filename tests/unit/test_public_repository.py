from __future__ import annotations

import json
from pathlib import Path

from neuroselect.reporting.public_repository import (
    MAX_TRACKED_FILE_BYTES,
    audit_public_commit_emails,
    audit_public_files,
    audit_public_history_blobs,
    audit_public_history_patch,
    audit_public_history_paths,
)


def test_public_file_audit_accepts_safe_source_and_clean_notebook(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "outputs": [],
                "source": ["print('safe')"],
            }
        ]
    }
    (tmp_path / "notebooks/example.ipynb").write_text(
        json.dumps(notebook),
        encoding="utf-8",
    )

    assert (
        audit_public_files(
            tmp_path,
            ("src/example.py", "notebooks/example.ipynb"),
        )
        == ()
    )


def test_public_file_audit_rejects_restricted_private_and_large_files(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/token.txt").write_text(
        "home=/" + "Users/example/private\n"
        "email=person" + "@gmail.com\n"
        "token=github_pat_" + "a" * 24,
        encoding="utf-8",
    )
    (tmp_path / "weights.safetensors").write_bytes(b"\0weights")
    (tmp_path / "large.txt").write_bytes(b"x" * (MAX_TRACKED_FILE_BYTES + 1))

    errors = audit_public_files(
        tmp_path,
        ("artifacts/token.txt", "weights.safetensors", "large.txt", "missing.txt"),
    )

    assert "restricted artifact/data path is tracked: artifacts/token.txt" in errors
    assert "GitHub token pattern found in artifacts/token.txt" in errors
    assert "absolute user-home path found in artifacts/token.txt" in errors
    assert "consumer email address found in artifacts/token.txt" in errors
    assert "restricted artifact/data file type is tracked: weights.safetensors" in errors
    assert any(error.startswith("tracked file exceeds") for error in errors)
    assert "tracked path is missing or not a regular file: missing.txt" in errors


def test_public_file_audit_rejects_notebook_state_and_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "executed.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "outputs": [{"output_type": "stream", "text": ["private"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "invalid.ipynb").write_text("{", encoding="utf-8")
    (tmp_path / "invalid-cells.ipynb").write_text("{}", encoding="utf-8")

    errors = audit_public_files(
        tmp_path,
        ("executed.ipynb", "invalid.ipynb", "invalid-cells.ipynb"),
    )

    assert "executed notebook cell is tracked: executed.ipynb cell 0" in errors
    assert "notebook output is tracked: executed.ipynb cell 0" in errors
    assert "invalid notebook JSON: invalid.ipynb" in errors
    assert "invalid notebook cell structure: invalid-cells.ipynb" in errors


def test_public_history_audit_rejects_restricted_paths_secrets_and_consumer_email() -> None:
    assert audit_public_history_paths(("src/example.py", "results/reference/.gitkeep")) == ()
    assert audit_public_history_paths(("data/private.edf",)) == (
        "Git history contains restricted artifact/data path is tracked: data/private.edf",
        "Git history contains restricted artifact/data file type is tracked: data/private.edf",
    )
    assert audit_public_history_patch("safe patch") == ()
    assert audit_public_history_patch("token=sk-" + "x" * 24) == (
        "OpenAI-style token pattern found in Git history",
    )
    assert audit_public_history_blobs((("src/example.py", 12),)) == ()
    assert audit_public_history_blobs((("old/large.bin", MAX_TRACKED_FILE_BYTES + 1),)) == (
        "Git history contains a file exceeding "
        f"{MAX_TRACKED_FILE_BYTES} bytes: old/large.bin ({MAX_TRACKED_FILE_BYTES + 1})",
    )
    assert audit_public_commit_emails(("71410422+user@users.noreply.github.com",)) == ()
    assert audit_public_commit_emails(("person" + "@hotmail.com",)) == (
        "consumer email address found in Git author/committer metadata",
    )
