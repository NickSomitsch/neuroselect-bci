"""Fail-closed checks for files and history intended for a public repository."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

MAX_TRACKED_FILE_BYTES = 1024 * 1024
ALLOWED_TRACKED_PATHS = frozenset({"results/reference/.gitkeep"})
FORBIDDEN_TRACKED_PREFIXES = ("artifacts/", "data/", "models/", "results/")
FORBIDDEN_TRACKED_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".db",
        ".edf",
        ".env",
        ".fif",
        ".joblib",
        ".key",
        ".npy",
        ".npz",
        ".onnx",
        ".p12",
        ".pem",
        ".pfx",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".sqlite",
        ".sqlite3",
    }
)
_CREDENTIAL_PATTERNS = (
    ("private key", re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY")),
    ("GitHub token", re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")),
    ("OpenAI-style token", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{30,}")),
    ("Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
)
_LOCAL_PATH_PATTERN = re.compile(r"(?:/" + r"Users/[^/\s]+/|/" + r"home/[^/\s]+/)")
_CONSUMER_EMAIL_PATTERN = re.compile(
    r"[A-Z0-9._%+-]{1,64}@(?:gmail|outlook|hotmail|icloud)\.[A-Z]{2,}",
    re.IGNORECASE,
)


def _normalized_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().removeprefix("./")


def _path_errors(path: str) -> tuple[str, ...]:
    if path in ALLOWED_TRACKED_PATHS:
        return ()
    errors: list[str] = []
    if path.startswith(FORBIDDEN_TRACKED_PREFIXES):
        errors.append(f"restricted artifact/data path is tracked: {path}")
    if PurePosixPath(path).suffix.casefold() in FORBIDDEN_TRACKED_SUFFIXES:
        errors.append(f"restricted artifact/data file type is tracked: {path}")
    return tuple(errors)


def _text_errors(text: str, *, source: str) -> tuple[str, ...]:
    errors: list[str] = []
    for label, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            errors.append(f"{label} pattern found in {source}")
    if _LOCAL_PATH_PATTERN.search(text):
        errors.append(f"absolute user-home path found in {source}")
    if _CONSUMER_EMAIL_PATTERN.search(text):
        errors.append(f"consumer email address found in {source}")
    return tuple(errors)


def _notebook_errors(path: str, text: str) -> tuple[str, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return (f"invalid notebook JSON: {path}",)
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list):
        return (f"invalid notebook cell structure: {path}",)
    errors: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"executed notebook cell is tracked: {path} cell {index}")
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list) or outputs:
            errors.append(f"notebook output is tracked: {path} cell {index}")
    return tuple(errors)


def audit_public_files(root: Path, tracked_paths: Iterable[str]) -> tuple[str, ...]:
    """Audit the exact set of files intended to be tracked publicly."""

    errors: list[str] = []
    for raw_path in sorted(set(tracked_paths)):
        relative = _normalized_path(raw_path)
        errors.extend(_path_errors(relative))
        path = root / relative
        if not path.is_file():
            errors.append(f"tracked path is missing or not a regular file: {relative}")
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            errors.append(
                f"tracked file exceeds {MAX_TRACKED_FILE_BYTES} bytes: {relative} ({size})"
            )
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        errors.extend(_text_errors(text, source=relative))
        if path.suffix.casefold() == ".ipynb":
            errors.extend(_notebook_errors(relative, text))
    return tuple(errors)


def audit_public_history_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Reject restricted data/model paths even when they occur only in Git history."""

    errors: list[str] = []
    for raw_path in sorted(set(paths)):
        if raw_path.strip():
            errors.extend(_path_errors(_normalized_path(raw_path)))
    return tuple(f"Git history contains {error}" for error in errors)


def audit_public_history_patch(patch: str) -> tuple[str, ...]:
    """Scan historical patches without echoing any matched secret value."""

    return _text_errors(patch, source="Git history")


def audit_public_history_blobs(blobs: Iterable[tuple[str, int]]) -> tuple[str, ...]:
    """Reject oversized file payloads retained anywhere in Git history."""

    errors = []
    for path, size in blobs:
        if size > MAX_TRACKED_FILE_BYTES:
            errors.append(
                "Git history contains a file exceeding "
                f"{MAX_TRACKED_FILE_BYTES} bytes: {_normalized_path(path)} ({size})"
            )
    return tuple(errors)


def audit_public_commit_emails(emails: Iterable[str]) -> tuple[str, ...]:
    """Reject consumer email addresses embedded in public commit metadata."""

    errors = []
    for email in sorted({item.strip() for item in emails if item.strip()}):
        if _CONSUMER_EMAIL_PATTERN.fullmatch(email):
            errors.append("consumer email address found in Git author/committer metadata")
    return tuple(errors)
