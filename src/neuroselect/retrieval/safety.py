"""Conservative prompt-injection detection for untrusted personal records."""

from __future__ import annotations

import re

from neuroselect.retrieval.models import InjectionRisk

_INJECTION_PATTERNS: tuple[tuple[InjectionRisk, re.Pattern[str]], ...] = (
    (
        InjectionRisk.INSTRUCTION_OVERRIDE,
        re.compile(
            r"\b(?:ignore|disregard|override)\b.{0,40}\b"
            r"(?:previous|prior|above|system|developer)\b.{0,20}\binstructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        InjectionRisk.ROLE_MARKER,
        re.compile(
            r"(?:<\|\s*(?:system|assistant|developer|user)\s*\|>|"
            r"\b(?:system|assistant|developer)\s*(?:message|prompt)\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        InjectionRisk.PROMPT_EXFILTRATION,
        re.compile(
            r"\b(?:reveal|show|print|repeat|expose)\b.{0,50}\b"
            r"(?:system prompt|hidden prompt|developer message|instructions?)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_prompt_injection(content: str) -> tuple[InjectionRisk, ...]:
    """Return stable risk labels; flagged records are quarantined from retrieval."""

    return tuple(risk for risk, pattern in _INJECTION_PATTERNS if pattern.search(content))
