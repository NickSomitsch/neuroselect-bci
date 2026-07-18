"""Versioned deterministic candidate backend for CPU-only development and tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from neuroselect.language.models import (
    BackendMetadata,
    CandidateGenerationRequest,
    CandidateProposal,
    FixtureBackendConfig,
)

DEFAULT_FIXTURE_CONFIG = Path("configs/language/fixture.yaml")


def load_fixture_backend_config(
    path: str | Path = DEFAULT_FIXTURE_CONFIG,
) -> FixtureBackendConfig:
    """Load a strict fixture backend definition from tracked YAML."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("fixture backend configuration must contain a YAML mapping")
    return FixtureBackendConfig.model_validate(payload)


def _normalize_context(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


class FixtureCandidateBackend:
    """Return stable suffix-conditioned proposals without loading a language model."""

    def __init__(self, config: FixtureBackendConfig | None = None) -> None:
        self.config = config or load_fixture_backend_config()
        self.metadata: BackendMetadata = self.config.metadata

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]:
        context = _normalize_context(request.confirmed_text)
        proposals: list[CandidateProposal] = []
        for rule in self.config.rules:
            if context.endswith(_normalize_context(rule.suffix)):
                proposals.extend(rule.candidates)
        proposals.extend(self.config.default_candidates)
        return tuple(proposals)
