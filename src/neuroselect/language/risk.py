"""Trusted, versioned sensitive-content tagging for generated candidates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from neuroselect.language.models import CandidateRiskPolicy

DEFAULT_CANDIDATE_RISK_POLICY = Path("configs/language/risk.yaml")


def load_candidate_risk_policy(
    path: str | Path = DEFAULT_CANDIDATE_RISK_POLICY,
) -> CandidateRiskPolicy:
    """Load the application-owned policy; model output cannot alter these rules."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("candidate risk policy must contain a YAML mapping")
    return CandidateRiskPolicy.model_validate(payload)


class CandidateRiskTagger:
    """Apply conservative text patterns after candidate normalization."""

    def __init__(self, policy: CandidateRiskPolicy | None = None) -> None:
        self.policy = policy or load_candidate_risk_policy()
        self._patterns = tuple(
            (
                rule.risk_tag,
                tuple(re.compile(pattern, re.IGNORECASE) for pattern in rule.patterns),
            )
            for rule in self.policy.rules
        )

    def tag(self, text: str) -> frozenset[str]:
        return frozenset(
            risk_tag
            for risk_tag, patterns in self._patterns
            if any(pattern.search(text) for pattern in patterns)
        )
