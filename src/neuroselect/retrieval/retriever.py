"""Deterministic lexical retrieval with permission and safety filtering."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from neuroselect.core.models import Candidate, CandidateKind, RecordPermission
from neuroselect.retrieval.models import (
    CandidateRetrievalEvidence,
    RetrievalHit,
    RetrievalPolicy,
    RetrievalRequest,
    StoredKnowledgeRecord,
)
from neuroselect.retrieval.store import SQLiteKnowledgeStore

DEFAULT_RETRIEVAL_POLICY = Path("configs/retrieval/lexical.yaml")
TOKEN_PATTERN = re.compile("[^\\W_]+(?:['\u2019][^\\W_]+)?", re.UNICODE)
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "for",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "the",
        "this",
        "to",
        "you",
    }
)


def load_retrieval_policy(path: str | Path = DEFAULT_RETRIEVAL_POLICY) -> RetrievalPolicy:
    """Load the tracked lexical and safety policy."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("retrieval policy must contain a YAML mapping")
    return RetrievalPolicy.model_validate(payload)


def _tokens(value: str) -> frozenset[str]:
    tokens = (
        match.group(0).casefold().replace("\u2019", "'") for match in TOKEN_PATTERN.finditer(value)
    )
    return frozenset(token for token in tokens if token not in STOP_WORDS)


def _score(
    query_tokens: frozenset[str], record_tokens: frozenset[str]
) -> tuple[float, tuple[str, ...]]:
    matched_terms = tuple(sorted(query_tokens.intersection(record_tokens)))
    if not matched_terms:
        return 0.0, ()
    query_coverage = len(matched_terms) / len(query_tokens)
    record_focus = len(matched_terms) / len(record_tokens)
    score = min(1.0, 0.75 * query_coverage + 0.25 * math.sqrt(record_focus))
    return score, matched_terms


class LexicalRetriever:
    """Retrieve only active, permissioned, non-quarantined profile records."""

    def __init__(
        self,
        store: SQLiteKnowledgeStore,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or load_retrieval_policy()

    def retrieve(self, request: RetrievalRequest) -> tuple[RetrievalHit, ...]:
        top_k = request.top_k or self.policy.default_top_k
        if top_k > self.policy.maximum_top_k:
            raise ValueError(
                f"requested top_k {top_k} exceeds policy maximum {self.policy.maximum_top_k}"
            )
        query_tokens = _tokens(request.query)
        if not query_tokens:
            return ()

        scored: list[RetrievalHit] = []
        for record in self.store.list_records(profile_id=request.profile_id):
            if not self._eligible(record, request):
                continue
            score, matched_terms = _score(query_tokens, _tokens(record.content))
            if score < self.policy.minimum_score or not matched_terms:
                continue
            scored.append(
                RetrievalHit(
                    record=record,
                    score=score,
                    matched_terms=matched_terms,
                    explanation=(
                        f"Matched {', '.join(matched_terms)} in a {record.kind.value} record "
                        f"from {record.source}."
                    ),
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.record.record_id))
        return tuple(scored[:top_k])

    @staticmethod
    def _eligible(record: StoredKnowledgeRecord, request: RetrievalRequest) -> bool:
        if record.injection_risk or not record.is_active_at(request.at_time):
            return False
        if request.permission not in record.permissions:
            return False
        return request.kinds is None or record.kind in request.kinds

    def retrieve_for_candidates(
        self,
        *,
        profile_id: str,
        confirmed_text: str,
        candidates: tuple[Candidate, ...],
        at_time: datetime,
        permission: RecordPermission = RecordPermission.SUGGEST,
        top_k: int | None = None,
    ) -> tuple[CandidateRetrievalEvidence, ...]:
        evidence: list[CandidateRetrievalEvidence] = []
        for candidate in candidates:
            if candidate.kind is CandidateKind.CONTROL:
                continue
            query = " ".join(part for part in (confirmed_text.strip(), candidate.text) if part)
            hits = self.retrieve(
                RetrievalRequest(
                    profile_id=profile_id,
                    query=query,
                    permission=permission,
                    at_time=at_time,
                    top_k=top_k,
                )
            )
            evidence.append(
                CandidateRetrievalEvidence(
                    candidate_id=candidate.candidate_id,
                    retrieval_support=hits[0].score if hits else 0.0,
                    record_ids=tuple(hit.record.record_id for hit in hits),
                    hits=hits,
                )
            )
        return tuple(evidence)
