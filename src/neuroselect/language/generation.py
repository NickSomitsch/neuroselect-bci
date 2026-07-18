"""Candidate-generation policy that keeps language output structured and non-authoritative."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from pydantic import ValidationError

from neuroselect.core.models import Candidate, CandidateKind, CandidateSet
from neuroselect.language.models import (
    BackendMetadata,
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateProposal,
    ControlPath,
    GenerationDiagnostics,
    ProposalRejectionReason,
    StructuredCandidateResponse,
)

CONTROL_CANDIDATES = (
    ("control-other", "Other…", ControlPath.OTHER),
    ("control-back", "Back", ControlPath.BACK),
    ("control-cancel", "Cancel", ControlPath.CANCEL),
)
RESERVED_CONTROL_KEYS = frozenset(path.value for path in ControlPath)


class CandidateGenerationError(RuntimeError):
    """Raised when a backend cannot provide a safe, complete visible set."""


class CandidateBackend(Protocol):
    """Minimal adapter interface for fixture and future local-model backends."""

    metadata: BackendMetadata

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]: ...


def parse_structured_proposals(payload: str) -> tuple[CandidateProposal, ...]:
    """Parse the strict JSON contract used by future constrained-model adapters."""

    try:
        decoded = json.loads(payload)
        response = StructuredCandidateResponse.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError) as error:
        raise CandidateGenerationError("backend returned invalid structured candidates") from error
    return response.candidates


def _normalize_visible_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", normalized)


def _canonical_key(value: str) -> str:
    characters = (character.casefold() if character.isalnum() else " " for character in value)
    return " ".join("".join(characters).split())


def _context_sha256(confirmed_text: str) -> str:
    canonical_context = _normalize_visible_text(confirmed_text)
    payload = json.dumps(
        {"confirmed_text": canonical_context}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CandidateGenerator:
    """Filter backend proposals and inject explicit, non-language control paths."""

    def __init__(self, backend: CandidateBackend) -> None:
        self.backend = backend

    def generate(self, request: CandidateGenerationRequest) -> CandidateGenerationResult:
        proposals = self.backend.generate(request)
        language_quota = request.candidate_count - len(CONTROL_CANDIDATES)
        rejected: Counter[ProposalRejectionReason] = Counter()
        valid: list[tuple[CandidateProposal, str, str]] = []
        seen_keys: set[str] = set()

        ordered_proposals = sorted(
            enumerate(proposals), key=lambda item: (-item[1].support, item[0])
        )
        for _, proposal in ordered_proposals:
            visible_text = _normalize_visible_text(proposal.text)
            canonical_key = _canonical_key(visible_text)
            rejection = self._rejection_reason(
                visible_text=visible_text,
                canonical_key=canonical_key,
                maximum_phrase_tokens=request.maximum_phrase_tokens,
                seen_keys=seen_keys,
            )
            if rejection is not None:
                rejected[rejection] += 1
                continue
            seen_keys.add(canonical_key)
            valid.append((proposal, visible_text, canonical_key))

        if len(valid) < language_quota:
            raise CandidateGenerationError(
                f"backend produced {len(valid)} safe unique language candidates; "
                f"{language_quota} required"
            )

        selected = valid[:language_quota]
        context_sha256 = _context_sha256(request.confirmed_text)
        language_candidates: list[Candidate] = []
        raw_support: dict[str, float] = {}
        for proposal, visible_text, canonical_key in selected:
            candidate_digest = hashlib.sha256(
                f"{context_sha256}\0{canonical_key}".encode()
            ).hexdigest()
            candidate_id = f"candidate-{candidate_digest[:20]}"
            language_candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    text=visible_text,
                    kind=(
                        CandidateKind.WORD
                        if len(visible_text.split()) == 1
                        else CandidateKind.PHRASE
                    ),
                    origins=frozenset(
                        {"generic-language", f"backend:{self.backend.metadata.backend_id}"}
                    ),
                )
            )
            raw_support[candidate_id] = proposal.support

        generic_language_support = self._normalize_support(raw_support)
        control_candidates = tuple(
            Candidate(
                candidate_id=candidate_id,
                text=text,
                kind=CandidateKind.CONTROL,
                origins=frozenset({"application-control"}),
            )
            for candidate_id, text, _ in CONTROL_CANDIDATES
        )
        all_candidates = (*language_candidates, *control_candidates)
        set_material = "\0".join(
            (
                context_sha256,
                self.backend.metadata.backend_id,
                self.backend.metadata.model_id,
                self.backend.metadata.model_revision,
                self.backend.metadata.generator_revision,
                self.backend.metadata.prompt_revision,
                str(request.candidate_count),
                str(request.maximum_phrase_tokens),
                *(candidate.candidate_id for candidate in all_candidates),
            )
        )
        set_digest = hashlib.sha256(set_material.encode()).hexdigest()
        candidate_set = CandidateSet(
            candidate_set_id=f"candidate-set-{set_digest[:20]}",
            context_sha256=context_sha256,
            candidates=all_candidates,
            generator_revision=self.backend.metadata.generator_revision,
            prompt_revision=self.backend.metadata.prompt_revision,
        )
        return CandidateGenerationResult(
            candidate_set=candidate_set,
            generic_language_support=generic_language_support,
            control_actions={
                candidate_id: action for candidate_id, _, action in CONTROL_CANDIDATES
            },
            backend=self.backend.metadata,
            diagnostics=GenerationDiagnostics(
                raw_proposal_count=len(proposals),
                selected_language_count=len(language_candidates),
                unused_valid_count=len(valid) - len(selected),
                rejected_by_reason=dict(rejected),
            ),
        )

    @staticmethod
    def _rejection_reason(
        *,
        visible_text: str,
        canonical_key: str,
        maximum_phrase_tokens: int,
        seen_keys: set[str],
    ) -> ProposalRejectionReason | None:
        if not visible_text or not canonical_key:
            return ProposalRejectionReason.EMPTY
        if any(unicodedata.category(character).startswith("C") for character in visible_text):
            return ProposalRejectionReason.UNSAFE_TEXT
        if canonical_key in RESERVED_CONTROL_KEYS:
            return ProposalRejectionReason.RESERVED_CONTROL
        if len(visible_text) > 160 or len(visible_text.split()) > maximum_phrase_tokens:
            return ProposalRejectionReason.TOO_LONG
        if canonical_key in seen_keys:
            return ProposalRejectionReason.DUPLICATE
        return None

    @staticmethod
    def _normalize_support(raw_support: dict[str, float]) -> dict[str, float]:
        total = sum(raw_support.values())
        if math.isclose(total, 0.0, abs_tol=1e-15):
            uniform = 1.0 / len(raw_support)
            return dict.fromkeys(raw_support, uniform)
        return {candidate_id: support / total for candidate_id, support in raw_support.items()}
