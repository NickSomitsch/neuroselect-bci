"""Deterministic transparent fusion with explicit uncertainty decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neuroselect.core.models import Candidate, CandidateKind, EvidenceBreakdown
from neuroselect.ranking.models import (
    ConfirmationLevel,
    DominanceFlag,
    RankedCandidate,
    RankingDisposition,
    RankingInputs,
    RankingPolicy,
    RankingReason,
    RankingResult,
    RiskLevel,
)
from neuroselect.retrieval.models import CandidateRetrievalEvidence

DEFAULT_RANKING_POLICY = Path("configs/ranking/default.yaml")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def load_ranking_policy(path: str | Path = DEFAULT_RANKING_POLICY) -> RankingPolicy:
    """Load the versioned, hand-set safety baseline."""

    with Path(path).open(encoding="utf-8") as config_file:
        payload: Any = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("ranking policy must contain a YAML mapping")
    return RankingPolicy.model_validate(payload)


@dataclass(frozen=True)
class _CandidateSignals:
    candidate: Candidate
    original_index: int
    neural: float | None
    generic_language: float
    personalization_lift: float
    retrieval: CandidateRetrievalEvidence | None
    risk_level: RiskLevel
    base_contributions: dict[str, float]
    dominance_flags: frozenset[str]

    @property
    def base_score(self) -> float:
        return sum(self.base_contributions.values())


class TransparentRanker:
    """Rank visible candidates without selecting, confirming, or mutating them."""

    def __init__(self, policy: RankingPolicy | None = None) -> None:
        self.policy = policy or load_ranking_policy()

    def rank(self, inputs: RankingInputs) -> RankingResult:
        retrieval_by_candidate = {item.candidate_id: item for item in inputs.retrieval_evidence}
        neural_probabilities = inputs.neural_evidence.candidate_probabilities
        neural_top_id, neural_top_support, neural_margin = self._neural_summary(
            candidates=inputs.candidate_set.candidates,
            probabilities=neural_probabilities,
        )
        lm_top_id = self._language_top_id(inputs)
        signals = tuple(
            self._signals_for_candidate(
                candidate=candidate,
                original_index=index,
                inputs=inputs,
                retrieval=retrieval_by_candidate.get(candidate.candidate_id),
                neural_top_id=neural_top_id,
                lm_top_id=lm_top_id,
            )
            for index, candidate in enumerate(inputs.candidate_set.candidates)
        )
        ranked = self._rank_with_diversity(signals)
        fused_top = ranked[0]
        fused_margin = (
            fused_top.breakdown.total_score - ranked[1].breakdown.total_score
            if len(ranked) > 1
            else max(0.0, fused_top.breakdown.total_score)
        )
        reasons = self._reason_codes(
            ranked=ranked,
            neural_top_id=neural_top_id,
            neural_top_support=neural_top_support,
            neural_margin=neural_margin,
            fused_margin=fused_margin,
        )
        disposition = self._disposition(reasons)
        confirmation_level = (
            ConfirmationLevel.ENHANCED
            if fused_top.risk_level is not RiskLevel.NONE
            or RankingReason.NEURAL_LANGUAGE_CONFLICT in reasons
            else ConfirmationLevel.STANDARD
        )
        return RankingResult(
            candidate_set_id=inputs.candidate_set.candidate_set_id,
            policy_revision=self.policy.policy_revision,
            neural_evidence_id=inputs.neural_evidence.evidence_id,
            disposition=disposition,
            confirmation_level=confirmation_level,
            reason_codes=tuple(reasons),
            ranked_candidates=ranked,
            fused_top_candidate_id=fused_top.candidate.candidate_id,
            neural_top_candidate_id=neural_top_id,
            display_top_candidate_id=(
                fused_top.candidate.candidate_id
                if disposition is RankingDisposition.DISPLAY
                else None
            ),
            fused_margin=fused_margin,
            neural_margin=neural_margin,
        )

    def _signals_for_candidate(
        self,
        *,
        candidate: Candidate,
        original_index: int,
        inputs: RankingInputs,
        retrieval: CandidateRetrievalEvidence | None,
        neural_top_id: str | None,
        lm_top_id: str,
    ) -> _CandidateSignals:
        weights = self.policy.weights
        neural = inputs.neural_evidence.candidate_probabilities.get(candidate.candidate_id)
        generic_language = inputs.generic_language_support.get(candidate.candidate_id, 0.0)
        personalization_lift = inputs.personalization_lift.get(candidate.candidate_id, 0.0)
        retrieval_support = retrieval.retrieval_support if retrieval is not None else 0.0
        risk_level = self._risk_level(candidate)
        base_contributions = {
            "neural": weights.neural * (neural or 0.0),
            "generic_language": weights.generic_language * generic_language,
            "personalization": weights.personalization * personalization_lift,
            "retrieval": weights.retrieval * retrieval_support,
            "risk": -weights.risk_penalty * risk_level.numeric,
        }
        flags: set[str] = set()
        if neural is None:
            flags.add(DominanceFlag.MISSING_NEURAL.value)
        elif candidate.candidate_id == lm_top_id and candidate.candidate_id != neural_top_id:
            non_neural = sum(
                base_contributions[name]
                for name in ("generic_language", "personalization", "retrieval")
            )
            if non_neural > (base_contributions["neural"] * self.policy.lm_dominance_ratio):
                flags.add(DominanceFlag.LM_OVER_NEURAL.value)
        return _CandidateSignals(
            candidate=candidate,
            original_index=original_index,
            neural=neural,
            generic_language=generic_language,
            personalization_lift=personalization_lift,
            retrieval=retrieval,
            risk_level=risk_level,
            base_contributions=base_contributions,
            dominance_flags=frozenset(flags),
        )

    def _rank_with_diversity(
        self, signals: tuple[_CandidateSignals, ...]
    ) -> tuple[RankedCandidate, ...]:
        remaining = list(signals)
        selected: list[RankedCandidate] = []
        selected_language: list[Candidate] = []
        while remaining:
            scored = [
                (signal, self._diversity_adjustment(signal.candidate, selected_language))
                for signal in remaining
            ]
            signal, diversity_adjustment = min(
                scored,
                key=lambda item: (
                    -(item[0].base_score + item[1]),
                    item[0].original_index,
                ),
            )
            contributions = {
                **signal.base_contributions,
                "diversity": diversity_adjustment,
            }
            breakdown = EvidenceBreakdown(
                candidate_id=signal.candidate.candidate_id,
                neural=signal.neural,
                generic_language=signal.generic_language,
                personal_lift=signal.personalization_lift,
                retrieval=(
                    signal.retrieval.retrieval_support if signal.retrieval is not None else 0.0
                ),
                diversity_adjustment=diversity_adjustment,
                risk=signal.risk_level.numeric,
                weighted_contributions=contributions,
                total_score=sum(contributions.values()),
                dominance_flags=signal.dominance_flags,
            )
            selected.append(
                RankedCandidate(
                    rank=len(selected) + 1,
                    candidate=signal.candidate,
                    breakdown=breakdown,
                    risk_level=signal.risk_level,
                    confirmation_level=(
                        ConfirmationLevel.ENHANCED
                        if signal.risk_level is not RiskLevel.NONE
                        else ConfirmationLevel.STANDARD
                    ),
                    retrieval_hits=signal.retrieval.hits if signal.retrieval is not None else (),
                )
            )
            if signal.candidate.kind is not CandidateKind.CONTROL:
                selected_language.append(signal.candidate)
            remaining.remove(signal)
        return tuple(selected)

    def _diversity_adjustment(
        self, candidate: Candidate, selected_language: list[Candidate]
    ) -> float:
        if candidate.kind is CandidateKind.CONTROL or not selected_language:
            return 0.0
        candidate_tokens = self._tokens(candidate.text)
        maximum_similarity = max(
            self._jaccard(candidate_tokens, self._tokens(selected.text))
            for selected in selected_language
        )
        return -self.policy.weights.maximum_diversity_penalty * maximum_similarity

    def _risk_level(self, candidate: Candidate) -> RiskLevel:
        if candidate.risk_tags.intersection(self.policy.high_risk_tags):
            return RiskLevel.HIGH
        if candidate.risk_tags.intersection(self.policy.elevated_risk_tags):
            return RiskLevel.ELEVATED
        return RiskLevel.NONE

    def _reason_codes(
        self,
        *,
        ranked: tuple[RankedCandidate, ...],
        neural_top_id: str | None,
        neural_top_support: float | None,
        neural_margin: float | None,
        fused_margin: float,
    ) -> list[RankingReason]:
        reasons: list[RankingReason] = []
        if neural_top_id is None or neural_top_support is None or neural_margin is None:
            reasons.append(RankingReason.MISSING_NEURAL_EVIDENCE)
        else:
            if neural_top_support < self.policy.minimum_neural_top_support:
                reasons.append(RankingReason.LOW_NEURAL_SUPPORT)
            if neural_margin < self.policy.minimum_neural_margin:
                reasons.append(RankingReason.LOW_NEURAL_MARGIN)
            if ranked[0].candidate.candidate_id != neural_top_id:
                reasons.append(RankingReason.NEURAL_LANGUAGE_CONFLICT)
        if any(
            DominanceFlag.LM_OVER_NEURAL.value in item.breakdown.dominance_flags for item in ranked
        ):
            reasons.append(RankingReason.LM_DOMINANCE_DETECTED)
        if ranked[0].breakdown.total_score < self.policy.minimum_fused_top_score:
            reasons.append(RankingReason.LOW_FUSED_SCORE)
        if fused_margin < self.policy.minimum_fused_margin:
            reasons.append(RankingReason.LOW_FUSED_MARGIN)
        if ranked[0].risk_level is not RiskLevel.NONE:
            reasons.append(RankingReason.SENSITIVE_CANDIDATE)
        return reasons

    @staticmethod
    def _disposition(reasons: list[RankingReason]) -> RankingDisposition:
        repeat_reasons = {
            RankingReason.LOW_NEURAL_SUPPORT,
            RankingReason.LOW_NEURAL_MARGIN,
            RankingReason.NEURAL_LANGUAGE_CONFLICT,
        }
        if any(reason in repeat_reasons for reason in reasons):
            return RankingDisposition.REQUEST_REPEAT
        abstention_reasons = {
            RankingReason.MISSING_NEURAL_EVIDENCE,
            RankingReason.LOW_FUSED_SCORE,
            RankingReason.LOW_FUSED_MARGIN,
        }
        if any(reason in abstention_reasons for reason in reasons):
            return RankingDisposition.ABSTAIN
        return RankingDisposition.DISPLAY

    @staticmethod
    def _neural_summary(
        *,
        candidates: tuple[Candidate, ...],
        probabilities: dict[str, float],
    ) -> tuple[str | None, float | None, float | None]:
        if not probabilities:
            return None, None, None
        original_order = {
            candidate.candidate_id: index for index, candidate in enumerate(candidates)
        }
        ordered = sorted(
            probabilities.items(), key=lambda item: (-item[1], original_order[item[0]])
        )
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
        return ordered[0][0], ordered[0][1], margin

    @staticmethod
    def _language_top_id(inputs: RankingInputs) -> str:
        original_order = {
            candidate.candidate_id: index
            for index, candidate in enumerate(inputs.candidate_set.candidates)
        }
        return min(
            inputs.generic_language_support,
            key=lambda candidate_id: (
                -inputs.generic_language_support[candidate_id],
                original_order[candidate_id],
            ),
        )

    @staticmethod
    def _tokens(value: str) -> frozenset[str]:
        return frozenset(match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value))

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / len(left.union(right))
