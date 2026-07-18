from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.core.models import (
    Candidate,
    CandidateKind,
    CandidateSet,
    EvidenceMode,
    NeuralSelectionEvidence,
)
from neuroselect.ranking import (
    ConfirmationLevel,
    DominanceFlag,
    FusionWeights,
    RankingDisposition,
    RankingInputs,
    RankingPolicy,
    RankingReason,
    RiskLevel,
    TransparentRanker,
    load_ranking_policy,
)
from neuroselect.retrieval import (
    CandidateRetrievalEvidence,
    KnowledgeKind,
    KnowledgeRecordInput,
    LexicalRetriever,
    RecordPermission,
    SQLiteKnowledgeStore,
)

POLICY_PATH = Path(__file__).parents[2] / "configs" / "ranking" / "default.yaml"
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def candidate(
    candidate_id: str,
    text: str,
    *,
    kind: CandidateKind | None = None,
    risk_tags: frozenset[str] = frozenset(),
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        text=text,
        kind=kind or (CandidateKind.WORD if len(text.split()) == 1 else CandidateKind.PHRASE),
        risk_tags=risk_tags,
    )


def standard_candidates() -> tuple[Candidate, ...]:
    return (
        candidate("intended", "open window"),
        candidate("likely", "thank you"),
        candidate("alternative", "some water"),
        candidate("control-other", "Other", kind=CandidateKind.CONTROL),
        candidate("control-back", "Back", kind=CandidateKind.CONTROL),
        candidate("control-cancel", "Cancel", kind=CandidateKind.CONTROL),
    )


def candidate_set(candidates: tuple[Candidate, ...] | None = None) -> CandidateSet:
    return CandidateSet(
        candidate_set_id="candidate-set-1",
        context_sha256="a" * 64,
        candidates=candidates or standard_candidates(),
        generator_revision="fixture-v1",
        prompt_revision="prompt-v1",
    )


def neural(probabilities: dict[str, float] | None) -> NeuralSelectionEvidence:
    if probabilities is None:
        return NeuralSelectionEvidence(
            evidence_id="neural-missing",
            mode=EvidenceMode.MANUAL,
            missing_reason="manual debug mode has no neural distribution",
            recorded_at=NOW,
        )
    return NeuralSelectionEvidence(
        evidence_id="neural-1",
        mode=EvidenceMode.SIMULATION,
        candidate_probabilities=probabilities,
        recorded_at=NOW,
    )


def inputs(
    probabilities: dict[str, float] | None,
    *,
    candidates: tuple[Candidate, ...] | None = None,
    language: dict[str, float] | None = None,
    personalization: dict[str, float] | None = None,
    retrieval: tuple[CandidateRetrievalEvidence, ...] = (),
) -> RankingInputs:
    visible = candidates or standard_candidates()
    language_ids = [item.candidate_id for item in visible if item.kind is not CandidateKind.CONTROL]
    default_support = 1.0 / len(language_ids)
    return RankingInputs(
        candidate_set=candidate_set(visible),
        neural_evidence=neural(probabilities),
        generic_language_support=language or dict.fromkeys(language_ids, default_support),
        personalization_lift=personalization or {},
        retrieval_evidence=retrieval,
    )


def strong_intended_probabilities() -> dict[str, float]:
    return {
        "intended": 0.62,
        "likely": 0.15,
        "alternative": 0.08,
        "control-other": 0.05,
        "control-back": 0.05,
        "control-cancel": 0.05,
    }


def test_linguistically_likely_but_unintended_candidate_cannot_override_neural_intent() -> None:
    ranking_inputs = inputs(
        strong_intended_probabilities(),
        language={"intended": 0.02, "likely": 0.93, "alternative": 0.05},
    )
    result = TransparentRanker().rank(ranking_inputs)
    likely = next(
        item for item in result.ranked_candidates if item.candidate.candidate_id == "likely"
    )

    assert result.disposition is RankingDisposition.DISPLAY
    assert result.fused_top_candidate_id == result.neural_top_candidate_id == "intended"
    assert result.display_top_candidate_id == "intended"
    assert RankingReason.LM_DOMINANCE_DETECTED in result.reason_codes
    assert DominanceFlag.LM_OVER_NEURAL.value in likely.breakdown.dominance_flags
    assert {item.candidate.candidate_id for item in result.ranked_candidates} == {
        item.candidate_id for item in standard_candidates()
    }
    assert result.requires_explicit_selection is True
    assert result.automatic_selection_permitted is False


def test_every_score_is_auditable_and_ranking_is_deterministic() -> None:
    ranking_inputs = inputs(
        strong_intended_probabilities(),
        language={"intended": 0.7, "likely": 0.2, "alternative": 0.1},
        personalization={"intended": 0.25, "likely": -0.1},
    )
    ranker = TransparentRanker()
    first = ranker.rank(ranking_inputs)
    second = ranker.rank(ranking_inputs)

    assert first == second
    assert first.policy_revision == "transparent-fusion-v1"
    assert tuple(item.rank for item in first.ranked_candidates) == (1, 2, 3, 4, 5, 6)
    for item in first.ranked_candidates:
        assert item.breakdown.total_score == pytest.approx(
            sum(item.breakdown.weighted_contributions.values())
        )
        assert set(item.breakdown.weighted_contributions) == {
            "neural",
            "generic_language",
            "personalization",
            "retrieval",
            "risk",
            "diversity",
        }


@pytest.mark.parametrize(
    ("probabilities", "expected_reason"),
    [
        (
            {
                "intended": 0.30,
                "likely": 0.25,
                "alternative": 0.20,
                "control-other": 0.10,
                "control-back": 0.10,
                "control-cancel": 0.05,
            },
            RankingReason.LOW_NEURAL_SUPPORT,
        ),
        (
            {
                "intended": 0.50,
                "likely": 0.45,
                "alternative": 0.02,
                "control-other": 0.01,
                "control-back": 0.01,
                "control-cancel": 0.01,
            },
            RankingReason.LOW_NEURAL_MARGIN,
        ),
    ],
)
def test_weak_or_ambiguous_neural_evidence_requests_repeat(
    probabilities: dict[str, float], expected_reason: RankingReason
) -> None:
    result = TransparentRanker().rank(inputs(probabilities))

    assert result.disposition is RankingDisposition.REQUEST_REPEAT
    assert expected_reason in result.reason_codes
    assert result.display_top_candidate_id is None


def test_missing_neural_evidence_abstains_instead_of_using_language_alone() -> None:
    result = TransparentRanker().rank(
        inputs(
            None,
            language={"intended": 0.01, "likely": 0.98, "alternative": 0.01},
        )
    )

    assert result.disposition is RankingDisposition.ABSTAIN
    assert RankingReason.MISSING_NEURAL_EVIDENCE in result.reason_codes
    assert result.neural_top_candidate_id is None
    assert result.display_top_candidate_id is None
    assert all(
        item.breakdown.neural is None
        and DominanceFlag.MISSING_NEURAL.value in item.breakdown.dominance_flags
        for item in result.ranked_candidates
    )


def test_neural_language_conflict_requests_repeat_and_enhanced_confirmation() -> None:
    result = TransparentRanker().rank(
        inputs(
            {
                "intended": 0.55,
                "likely": 0.43,
                "alternative": 0.01,
                "control-other": 0.005,
                "control-back": 0.003,
                "control-cancel": 0.002,
            },
            language={"intended": 0.0, "likely": 1.0, "alternative": 0.0},
        )
    )

    assert result.fused_top_candidate_id == "likely"
    assert result.neural_top_candidate_id == "intended"
    assert result.disposition is RankingDisposition.REQUEST_REPEAT
    assert result.confirmation_level is ConfirmationLevel.ENHANCED
    assert RankingReason.NEURAL_LANGUAGE_CONFLICT in result.reason_codes
    assert result.display_top_candidate_id is None


def test_control_can_rank_first_only_from_explicit_neural_support() -> None:
    result = TransparentRanker().rank(
        inputs(
            {
                "intended": 0.08,
                "likely": 0.05,
                "alternative": 0.04,
                "control-other": 0.03,
                "control-back": 0.03,
                "control-cancel": 0.77,
            },
            language={"intended": 0.7, "likely": 0.2, "alternative": 0.1},
        )
    )
    top = result.ranked_candidates[0]

    assert result.disposition is RankingDisposition.DISPLAY
    assert result.display_top_candidate_id == "control-cancel"
    assert top.candidate.kind is CandidateKind.CONTROL
    assert top.breakdown.generic_language == 0.0
    assert top.breakdown.retrieval == 0.0


def test_sensitive_candidate_is_penalized_and_requires_enhanced_confirmation() -> None:
    sensitive_candidates = list(standard_candidates())
    sensitive_candidates[0] = candidate(
        "intended",
        "medical request",
        risk_tags=frozenset({"medical"}),
    )
    result = TransparentRanker().rank(
        inputs(
            {
                "intended": 0.90,
                "likely": 0.03,
                "alternative": 0.02,
                "control-other": 0.02,
                "control-back": 0.02,
                "control-cancel": 0.01,
            },
            candidates=tuple(sensitive_candidates),
            language={"intended": 0.8, "likely": 0.1, "alternative": 0.1},
        )
    )
    top = result.ranked_candidates[0]

    assert result.disposition is RankingDisposition.DISPLAY
    assert result.confirmation_level is ConfirmationLevel.ENHANCED
    assert RankingReason.SENSITIVE_CANDIDATE in result.reason_codes
    assert top.risk_level is RiskLevel.HIGH
    assert top.confirmation_level is ConfirmationLevel.ENHANCED
    assert top.breakdown.weighted_contributions["risk"] == pytest.approx(-0.35)


def test_diversity_penalty_reorders_near_duplicate_phrases() -> None:
    candidates = (
        candidate("cold-water", "cold water"),
        candidate("still-water", "still water"),
        candidate("call-anna", "call Anna"),
    )
    result = TransparentRanker().rank(
        inputs(
            {"cold-water": 0.34, "still-water": 0.335, "call-anna": 0.325},
            candidates=candidates,
            language={"cold-water": 0.34, "still-water": 0.34, "call-anna": 0.32},
        )
    )

    assert tuple(item.candidate.candidate_id for item in result.ranked_candidates) == (
        "cold-water",
        "call-anna",
        "still-water",
    )
    still_water = result.ranked_candidates[2]
    assert still_water.breakdown.diversity_adjustment == pytest.approx(-0.08 / 3)


def test_retrieval_score_and_provenance_remain_separate_and_visible(tmp_path: Path) -> None:
    candidates = (
        candidate("water", "still water"),
        candidate("rest", "rest now"),
    )
    with SQLiteKnowledgeStore(tmp_path / "ranking.sqlite3") as store:
        store.add(
            profile_id="profile-a",
            record=KnowledgeRecordInput(
                record_id="water-preference",
                kind=KnowledgeKind.PREFERENCE,
                content="Mara prefers still water at room temperature.",
                source="user:manual",
                permissions=frozenset({RecordPermission.SUGGEST, RecordPermission.EXPLAIN}),
            ),
            at_time=NOW,
        )
        retrieval = LexicalRetriever(store).retrieve_for_candidates(
            profile_id="profile-a",
            confirmed_text="I need",
            candidates=candidates,
            at_time=NOW,
        )
        result = TransparentRanker().rank(
            inputs(
                {"water": 0.7, "rest": 0.3},
                candidates=candidates,
                language={"water": 0.5, "rest": 0.5},
                retrieval=retrieval,
            )
        )

    water = next(
        item for item in result.ranked_candidates if item.candidate.candidate_id == "water"
    )
    assert water.breakdown.retrieval > 0.0
    assert water.breakdown.weighted_contributions["retrieval"] == pytest.approx(
        0.12 * water.breakdown.retrieval
    )
    assert water.retrieval_hits[0].record.record_id == "water-preference"
    assert "user:manual" in water.retrieval_hits[0].explanation


def test_policy_thresholds_can_force_abstention_without_exposing_a_top_choice() -> None:
    base_policy = load_ranking_policy(POLICY_PATH)
    strict_policy = RankingPolicy.model_validate(
        {**base_policy.model_dump(), "minimum_fused_top_score": 0.9}
    )
    result = TransparentRanker(strict_policy).rank(inputs(strong_intended_probabilities()))

    assert result.disposition is RankingDisposition.ABSTAIN
    assert RankingReason.LOW_FUSED_SCORE in result.reason_codes
    assert result.display_top_candidate_id is None


def test_policy_and_input_contracts_reject_opaque_or_misaligned_evidence(
    tmp_path: Path,
) -> None:
    policy = load_ranking_policy(POLICY_PATH)
    assert policy.weights.neural == 0.65
    assert policy.high_risk_tags == frozenset({"medical", "financial", "legal", "consent"})

    with pytest.raises(ValidationError, match="must sum to one"):
        FusionWeights(neural=0.65, generic_language=0.1, personalization=0.1, retrieval=0.1)
    with pytest.raises(ValidationError, match="cannot exceed 0.35"):
        FusionWeights(neural=0.6, generic_language=0.2, personalization=0.1, retrieval=0.1)
    with pytest.raises(ValidationError, match="must be disjoint"):
        RankingPolicy.model_validate(
            {
                **policy.model_dump(),
                "elevated_risk_tags": ["medical"],
                "high_risk_tags": ["medical"],
            }
        )

    non_mapping = tmp_path / "invalid.yaml"
    non_mapping.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_ranking_policy(non_mapping)

    bad_neural = neural({"intended": 1.0})
    with pytest.raises(ValidationError, match="exactly the visible candidates"):
        RankingInputs(
            candidate_set=candidate_set(),
            neural_evidence=bad_neural,
            generic_language_support={
                "intended": 0.4,
                "likely": 0.3,
                "alternative": 0.3,
            },
        )

    with pytest.raises(ValidationError, match="only language candidates"):
        inputs(
            strong_intended_probabilities(),
            personalization={"control-cancel": 1.0},
        )
