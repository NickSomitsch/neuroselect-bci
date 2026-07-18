"""Auditable condition catalog for the planned comparison matrix."""

from __future__ import annotations

from neuroselect.evaluation.models import (
    ConditionAvailability,
    ConditionDefinition,
    ConditionFamily,
    EvaluationCondition,
    NeuralMode,
    RankingMode,
    RetrievalMode,
)


def condition_catalog() -> tuple[ConditionDefinition, ...]:
    """Return every planned condition, including honest dependency-gated entries."""

    unavailable_lora = (
        "A real personal LoRA adapter and held-out adapter evaluation are not implemented yet."
    )
    return (
        ConditionDefinition(
            condition=EvaluationCondition.A_BCI_ONLY,
            label="A. Simulated BCI only",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.NEURAL_ONLY,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.NONE,
            safeguards_enabled=False,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.B_GENERIC_LANGUAGE_ONLY,
            label="B. Generic language only",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.LANGUAGE_ONLY,
            neural_mode=NeuralMode.MISSING,
            retrieval_mode=RetrievalMode.NONE,
            safeguards_enabled=False,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.C_NEURAL_LANGUAGE,
            label="C. Simulated neural plus generic language",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.WEIGHTED_BASELINE,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.NONE,
            safeguards_enabled=False,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.D_NEURAL_PERSONALIZED,
            label="D. Neural plus personalized LoRA",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.UNAVAILABLE,
            ranking_mode=RankingMode.WEIGHTED_BASELINE,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.NONE,
            personalization_enabled=True,
            safeguards_enabled=False,
            unavailable_reason=unavailable_lora,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.E_NEURAL_PERSONALIZED_RAG,
            label="E. Neural plus LoRA and RAG",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.UNAVAILABLE,
            ranking_mode=RankingMode.WEIGHTED_BASELINE,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.CURRENT,
            personalization_enabled=True,
            safeguards_enabled=False,
            unavailable_reason=unavailable_lora,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.F_COMPLETE_SYSTEM,
            label="F. Complete calibrated system",
            family=ConditionFamily.BASELINE,
            availability=ConditionAvailability.UNAVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.CURRENT,
            personalization_enabled=True,
            safeguards_enabled=True,
            unavailable_reason=(
                "The complete condition requires the real LoRA and calibrated P300 decoder; "
                "the current seeded simulator is not a substitute."
            ),
        ),
        ConditionDefinition(
            condition=EvaluationCondition.CURRENT_NEURAL_LANGUAGE_RAG,
            label="Current neural, language, and RAG without abstention",
            family=ConditionFamily.CURRENT_SYSTEM,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.WEIGHTED_BASELINE,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.CURRENT,
            safeguards_enabled=False,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.CURRENT_SAFE_FUSION,
            label="Current transparent fusion with repeat and abstention",
            family=ConditionFamily.CURRENT_SYSTEM,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.CURRENT,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_UNIFORM_NEURAL,
            label="Ablation: replace neural evidence with uniform probabilities",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.UNIFORM,
            retrieval_mode=RetrievalMode.CURRENT,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_SHUFFLED_NEURAL,
            label="Ablation: deterministically shuffle neural probabilities",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SHUFFLED,
            retrieval_mode=RetrievalMode.CURRENT,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_REMOVE_RAG,
            label="Ablation: remove RAG",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.NONE,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_SHUFFLED_RETRIEVAL,
            label="Ablation: shuffle retrieved records across candidates",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.SHUFFLED,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_IRRELEVANT_RETRIEVAL,
            label="Ablation: inject an irrelevant retrieved record",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.IRRELEVANT,
            safeguards_enabled=True,
        ),
        ConditionDefinition(
            condition=EvaluationCondition.ABLATION_REMOVE_CONTEXT,
            label="Ablation: remove confirmed conversation context",
            family=ConditionFamily.ABLATION,
            availability=ConditionAvailability.AVAILABLE,
            ranking_mode=RankingMode.TRANSPARENT_SAFE_FUSION,
            neural_mode=NeuralMode.SIMULATED,
            retrieval_mode=RetrievalMode.CURRENT,
            safeguards_enabled=True,
        ),
    )


def condition_by_id(condition: EvaluationCondition) -> ConditionDefinition:
    return next(item for item in condition_catalog() if item.condition is condition)
