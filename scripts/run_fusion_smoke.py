"""Run deterministic candidate, retrieval, neural, and ranking fusion in memory."""

from __future__ import annotations

from datetime import datetime

from neuroselect.bci import SeededNeuralSimulator, SimulationConfig
from neuroselect.core.models import CandidateKind
from neuroselect.language import (
    CandidateGenerationRequest,
    CandidateGenerator,
    FixtureCandidateBackend,
)
from neuroselect.ranking import RankingInputs, TransparentRanker
from neuroselect.retrieval import (
    KnowledgeRecordInput,
    LexicalRetriever,
    SQLiteKnowledgeStore,
)
from neuroselect.synthetic import load_profiles

SMOKE_TIME = datetime.fromisoformat("2026-07-18T12:00:00+02:00")


def main() -> None:
    generation = CandidateGenerator(FixtureCandidateBackend()).generate(
        CandidateGenerationRequest(confirmed_text="I would like")
    )
    target = next(
        candidate
        for candidate in generation.candidate_set.candidates
        if candidate.kind is not CandidateKind.CONTROL
    )
    profile = next(
        profile for profile in load_profiles() if profile.profile_id == "synthetic-concise"
    )

    with SQLiteKnowledgeStore(":memory:") as store:
        for record in profile.knowledge:
            store.add(
                profile_id=profile.profile_id,
                record=KnowledgeRecordInput.model_validate(record.model_dump()),
                at_time=SMOKE_TIME,
            )
        retrieval = LexicalRetriever(store).retrieve_for_candidates(
            profile_id=profile.profile_id,
            confirmed_text="I would like",
            candidates=generation.candidate_set.candidates,
            at_time=SMOKE_TIME,
        )
        simulated = SeededNeuralSimulator(
            SimulationConfig(
                target_concentration=60.0,
                lapse_probability=0.0,
                ambiguous_probability=0.0,
            )
        ).simulate(
            candidate_ids=tuple(
                candidate.candidate_id for candidate in generation.candidate_set.candidates
            ),
            intended_candidate_id=target.candidate_id,
            session_id="fusion-smoke",
            round_index=0,
        )
        result = TransparentRanker().rank(
            RankingInputs(
                candidate_set=generation.candidate_set,
                neural_evidence=simulated.evidence,
                generic_language_support=generation.generic_language_support,
                retrieval_evidence=retrieval,
            )
        )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
