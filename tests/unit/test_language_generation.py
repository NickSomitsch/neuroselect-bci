from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuroselect.core.models import CandidateKind
from neuroselect.language import (
    BackendMetadata,
    CandidateGenerationError,
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateGenerator,
    CandidateProposal,
    ControlPath,
    FixtureBackendConfig,
    FixtureCandidateBackend,
    ProposalRejectionReason,
    load_fixture_backend_config,
    parse_structured_proposals,
)

FIXTURE_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "language" / "fixture.yaml"
STUB_METADATA = BackendMetadata(
    backend_id="test-backend",
    model_id="test-model",
    model_revision="test-model-v1",
    generator_revision="test-generator-v1",
    prompt_revision="test-prompt-v1",
    deterministic=True,
)


class StubBackend:
    metadata = STUB_METADATA

    def __init__(self, proposals: tuple[CandidateProposal, ...]) -> None:
        self.proposals = proposals

    def generate(self, request: CandidateGenerationRequest) -> tuple[CandidateProposal, ...]:
        del request
        return self.proposals


def proposal(text: str, support: float) -> CandidateProposal:
    return CandidateProposal(text=text, support=support)


@pytest.mark.parametrize("candidate_count", [4, 6, 8, 12])
def test_fixture_generates_exact_visible_count_with_explicit_controls(
    candidate_count: int,
) -> None:
    request = CandidateGenerationRequest.model_validate(
        {"confirmed_text": "", "candidate_count": candidate_count}
    )
    result = CandidateGenerator(FixtureCandidateBackend()).generate(request)
    candidates = result.candidate_set.candidates
    language_candidates = tuple(
        candidate for candidate in candidates if candidate.kind is not CandidateKind.CONTROL
    )

    assert len(candidates) == candidate_count
    assert len(language_candidates) == candidate_count - 3
    assert tuple(candidate.candidate_id for candidate in candidates[-3:]) == (
        "control-other",
        "control-back",
        "control-cancel",
    )
    assert result.control_actions == {
        "control-other": ControlPath.OTHER,
        "control-back": ControlPath.BACK,
        "control-cancel": ControlPath.CANCEL,
    }
    assert set(result.generic_language_support) == {
        candidate.candidate_id for candidate in language_candidates
    }
    assert sum(result.generic_language_support.values()) == pytest.approx(1.0)


def test_default_fixture_supports_words_and_short_phrases() -> None:
    result = CandidateGenerator(FixtureCandidateBackend()).generate(CandidateGenerationRequest())
    kinds = {candidate.kind for candidate in result.candidate_set.candidates}

    assert CandidateKind.WORD in kinds
    assert CandidateKind.PHRASE in kinds
    assert all(
        len(candidate.text.split()) <= 4
        for candidate in result.candidate_set.candidates
        if candidate.kind is not CandidateKind.CONTROL
    )


def test_fixture_honors_a_single_word_only_display() -> None:
    result = CandidateGenerator(FixtureCandidateBackend()).generate(
        CandidateGenerationRequest(candidate_count=12, maximum_phrase_tokens=1)
    )

    assert all(
        candidate.kind is CandidateKind.WORD for candidate in result.candidate_set.candidates[:-3]
    )


def test_suffix_fixture_is_reproducible_after_context_normalization() -> None:
    generator = CandidateGenerator(FixtureCandidateBackend())

    first = generator.generate(CandidateGenerationRequest(confirmed_text="Could   you"))
    unrelated = generator.generate(CandidateGenerationRequest(confirmed_text="Hello"))
    second = generator.generate(CandidateGenerationRequest(confirmed_text="  Could you  "))

    assert first == second
    assert first != unrelated
    assert first.candidate_set.context_sha256 == (
        "f9208b8526553af6f29c6c0bc89a331a1ceb949ea31a960b67989051e11fa14d"
    )
    assert first.backend.deterministic is True
    assert first.candidate_set.generator_revision == "deterministic-generator-v1"
    assert first.candidate_set.prompt_revision == "structured-next-span-v1"
    assert first.candidate_set.candidates[0].text == "please help"


def test_policy_filters_duplicates_reserved_controls_and_unsafe_text() -> None:
    backend = StubBackend(
        (
            proposal("Thank you!", 0.9),
            proposal(" thank   you ", 0.8),
            proposal("Other", 0.7),
            proposal("this phrase has five words", 0.6),
            proposal("\0unsafe", 0.5),
            proposal("   ", 0.4),
        )
    )
    result = CandidateGenerator(backend).generate(CandidateGenerationRequest(candidate_count=4))

    assert result.candidate_set.candidates[0].text == "Thank you!"
    assert result.generic_language_support == {result.candidate_set.candidates[0].candidate_id: 1.0}
    assert result.diagnostics.rejected_by_reason == {
        ProposalRejectionReason.DUPLICATE: 1,
        ProposalRejectionReason.RESERVED_CONTROL: 1,
        ProposalRejectionReason.TOO_LONG: 1,
        ProposalRejectionReason.UNSAFE_TEXT: 1,
        ProposalRejectionReason.EMPTY: 1,
    }


def test_zero_backend_support_becomes_uniform_without_claiming_calibration() -> None:
    backend = StubBackend(
        (proposal("alpha", 0.0), proposal("beta gamma", 0.0), proposal("delta", 0.0))
    )
    result = CandidateGenerator(backend).generate(CandidateGenerationRequest(candidate_count=6))

    assert tuple(result.generic_language_support.values()) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_generation_fails_closed_when_safe_unique_output_is_insufficient() -> None:
    backend = StubBackend((proposal("only one", 1.0),))

    with pytest.raises(CandidateGenerationError, match="3 required"):
        CandidateGenerator(backend).generate(CandidateGenerationRequest(candidate_count=6))


def test_structured_response_parser_accepts_only_the_versioned_shape() -> None:
    proposals = parse_structured_proposals(
        json.dumps(
            {
                "candidates": [
                    {"text": "hello", "support": 0.7},
                    {"text": "thank you", "support": 0.3},
                ]
            }
        )
    )
    assert proposals == (proposal("hello", 0.7), proposal("thank you", 0.3))

    for invalid in (
        "not-json",
        '{"candidates": []}',
        '{"candidates": [{"text": "hello", "support": 1, "unknown": true}]}',
        '[{"text": "hello", "support": 1}]',
    ):
        with pytest.raises(CandidateGenerationError, match="invalid structured"):
            parse_structured_proposals(invalid)


def test_fixture_configuration_is_strict_and_versioned(tmp_path: Path) -> None:
    config = load_fixture_backend_config(FIXTURE_CONFIG_PATH)
    assert config.schema_version == "1.0"
    assert len(config.default_candidates) == 14
    assert config.metadata == FixtureCandidateBackend(config).metadata

    non_mapping = tmp_path / "non-mapping.yaml"
    non_mapping.write_text("- invalid\n- config\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_fixture_backend_config(non_mapping)

    payload = config.model_dump(mode="json")
    payload["rules"] = [payload["rules"][0], payload["rules"][0]]
    with pytest.raises(ValidationError, match="suffixes must be unique"):
        FixtureBackendConfig.model_validate(payload)


def test_result_rejects_crossed_language_and_control_evidence() -> None:
    result = CandidateGenerator(FixtureCandidateBackend()).generate(CandidateGenerationRequest())
    payload = result.model_dump()
    payload["generic_language_support"] = {"control-other": 1.0}

    with pytest.raises(ValidationError, match="exactly the language candidates"):
        CandidateGenerationResult.model_validate(payload)

    payload = result.model_dump()
    payload["control_actions"] = {"control-other": "other"}
    with pytest.raises(ValidationError, match="exactly the control candidates"):
        CandidateGenerationResult.model_validate(payload)


def test_request_rejects_unsupported_display_and_phrase_limits() -> None:
    with pytest.raises(ValidationError):
        CandidateGenerationRequest(candidate_count=5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CandidateGenerationRequest(maximum_phrase_tokens=0)
    with pytest.raises(ValidationError):
        CandidateProposal(text="invalid", support=float("inf"))
