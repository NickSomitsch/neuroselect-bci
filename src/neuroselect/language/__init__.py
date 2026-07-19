"""Structured, non-mutating candidate-language generation."""

from neuroselect.language.fixture import FixtureCandidateBackend, load_fixture_backend_config
from neuroselect.language.generation import (
    CandidateBackend,
    CandidateGenerationError,
    CandidateGenerator,
    parse_structured_proposals,
)
from neuroselect.language.models import (
    BackendMetadata,
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateProposal,
    CandidateRiskPolicy,
    CandidateRiskRule,
    ControlPath,
    FixtureBackendConfig,
    FixtureRule,
    GenerationDiagnostics,
    ProposalRejectionReason,
)
from neuroselect.language.risk import CandidateRiskTagger, load_candidate_risk_policy

__all__ = [
    "BackendMetadata",
    "CandidateBackend",
    "CandidateGenerationError",
    "CandidateGenerationRequest",
    "CandidateGenerationResult",
    "CandidateGenerator",
    "CandidateProposal",
    "CandidateRiskPolicy",
    "CandidateRiskRule",
    "CandidateRiskTagger",
    "ControlPath",
    "FixtureBackendConfig",
    "FixtureCandidateBackend",
    "FixtureRule",
    "GenerationDiagnostics",
    "ProposalRejectionReason",
    "load_candidate_risk_policy",
    "load_fixture_backend_config",
    "parse_structured_proposals",
]
