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
    ControlPath,
    FixtureBackendConfig,
    FixtureRule,
    GenerationDiagnostics,
    ProposalRejectionReason,
)

__all__ = [
    "BackendMetadata",
    "CandidateBackend",
    "CandidateGenerationError",
    "CandidateGenerationRequest",
    "CandidateGenerationResult",
    "CandidateGenerator",
    "CandidateProposal",
    "ControlPath",
    "FixtureBackendConfig",
    "FixtureCandidateBackend",
    "FixtureRule",
    "GenerationDiagnostics",
    "ProposalRejectionReason",
    "load_fixture_backend_config",
    "parse_structured_proposals",
]
