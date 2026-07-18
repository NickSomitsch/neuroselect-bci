"""Local-first personal knowledge storage and safe deterministic retrieval."""

from neuroselect.core.models import KnowledgeKind, RecordPermission
from neuroselect.retrieval.models import (
    CandidateRetrievalEvidence,
    InjectionRisk,
    KnowledgeRecordInput,
    KnowledgeRecordPatch,
    RetrievalHit,
    RetrievalPolicy,
    RetrievalRequest,
    StoredKnowledgeRecord,
)
from neuroselect.retrieval.retriever import (
    LexicalRetriever,
    load_retrieval_policy,
)
from neuroselect.retrieval.safety import detect_prompt_injection
from neuroselect.retrieval.store import (
    KnowledgeRecordConflictError,
    KnowledgeRecordNotFoundError,
    KnowledgeStoreError,
    KnowledgeStoreSchemaError,
    SQLiteKnowledgeStore,
)

__all__ = [
    "CandidateRetrievalEvidence",
    "InjectionRisk",
    "KnowledgeKind",
    "KnowledgeRecordConflictError",
    "KnowledgeRecordInput",
    "KnowledgeRecordNotFoundError",
    "KnowledgeRecordPatch",
    "KnowledgeStoreError",
    "KnowledgeStoreSchemaError",
    "LexicalRetriever",
    "RecordPermission",
    "RetrievalHit",
    "RetrievalPolicy",
    "RetrievalRequest",
    "SQLiteKnowledgeStore",
    "StoredKnowledgeRecord",
    "detect_prompt_injection",
    "load_retrieval_policy",
]
