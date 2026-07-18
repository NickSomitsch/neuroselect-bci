# ADR 0006: Local personal-knowledge retrieval

- Status: accepted
- Date: 2026-07-17

## Decision

- Store only explicit, user-approved knowledge records in a local SQLite database. Do not
  ingest an entire message history, filesystem, or cloud account automatically.
- Scope every record by profile and require a source reference, suggestion/explanation
  permissions, enabled state, and optional validity window. Temporary current-event records
  require an expiry time.
- Support add, edit, disable, and physical delete operations. Use monotonically increasing
  revisions and optimistic revision checks so stale edits cannot silently overwrite a record.
- Treat stored content as untrusted data. A versioned conservative detector flags instruction
  overrides, role markers, and prompt-exfiltration language. Flagged records remain visible for
  editing or deletion but are quarantined from retrieval.
- Use a deterministic Unicode-aware lexical baseline for the early vertical slice. Filter by
  profile, permission, enabled state, validity, knowledge kind, and quarantine status before
  scoring. Return matched terms, source provenance, revision, and a user-visible explanation.
- Represent retrieval influence separately for every language candidate. Control candidates do
  not receive retrieval evidence, and retrieval never appends text or confirms a selection.
- Keep generated demo databases under ignored `artifacts/`; track only the synthetic source
  profiles, schema, import script, and versioned retrieval policy.

## Consequences

The baseline runs without a model download, is easy to audit, and provides deterministic tests
for permissions, stale facts, deletion, and injection quarantine. Lexical overlap is not a
semantic retrieval quality baseline and will miss paraphrases; sentence embeddings and a local
vector index remain a later measured enhancement. SQLite files are local but not encrypted by
this implementation, so deployments containing real personal data require OS-level full-disk
encryption, restrictive filesystem permissions, and documented backup deletion procedures.
