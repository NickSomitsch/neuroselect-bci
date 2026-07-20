# Threat model

## Assets and trust boundaries

Protected assets include raw EEG, derived epochs, confirmed and provisional messages, personal RAG
records, adapter training text, checkpoints, confirmation nonces, and research provenance. The
current trusted boundary is one local OS user, a loopback FastAPI process, the local browser, and
files produced or explicitly imported by that user. Dataset hosts, downloaded files, model
checkpoints, retrieved text, browser extensions, other local processes, and shared artifacts are
outside or across that boundary.

## Threats and controls

| Threat | Current controls | Residual risk / required follow-up |
|---|---|---|
| Public or LAN access to sessions | Configuration rejects non-loopback service addresses | No authentication; another local process or malicious browser context may still reach the API |
| Prompt injection in personal records | Permission, profile, validity, enabled-state, quarantine, and lexical injection filters | Lexical filters are incomplete; never treat retrieved instructions as authority |
| Language prior overwhelms EEG | Separate normalized signals, dominance flags, repeat/abstain thresholds, explicit selection | Hand-set weights are not proven optimal and candidate wording can still bias selection |
| Incorrect attribution of generated text | Provisional candidates, enhanced selection confirmation, nonce-bound final confirmation | Confirmation cannot prove voluntariness, capacity, comprehension, or authorship |
| Malicious or corrupt checkpoints | SHA-256 manifests; EEGNet tensor-only weights-only loading | Classical joblib is executable and must be locally trusted; checksums prove identity, not safety |
| Train/test or session leakage | Subject/session/trial group checks, pinned splits, chronological adaptation protocol | New importers or analyses can reintroduce leakage and require explicit tests |
| Sensitive data in logs/artifacts | No telemetry, ignored local directories, explicit artifact commands | Terminal logs, environment provenance, reports, backups, and screenshots may still disclose data |
| Stale or incorrect personal facts | Revision, validity, enable/disable, provenance, and deletion support | A valid record may still be wrong; the user must be able to inspect and reject it |
| Dependency or dataset compromise | Locked dependencies, pinned dataset inventory, checksum verification, read-only CI permissions | Upstream compromise and local package execution remain possible; review updates independently |
| Confirmation replay or mismatch | Short-lived nonce, session ID, exact text hash, single-use state transition | The prototype has no authenticated user identity or protected display channel |

## Security assumptions

The host account, browser, and local filesystem are trusted; the API is not reachable from an
untrusted network; imported data and checkpoints are reviewed; and the operator can distinguish
synthetic from personal data. If any assumption is false, stop using the prototype until the
deployment adds authentication, authorization, transport protection, isolation, audit controls,
and an incident-response plan.

## Out of scope for the current release

Cloud deployment, multi-user tenancy, mobile clients, remote EEG upload, federated learning,
clinical integration, live amplifiers, and human-subject identity management are not implemented.
Any future live adapter must update this threat model before accepting hardware data.
