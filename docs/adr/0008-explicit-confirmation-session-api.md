# ADR 0008: Explicit-confirmation session API

- Status: accepted
- Date: 2026-07-18

## Decision

- Add an in-memory session orchestrator that composes the deterministic candidate generator,
  personal retrieval, manual/simulated neural input, transparent ranker, and pure state machine.
  Component outputs remain separately typed and visible.
- Expose a versioned FastAPI surface bound only to the configured loopback address. Setup and CI
  do not start a server, download model weights, or enable external network access.
- Support manual keyboard debugging and seeded simulation. Simulation target indices are explicit
  experiment inputs; simulator ground truth is retained internally and omitted from session views.
- Keep generated candidates provisional. A standard candidate requires an explicit selection;
  a risky or non-top candidate requires a second explicit confirmation. Repeat/abstain ranking
  dispositions prevent simulated neural selection from being accepted.
- Implement explicit reject, repeat, back, clear, other, and cancel operations through the state
  machine. Terminal finalized/cancelled sessions reject further mutation.
- Require a dedicated finalization request followed by a one-time nonce, exact confirmed-text
  SHA-256, and `explicit_confirmation=true`. Challenges expire after the configured five-minute
  window. Selected risky content additionally requires a high-risk acknowledgement.
- Return aggregate round, selection, rejection, repeat, backtrack, clear, other, and manual-input
  counters. Do not expose confirmation nonces outside the challenge response or simulator ground
  truth in ordinary session state.
- Allow the local SQLite store to cross the ASGI thread boundary while serializing database
  access with a reentrant lock. Run orchestration calls on the single application event loop.

## Consequences

The repository now has a complete CPU-only application service for deterministic API and future
UI integration tests. Sessions and action history are intentionally process-local and disappear
on restart. This API is not authenticated, multi-user, multi-worker, remotely deployable, or
suitable for clinical operation. Persistent sessions, authentication, rate limiting, CSRF/CORS
policy, encryption key management, and deployment hardening remain out of scope for the MVP.
