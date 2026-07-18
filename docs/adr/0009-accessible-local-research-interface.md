# ADR 0009: Accessible local research interface

- Status: Accepted
- Date: 2026-07-18

## Context

The simulated vertical slice needs a convincing interface before adding real EEG decoding. The
interface must preserve the evidence boundaries established by the ranking and session APIs. It
must never turn a model suggestion into confirmed text without an explicit action, and it must be
usable without a mouse.

## Decision

Use the existing React and TypeScript application as a local web client for the loopback FastAPI
service. Vite proxies `/api` and `/health` to `127.0.0.1:8000` in development, avoiding a broader
cross-origin policy.

The interface:

- obtains synthetic profile summaries from `/api/v1/profiles` rather than embedding profile data;
- lets a researcher configure 4, 6, 8, or 12 total candidate targets and a one-to-eight-word phrase
  bound for each new round;
- presents language candidates separately from application-owned other, back, and cancel paths;
- labels neural, generic-language, personalization, and retrieval evidence independently;
- exposes retrieval record provenance and dominance warnings in text;
- blocks simulated language selection when the ranker requests repeat or abstains;
- requires an additional dialog for provisional enhanced-risk or non-top selections;
- binds final confirmation to the API's exact text, hash, one-time nonce, and optional high-risk
  acknowledgement;
- uses native controls, large targets, visible focus, skip navigation, live status and error
  regions, number and repeat shortcuts, and optional focus scanning;
- supports adjustable scan timing, high contrast, reduced motion, and narrow-screen layouts; and
- displays recorded-EEG replay as unavailable until the dataset and streaming adapter exist.

Automatic focus scanning is disabled by default. It changes focus only and never activates a
candidate. Candidate-number shortcuts are ignored while the user is entering text or changing a
form control.

## Consequences

The first simulated/manual end-to-end demonstration now runs on a normal developer computer with
no GPU and no downloaded model or EEG data. Evidence explanations are intentionally dense and may
need usability testing before use with participants.

The UI's TypeScript types mirror the versioned API today. A generated OpenAPI client can replace
them once the API stabilizes. Recorded P300 playback, live EEG, participant authentication,
deployment, and claims about clinical accessibility remain out of scope for this step.
