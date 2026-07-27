# ADR 0026: Offline journal publication strategy

- Status: Accepted
- Date: 2026-07-27

## Context

NeuroSelect now has clean, checksum-addressed controlled-simulation, held-out synthetic-language,
original-task Study P, and counterfactual-replay artifacts. The evidence supports an offline
computational methods paper, but it does not include participant use of NeuroSelect. A publication
plan must not turn protocol compliance into an efficacy claim or hide target-availability and
profile-heterogeneity findings.

The primary analyses were completed before this publication protocol was written. Calling the
entire paper preregistered would therefore be inaccurate. New statistical analyses and any
candidate-generation v2 work can, however, be locked prospectively from this decision.

## Decision

Target *Research in Biomedical Engineering and Technology* with an Original Research article
framed as an offline BCI/neural-engineering methods and validation study. Use *Frontiers in
Neuroinformatics* only as a funded fallback. Consider JOSS later as a distinct software paper
after sustained public maturation; it is not a venue switch for the empirical manuscript.

The tracked `offline_methods_v1` protocol fixes five research questions, the primary estimands,
the exact source runs and manifest digests, allowed and prohibited claims, and external submission
gates. Existing Steps 9 and 11–13 are frozen primary evidence. Controlled simulation remains
engineering context. A future target-blind candidate-generation v2 is exploratory, must be
reported regardless of outcome, and cannot replace an unfavorable primary result.

The protocol checker distinguishes two states:

- **protocol ready** means the framing is internally coherent and every frozen source matches;
- **submission ready** additionally requires written open-access, secondary-use ethics, domain
  review, and author-metadata decisions.

No new participants will be recruited. The paper describes secondary analysis of public,
deidentified EEG plus synthetic language data and retains all existing mind-reading, clinical,
medical-device, and automatic-selection claim boundaries.

## Consequences

Publication analysis and manuscript generation must consume the tracked protocol rather than
choosing source runs ad hoc. Source replacement requires a new protocol revision and an explicit
explanation; the v1 evidence cannot be silently overwritten.

The repository may build and validate the paper before the external submission gates are
complete. Actual submission fails closed until those gates are recorded as satisfied.
