# ADR 0016: Counterfactual v2 target and message provenance

- Status: Accepted
- Date: 2026-07-23

## Context

Counterfactual v1 always represented the mapped candidate as an available target. When a generated
phrase was absent, the recorded neural signature was correctly moved to `Other`, but the shared
trial record could then count `Other` as successful phrase recall. Counterfactual trials also used
the EEG subject as the profile identity and discarded synthetic message/span structure.

## Decision

Protocol v2 records the intended text and optional visible intended candidate separately from the
candidate receiving the recorded target signature. An absent intended candidate has no target
rank, cannot count toward top-k recall or message completion, and may only count toward a separate
`Other`-fallback metric.

Each trial also records synthetic profile, message, span, EEG subject, and EEG session identities.
Complete-message metrics require every declared span. Hierarchical EEG intervals cluster by the
recorded subject, while profile slices use the synthetic profile.

Replay duration is the interval from the first to last recorded flash onset plus the median
inter-flash interval. Locked explicit-action and enhanced-confirmation costs are then added.
Protocol v1 artifacts remain readable through additive compatibility defaults, but new prepared
inputs and outputs use schema and protocol version 2.

## Consequences

Natural candidate-generation experiments can measure target availability without treating a safe
fallback as successful language prediction. Target-present ranking experiments remain possible,
but their intended-candidate presence is explicit. Modeled timing is grounded in the source event
stream and remains distinct from measured human communication speed.
