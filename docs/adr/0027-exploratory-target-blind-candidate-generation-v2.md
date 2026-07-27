# ADR 0027: Exploratory target-blind candidate generation v2

- Status: Accepted
- Date: 2026-07-27

## Context

The frozen primary language evaluation made the intended synthetic span available in 28.7% of
rounds and never made every span of a complete message available. The broad v1 non-test
vocabularies contain useful phrases, but selecting nine visible candidates from those pools often
crowds out a relevant object, time qualifier, or ending. The opening round is unconstrained.

The test benchmark structure and primary outcomes were already visible before this change.
Candidate-generation v2 therefore cannot become new primary or confirmatory evidence.

## Decision

Add one explicitly exploratory, profile-conditioned contextual generator fitted only on train and
validation messages. It retrieves nine source-grounded phrases using visible confirmed context and
round number. It uses fixed grammatical routing for openings, request objects, time qualifiers,
locations, and request endings. It does not load Qwen, retrain an adapter, or receive an intended
target parameter.

The fitting function structurally traverses only the configured train and validation mappings.
Every candidate-bank entry retains source split and message provenance. Tests require the fitted
bank digest to remain unchanged when the complete test partition is replaced. Intended test spans
are compared with the generated candidates only after generation.

The comparison is paired to the frozen 3,990-span v1 artifact. It reports overall, profile, and
span-position availability; complete-message availability; gained and lost trials; and 10,000
message-clustered bootstrap resamples. The source configuration records
`exploratory_test_exposed`, and outcome-based omission remains forbidden.

## Consequences

The result may be reported only as an exploratory supplement and cannot replace the frozen v1
result. Profile conditioning and grammar routing differ from the generic v1 generator, so any
change is an algorithm-level comparison rather than an isolated prompt ablation.

The full evaluation is deterministic and CPU-only. This makes it reproducible on the supported
MacBook without downloading or running the 4B language model, while preserving the fixed nine
visible language-candidate budget.
