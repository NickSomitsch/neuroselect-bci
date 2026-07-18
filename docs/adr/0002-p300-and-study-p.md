# ADR 0002: P300 and bigP3BCI Study P for the MVP

- Status: accepted
- Date: 2026-07-17

## Context

The MVP needs public, replayable EEG with multiple subjects and sessions, low compute requirements, calibrated selection probabilities, and a direct relationship to visible candidate choice.

Imagined speech is insufficiently reliable for the proposed claims. Motor imagery primarily supports navigation commands. SSVEP is a viable selection paradigm but adds continuous-flicker and accessibility concerns.

## Decision

Use P300 as the sole real-EEG paradigm for the MVP and bigP3BCI Study P v1.0.0 as the primary dataset. Pin source checksums before preprocessing and keep raw source data immutable outside Git.

P300 evidence maps to a visible tile identifier, not word meaning. In counterfactual replay, preserve recorded timing and target/non-target evidence while mapping the attended target to a generated candidate tile. Label the result as offline replay simulation and report the dataset's original spelling-task accuracy separately.

## Known metadata discrepancy

Current generated catalog text is inconsistent about Study P participant health status. The original predictive-spelling publication describes able-bodied participants, while one generated overview label says ALS. Before publishing a data card, verify the primary paper, participant metadata, and source files and document the discrepancy without inferring clinical status.

## Consequences

The BCI adapter remains generic so later SSVEP or motor-imagery navigation adapters can be added. Imagined-speech decoding remains explicitly out of scope.
