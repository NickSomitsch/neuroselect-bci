# ADR 0002: P300 and bigP3BCI Study P for the MVP

- Status: accepted
- Date: 2026-07-17

## Context

The MVP needs public, replayable EEG with multiple subjects and sessions, low compute requirements, calibrated selection probabilities, and a direct relationship to visible candidate choice.

Imagined speech is insufficiently reliable for the proposed claims. Motor imagery primarily supports navigation commands. SSVEP is a viable selection paradigm but adds continuous-flicker and accessibility concerns.

## Decision

Use P300 as the sole real-EEG paradigm for the MVP and bigP3BCI Study P v1.0.0 as the primary dataset. Pin source checksums before preprocessing and keep raw source data immutable outside Git.

P300 evidence maps to a visible tile identifier, not word meaning. In counterfactual replay, preserve recorded timing and target/non-target evidence while mapping the attended target to a generated candidate tile. Label the result as offline replay simulation and report the dataset's original spelling-task accuracy separately.

## Participant-metadata audit

Resolved on 2026-07-18 for release 1.0.0. The official SHA-256 inventory was used to select one
EDF per Study P subject, and only the fixed EDF patient-header range was read. All 19 subject
headers contain the dataset's `NonALS` label. This agrees with the source study table and the
current MOABB machine-readable `has_als: false` setting.

MOABB's generated Study P overview and class docstring nevertheless say "19 ALS subjects," while
the same generated page's participant field says "healthy." NeuroSelect therefore records only
the source label `NonALS`; it does not infer a broader clinical or health status. See the
[dataset card](../dataset-card.md) for the evidence hierarchy and limitations.

## Consequences

The BCI adapter remains generic so later SSVEP or motor-imagery navigation adapters can be added. Imagined-speech decoding remains explicitly out of scope.
