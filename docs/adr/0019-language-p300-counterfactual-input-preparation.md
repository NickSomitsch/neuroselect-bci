# ADR 0019: Language/P300 counterfactual input preparation

- Status: Accepted
- Date: 2026-07-24

## Context

The held-out language evaluation and classical P300 evaluation now exist as separate, verified
local artifacts. Counterfactual protocol v2 requires complete synthetic messages, fixed candidate
sets, distinct recorded selection trials, adapter provenance, and an explicit flash layout. Study
P provides occurrence-level stimulus event codes rather than the tile signatures of a NeuroSelect
display, so the pairing step must not imply that its candidate text or layout was observed.

The current development language run evaluates four complete four-span messages, while the P300
test artifact contains seven labeled selection trials. A partial message would violate the v2
message contract.

## Decision

Use a checksum-addressed preparation artifact between component evaluation and counterfactual
fusion. The builder:

- verifies both source manifests before reading their results;
- deterministically orders complete, successful language messages using a tracked seed;
- selects whole messages only and pairs their spans in order with distinct recorded P300 trials;
- preserves candidate sets, generic support, personalization lift, retrieval evidence, adapter
  identities, event order, event IDs, onsets, stimulus codes, probabilities, and recorded target
  codes;
- assigns the recorded target-code set to the first source-layout candidate and distributes all
  non-target event codes deterministically into balanced, non-empty signatures for the remaining
  candidates; and
- writes canonical input JSON plus a manifest containing the preparation recipe, source language
  manifest/result checksums, source decoder manifest/evaluation checksums, adapter checksums, and
  output checksum.

The fusion runner subsequently remaps the recorded target signature to the visible intended
candidate, or to `Other` when the intended phrase was not naturally generated. Candidate text is
never inserted by the preparation step.

The tracked development recipe selects one complete message because seven EEG trials cannot cover
two four-span messages. It runs A–F plus only the ablations supported by the current language
artifact. Irrelevant-retrieval and no-context ablations remain excluded because the source
evaluation did not record those alternate snapshots.

Development input is always non-claim-eligible. Research eligibility requires an unlimited
preparation recipe, a claim-eligible full language evaluation, and exact coverage of every
language and recorded EEG trial. The input limitations and eligibility propagate into the fusion
result.

## Consequences

Step 4 can run locally from the existing development language and P300 artifacts without
regenerating candidates or retraining either model. The current output exercises the complete
data path but does not establish real-time BCI performance, word-level participant intent, or
personalization benefit. The balanced event signatures are a deterministic counterfactual layout,
not an observed Study P or NeuroSelect screen.
