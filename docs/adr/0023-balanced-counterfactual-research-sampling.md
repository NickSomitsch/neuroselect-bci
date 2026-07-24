# ADR 0023: Balanced counterfactual research sampling

## Status

Accepted

## Context

ADR 0022 established that the complete held-out language benchmark contains 1,000 messages and
3,990 target spans, while the three held-out Study P subjects provide a nominal maximum of 210
labeled character selections before preprocessing rejection. Requiring one distinct P300
selection for every language span is infeasible. Reusing selections would create pseudoreplication,
and silently reducing the language evaluation would hide candidate-generation failures.

The language component and offline counterfactual replay answer different questions and may use
different, explicitly reported denominators. A counterfactual sample must remain large enough to
cover every held-out EEG subject and synthetic profile symmetrically while preserving complete
messages.

## Decision

Retain the unlimited language evaluation over all 1,000 messages and 3,990 spans as separate
component evidence. For counterfactual replay, preregister a fixed 4 × 3 factorial sample:

- four synthetic profiles;
- three held-out EEG subjects (`P_02`, `P_11`, and `P_13`);
- three complete messages in every profile/subject cell; and
- exactly four target spans in every sampled message.

The resulting counterfactual input contains 144 trials: 36 per profile, 48 per EEG subject, and
12 per profile/subject cell. All four spans from one message map to distinct selections from the
same EEG subject. No recorded selection may appear twice.

Within each profile, messages are ordered by SHA-256 over the fixed seed, profile ID, and message
ID. Only complete, successful four-span messages are eligible. Within each EEG subject, usable
selection trials are ordered by SHA-256 over the fixed seed, subject, session, and source selection
ID. These rules make the sample independent of filesystem order and byte-stable for identical
inputs.

The builder fails unless every profile supplies nine eligible messages and every held-out EEG
subject supplies at least 48 usable selections. Its manifest records the sampling revision plus
exact profile and EEG-subject trial counts. The artifact reader recomputes those counts and rejects
imbalanced or inconsistent v2 manifests.

The readiness gate separately requires:

1. the complete research language result and four full research adapters;
2. checksum-verified decoder metadata covering all 13 training subjects and all three validation
   subjects;
3. held-out evaluation data from all three test subjects;
4. at least 48 usable selections from every test subject; and
5. the primary A–F fusion protocol with 2,000 hierarchical bootstrap resamples.

## Consequences

The protocol is feasible within the pinned Study P inventory while retaining exact one-to-one
mapping inside the sampled counterfactual dataset. The full language and 144-trial counterfactual
denominators must always be reported separately.

The fixed sample supports a reproducible offline benchmark and paired descriptive comparisons. It
does not establish population-level or clinical generalization: there are only three held-out EEG
subjects, all source selections came from the original Study P task, and no participant selected
the remapped language candidates live.

Step 9 must prepare the complete subject-split Study P data and retrain the original-task decoder.
Only after its clean metadata contains all training and validation subjects and its evaluation
contains at least 48 usable selections per held-out subject can the balanced counterfactual input
be built.
