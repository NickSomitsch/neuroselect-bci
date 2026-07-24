# ADR 0022: Research evidence expansion readiness

## Status

Superseded by ADR 0023

## Context

The Step 6 development report correctly separates controlled simulation, original-task EEG, and
counterfactual replay, but its counterfactual result contains only one four-span language message
paired with four of seven available P300 selections. Moving to research evidence requires exact
tracked protocols and a capacity check before expensive Qwen inference, adapter training, or Study
P preprocessing begins.

The complete synthetic test split contains 1,000 held-out messages and 3,990 target spans. A
research counterfactual recipe that maps every span to a unique recorded P300 selection would
therefore require 3,990 labeled selections. The pinned Study P inventory contains 30 labeled
`Train` recordings across the three held-out subjects. At seven character selections per
recording, its nominal capacity is 210 selections before artifact rejection. Downloading the
remaining files cannot satisfy full one-to-one coverage.

## Decision

Add a fail-closed Step 7 readiness artifact governed by
`configs/experiments/research_evidence_expansion.yaml`. The checker derives message and span demand
from the benchmark sources and verifies:

1. an unlimited research-tier language protocol;
2. the non-development LoRA training recipe with validation and test evaluation;
3. one checksum-valid research adapter for every synthetic profile;
4. a complete, claim-eligible, clean held-out language artifact;
5. a checksum-verified, clean original-task decoder evaluation covering all required held-out EEG
   subjects;
6. enough distinct labeled P300 selections to map every included language span;
7. an unlimited research input-preparation protocol; and
8. exactly the primary A–F fusion conditions.

The default checker exits unsuccessfully when any condition is unmet. `--allow-incomplete` is an
explicit audit mode: it writes the same canonical JSON and reports every blocker but returns
success for local inspection.

Research input eligibility requires every included language trial to receive a distinct P300
trial. It no longer requires consuming every available EEG trial; unused recorded trials are
allowed and remain visible through source counts. This permits a future smaller language sample
without weakening one-to-one mapping.

Tracked Make targets provide exact commands for training one research adapter, running full
language evaluation, preparing research input, and evaluating the primary A–F matrix. No aggregate
target automatically starts these expensive jobs.

## Consequences

Step 7 is validated when the checker accurately reports both satisfied protocol requirements and
missing evidence. It does not make the current system research-ready. With the current local
artifacts it reports 1,000 messages, 3,990 required selections, seven available selections,
zero of four research adapters, a missing full language result, and missing EEG coverage for
`P_11` and `P_13`.

The next protocol step must preregister a deterministic, subject-balanced counterfactual sample
bounded by the usable held-out P300 selections. The complete 3,990-span language evaluation remains
separate component evidence. EEG selections must not be duplicated to inflate the counterfactual
sample, and omitted language spans must be reported rather than hidden.
