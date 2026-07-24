# ADR 0020: Development counterfactual evaluation

- Status: Accepted
- Date: 2026-07-24

## Context

Step 4 emits a manifest-verified counterfactual input containing one complete four-span language
message paired with four distinct recorded P300 selection trials. The generic fusion command
previously allowed a caller to replace the specification embedded in that input. Doing so changed
the in-memory input digest, breaking the checksum identity of the prepared file. Result validation
also checked that every requested condition appeared, but did not require every condition to
contain the same logical trials or every expected comparison interval.

## Decision

Step 5 runs the tracked development fusion specification exactly as embedded in the Step 4 input.
The runner refuses a different configuration instead of silently rewriting the prepared input.
Manifest-backed inputs are verified before evaluation, and written fusion artifacts are read back
through the strict artifact reader before the command succeeds.

The result contract requires:

- one record for each logical profile/message/span trial under every requested condition;
- identical logical-trial coverage across the paired condition matrix;
- one mapping-provenance record per logical trial and distinct recorded source trials;
- overall metrics for every requested condition; and
- when condition F is present, both top-1 recall and selection-completion intervals for every
  other condition against F.

The tracked development matrix contains A–F plus uniform-neural, shuffled-neural, remove-RAG, and
shuffled-retrieval ablations. Irrelevant-retrieval and no-context ablations remain unavailable
because Step 3 did not record their alternate evidence snapshots.

## Consequences

`make counterfactual-evaluation` now provides the exact Step 5 path from the existing Step 4 input
to a checksum-verified development result. The result remains non-claim-eligible because its input
uses limited development language evidence and only one complete message. It demonstrates that
the paired offline mechanics execute; it does not demonstrate participant word intent, real-time
communication performance, or personalization benefit.

The next stage may consume this result as a separate counterfactual evidence table in a
development report. A research-grade evaluation still requires full language coverage and enough
recorded P300 selection trials to pair every complete message under an unlimited preparation
recipe.
