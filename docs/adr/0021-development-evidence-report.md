# ADR 0021: Development evidence report

- Status: Accepted
- Date: 2026-07-24

## Context

Steps 1–5 now produce three different kinds of local evidence: controlled simulation,
original-task Study P decoding, and offline counterfactual candidate replay. Their metrics answer
different questions and must not be pooled. In particular, the Step 5 complete-system value is
conditioned by natural candidate availability and comes from one four-span development message.

The release report recipe intentionally treats local EEG and counterfactual artifacts as optional
and points to the future research output path. It therefore does not provide a direct report over
the current development artifacts.

## Decision

Add a separate tracked Step 6 development report recipe requiring:

- the controlled held-out simulation artifact;
- the xDAWN original-task Study P evaluation; and
- the Step 5 development counterfactual evaluation.

Verify every source manifest and output checksum without loading executable model checkpoints.
Render each run as a separate evidence table with its own scope statement, source Git revision,
manifest checksum, claim eligibility, and limitations. Never compute a combined score.

Counterfactual tables must show target availability and recall conditional on availability beside
unconditional top-k and completion metrics. This prevents the 25% development result from being
presented without the fact that only one of four intended spans was naturally visible.

The report command writes canonical JSON, Markdown, and a checksum manifest, then reads them back
through the strict verifier. Dirty source artifacts remain visible and make `release_ready=false`;
this is expected for a development report and does not erase otherwise valid component results.

## Consequences

`make development-report` produces an auditable local summary at
`artifacts/reports/neuroselect-development-evidence-v1/`. The counterfactual table remains
non-claim-eligible, controlled simulation remains an engineering check, and xDAWN metrics retain
their original-task-only meaning.

The next stage is a research-grade evidence expansion, not reinterpretation of this report:
increase natural-candidate coverage, prepare enough recorded P300 trials for complete pairing, and
rerun the tracked unlimited protocols before considering release-grade counterfactual claims.
