# ADR 0024: Full Study P research evidence

## Status

Accepted

## Context

The development xDAWN/LDA artifact was trained from one recording for `P_01`, calibrated from one
recording for `P_06`, and evaluated from one recording for `P_02`. It establishes that the
pipeline runs, but it does not cover the fixed Study P subject split or the 48 selections per
held-out subject required by ADR 0023.

Study P contains ten labeled `Train` EDF files per participant: five in `SE001` and five in
`SE002`. Across 19 participants, complete research preparation therefore requires 190 recordings.
Serial downloading makes the exact workflow unnecessarily slow, but parallelism must not weaken
source verification.

## Decision

Add a tracked Step 9 evidence recipe requiring:

- all 13 fixed training subjects;
- all 3 fixed validation/calibration subjects;
- all 3 held-out test subjects;
- both sessions and exactly 5 labeled recordings per subject/session;
- at least 48 usable labeled, timed selection trials for each test subject; and
- the pinned xDAWN, shrinkage-LDA, and Platt-calibration configuration.

Permit one to sixteen download workers against PhysioNet's documented public AWS mirror. Every
worker writes to a temporary file in the final directory, streams the official SHA-256 digest,
and atomically replaces the destination only after verification. Existing EDFs are reused only
after their checksum passes.

Write the research decoder under `artifacts/models/p300-xdawn-lda-research-v1/`, separate from the
development artifact. Before training, require the complete data audit. After training, verify
the decoder metadata and evaluation JSON checksums without loading the joblib checkpoint. Require
exact train/calibration/test subject sets and retain the clean-worktree policy.

## Consequences

The full data layer occupies only a few gigabytes but preprocessing and classical model fitting
remain explicit local jobs. Interrupted downloads are recoverable: incomplete temporary files are
removed and verified completed EDFs are retained.

A decoder produced while tracked source changes are uncommitted can validate functionality but is
not clean research evidence. It must be regenerated from the same prepared data after the
implementation commit.

Passing Step 9 establishes complete original-task EEG evidence and enough distinct selections for
the balanced offline replay sample. It does not create the four research language adapters or the
complete 3,990-span language result; those are Step 10 and Step 11.
