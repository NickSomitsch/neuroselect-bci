# ADR 0013: EEGNet, frozen-feature personalization, and chronological drift

- Status: Accepted
- Date: 2026-07-20

## Context

The classical baseline establishes a reproducible original-task floor, but the research plan also
requires a compact neural baseline and a narrowly scoped form of user adaptation. Adaptation must
not turn the held-out second session into training data or silently fine-tune the full decoder.
Study P sessions are condition-counterbalanced, so condition/order effects can be mistaken for
general longitudinal drift unless the primary chronology and reverse sensitivity are separated.

## Decision

Implement a compact in-repository EEGNet with temporal convolution, depthwise spatial filtering,
separable temporal convolution, and a binary linear head. Per-channel normalization is fitted only
on training-subject epochs. The feature model and head are trained on the tracked training
subjects, selected using validation-subject loss, and temperature-scaled using validation-subject
logits. Unknown epochs never enter normalization, optimization, calibration, or metrics.

The primary personalization protocol is strictly chronological:

1. Start with the subject-independent checkpoint, which has never seen the held-out subject.
2. Use labeled SE001 selection trials only. The earlier 70% of trials fit the linear head; the
   remaining trials fit the temperature and provide early-stopping loss.
3. Freeze normalization, temporal, spatial, and separable-convolution parameters and batch-normal
   state. Record hashes of every feature-layer tensor before and after adaptation.
4. Evaluate on SE002 only after adaptation. No SE002 tensor, label, prediction, or summary is
   passed into the adaptation function.
5. If there are too few labeled SE001 trials, retain the subject-independent decoder and mark the
   result as requiring conservative abstention. Do not partially fit an under-supported adapter.

SE002→SE001 may be run later as a condition/order sensitivity analysis, but it is not pooled with
or described as primary chronological drift. Per-subject results retain event-level AUROC,
balanced accuracy, Brier score, negative log likelihood, expected calibration error, and
selection-code accuracy before and after adaptation. Aggregate deltas are descriptive; they are
not evidence of benefit until run on the pinned real dataset with uncertainty estimates.

Training selects MPS when available and configured as `auto`; CPU is a supported deterministic
path and is used by tests. Checkpoints contain tensor state dictionaries loaded with PyTorch's
weights-only mode. JSON metadata, evaluation, drift report, dataset/config hashes, environment,
device, and all output checksums remain independently inspectable.

## Consequences

This implements a reproducible neural comparator and prevents full-model or future-session
leakage during personalization. It does not establish that adaptation improves Study P results;
the real dataset is not present in this checkout. It also does not perform tile aggregation,
neural-language fusion, counterfactual candidate mapping, statistical comparison, or WebSocket
replay control. Those belong to the next experiment/fusion step.
