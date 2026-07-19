# ADR 0012: Classical calibrated P300 decoder and offline replay

- Status: Accepted
- Date: 2026-07-19

## Context

The Study P layer now produces checksum-addressed, decoder-ready epochs with subject, session,
recording, selection-trial, event, and label provenance. The first real-data decoder must establish
a reproducible CPU baseline without treating the dataset author's Train/Test directory names as
model splits or converting unlabeled test blocks into non-target examples.

## Decision

Use a deterministic xDAWN spatial filter followed by shrinkage linear discriminant analysis.
Fit xDAWN and LDA only on the 13 training subjects in the tracked subject split. Fit a one-variable
logistic (Platt) calibrator only to the base decoder's scores for the three validation subjects.
Do not use test-subject epochs for feature fitting, covariance estimation, calibration, threshold
selection, or model selection.

Unknown (`-1`) events are excluded before both fitting and supervised metric calculation. The
decoder may still produce a probability for an unknown event so that an originally unlabeled block
can be replayed, but no accuracy or calibration claim is derived from that event. Training,
calibration, and evaluation reject overlapping epoch, selection-trial, recording, and—under the
primary protocol—subject identifiers.

Report labeled-event AUROC, balanced accuracy, Brier score, negative log likelihood, and expected
calibration error. Also aggregate repeated flashes by stimulus code within each labeled selection
trial and report exact target-code-set accuracy. This is an original-task row/column-code metric,
not open-vocabulary word accuracy and not evidence that a participant selected a NeuroSelect
candidate.

Checkpoints use joblib because the fitted MNE/scikit-learn pipeline is not represented by a stable
portable tensor format. NeuroSelect verifies the checkpoint and JSON result hashes before loading,
but joblib is executable serialization: only locally produced, trusted checkpoints may be opened.
Manifests record Git and dirty-tree identity, dataset/config/output hashes, package versions, and
device provenance.

Provide a pull-based virtual replay clock over one prepared recording. Replay preserves source
event order, onset time, label availability, and epoch provenance; supports pause, seek, reset, and
speed changes; and can attach decoder probabilities. It deliberately does not sleep, fabricate
real-time acquisition, open a network stream, or claim to be a live BCI adapter.

Epoch artifact schema 1.1 adds source onset seconds and stimulus/target codes required for replay
and selection-code aggregation. Schema 1.0 epoch artifacts must be regenerated from their pinned,
checksum-verified standardized recordings; they are not silently assigned reconstructed metadata.

## Consequences

The baseline is CPU-compatible, interpretable, calibrated without test leakage, and suitable for
offline original-task evaluation and deterministic integration tests. It does not implement
EEGNet, subject-specific adaptation, chronological session-drift analysis, counterfactual
candidate mapping, or live hardware. Those remain later plan steps.
