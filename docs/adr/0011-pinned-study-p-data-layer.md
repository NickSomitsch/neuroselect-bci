# ADR 0011: Pinned Study P data and preprocessing layer

- Status: accepted
- Date: 2026-07-18

## Context

The first real-EEG phase needs a reproducible route from a large public EDF+ release to
decoder-ready P300 epochs. Dataset-author `Train` and `Test` directories describe spelling blocks;
they are not safe model-evaluation partitions. Raw data must remain outside Git and setup/CI must
stay offline.

## Decision

- Pin bigP3BCI v1.0.0 by DOI `10.13026/0byy-ry86` and pin the official `SHA256SUMS.txt` file by
  SHA-256 `75ce052ae8626a73b43887c994c4c0d17e5b0d775ad3083f759af20028e32fbb`.
- Require explicit `--download --accept-license` flags and an explicit subject list. Verify every
  EDF against the official inventory before MNE reads it.
- Preserve raw sources under ignored `data/raw/`; never modify them. Standardize each recording as
  EEG-only MNE FIF plus a typed JSON sidecar and checksums under ignored `data/processed/`.
- Extract flash onsets from `StimulusBegin`, labels from `StimulusType`, optional stimulus/target
  fields from their source channels, and enclosing selection-trial groups from
  `PhaseInSequence` segments.
- Preserve a zero-only `StimulusType` stream as `unknown`, never as a set of negative examples.
  Require author-labeled Train blocks to contain both target and non-target events; retain
  unlabeled Test blocks for replay without admitting them to supervised training or scoring.
- Apply the pinned CPU recipe: 0.5–20 Hz band-pass, 60 Hz notch, average reference, -0.1 to 0.8 s
  epochs, pre-stimulus baseline, peak-to-peak/flat rejection, and 128 Hz output.
- Use a deterministic 13/3/3 subject-separated train/validation/test split as the primary model
  evaluation. Retain both SE001→SE002 and SE002→SE001 folds for within-subject drift evaluation.
- Reject any split where an epoch, selection trial, recording, or—when required—subject crosses a
  boundary.

## Consequences

The data layer adds NumPy and MNE but no GPU dependency. Tests use deterministic in-memory MNE
recordings and local byte fixtures; they do not download or redistribute EEG. Study P's sessions
counterbalance predictive and non-predictive spelling conditions, so session-drift results must
report the condition order and cannot be interpreted as pure temporal drift.

This step prepares labels and tensors only. It does not establish decoder accuracy, calibration,
or replay performance; those belong to the next implementation step.
