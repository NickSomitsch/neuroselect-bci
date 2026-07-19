# NeuroSelect classical P300 baseline model card

## Model and intended use

`xdawn-shrinkage-lda-platt-v1` is the first CPU baseline for the pinned bigP3BCI Study P data
layer. It classifies a preprocessed stimulus epoch as target or non-target and emits a calibrated
target probability. It is intended for reproducible offline research baselines and chronological
replay integration.

It is not a medical device, diagnostic model, imagined-speech decoder, open-vocabulary decoder,
or live communication system. A target probability describes compatibility with the Study P P300
selection task; it does not establish a private thought or authorize generated text.

## Architecture and data boundary

- Input: artifact-screened 32-channel epochs from −0.1 to 0.8 seconds, resampled to 128 Hz.
- Features: two xDAWN components per class, flattened across epoch time.
- Classifier: shrinkage LDA (`lsqr`, automatic shrinkage).
- Calibration: logistic/Platt scaling on held validation subjects.
- Primary split: 13 training, three validation, and three test subjects from the tracked manifest.
- Unknown source events are prediction/replay-only and are never used as supervised labels.

The implementation rejects epoch, selection-trial, recording, and subject overlap across the
development/test boundary. Configuration, data collections, checkpoint, results, environment, and
device information are checksum-addressed in each run manifest.

## Metrics

Required original-task metrics are event-level AUROC, balanced accuracy, Brier score, negative log
likelihood, expected calibration error, and selection-trial target-code-set accuracy. Predictive
and non-predictive conditions and chronological sessions must be reported separately when the full
dataset evaluation is run.

## Limitations and risks

- The baseline has not yet been evaluated across all 19 Study P subjects in this checkout.
- P300 responses and calibration can drift between sessions and subjects.
- Source condition/order may be confounded with session.
- Stimulus-code accuracy is not final character or NeuroSelect candidate accuracy.
- Joblib checkpoints are executable serialization and must come from a trusted local run.
- The model does not provide automatic selection; downstream use must retain repeat, abstention,
  and explicit-confirmation safeguards.

## Reproduction

After preparing all required subject artifacts, run `make p300-baseline`. Use
`make p300-replay P300_REPLAY_ARGS="<epoch-directory> --checkpoint <run-directory>"` for a
virtual-clock JSONL replay. No dataset or model artifact is downloaded during setup or CI.

