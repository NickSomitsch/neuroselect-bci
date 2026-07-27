# NeuroSelect P300 decoder model card

## Model and intended use

`xdawn-shrinkage-lda-platt-v1` is the first CPU baseline for the pinned bigP3BCI Study P data
layer. It classifies a preprocessed stimulus epoch as target or non-target and emits a calibrated
target probability. It is intended for reproducible offline research baselines and chronological
replay integration.

`eegnet-temperature-v1` is the compact neural comparator. It supports subject-independent
evaluation and the `eegnet-linear-head-temperature-v1` adaptation protocol. Both decoders solve
the original Study P target/non-target event task; neither decodes words or private thoughts.

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

The neural comparator uses training-only per-channel normalization, a temporal convolution,
depthwise spatial convolution, separable temporal convolution, and a binary linear head. It is
temperature-scaled on held validation subjects. Its tensor-only checkpoint is loaded in
weights-only mode.

For held-out subjects, the primary personalization protocol freezes the complete EEGNet feature
extractor. Earlier labeled SE001 trials fit only a replacement linear head; later labeled SE001
trials provide early stopping and temperature scaling. SE002 remains untouched until evaluation.
Insufficient SE001 calibration data triggers the subject-independent fallback and a conservative-
abstention flag. Reverse-session analysis is labeled as sensitivity analysis, not chronology.

The implementation rejects epoch, selection-trial, recording, and subject overlap across the
development/test boundary. Configuration, data collections, checkpoint, results, environment, and
device information are checksum-addressed in each run manifest.

## Downstream counterfactual aggregation

The offline fusion protocol converts calibrated flash probabilities into a distribution over one
fixed visible tile grid. It sums target/non-target log likelihoods for each tile's stimulus-code
signature and normalizes the scores with a configured softmax temperature. Counterfactual mapping
swaps tile signatures while preserving every source event, code, onset, and probability. If the
intended language candidate is absent, the recorded target maps to `Other`.

This downstream distribution is not a word decoder. A successful tile replay means the remapped
tile was ranked or completed under a counterfactual protocol; it does not mean the source
participant intended its displayed text. Original-task, counterfactual, and controlled-simulation
metrics remain separate.

## Metrics

Required original-task metrics are event-level AUROC, balanced accuracy, Brier score, negative log
likelihood, expected calibration error, exact target-event-set accuracy, target-event recall@K,
target-event average precision, and top-event hit rate. The legacy
`selection_code_set_accuracy` field is retained for artifact compatibility, but Study P codes in
the research evaluation identify occurrence-level events; the metric is not ordinary symbol,
character, or NeuroSelect candidate accuracy.

## Limitations and risks

- The research xDAWN evaluation uses the complete fixed 13/3/3 subject split. Only three subjects
  are held out, so subject-level and bootstrap comparisons remain descriptive.
- P300 responses and calibration can drift between sessions and subjects.
- Source condition/order may be confounded with session.
- Exact target-event-set accuracy is a stringent all-events metric and is not final character or
  NeuroSelect candidate accuracy.
- Joblib checkpoints are executable serialization and must come from a trusted local run.
- EEGNet training can vary across hardware/runtime versions despite fixed seeds; manifests record
  PyTorch and device versions and comparisons must not mix environments silently.
- Subject adaptation estimates can be unstable with few selection trials and must never expand to
  full-layer fine-tuning under this protocol.
- Counterfactual fusion has been evaluated over a balanced 144-trial subset from the three held-out
  subjects, but the displayed text was not selected by the source participants and cannot support
  live-use or clinical claims.
- The model does not provide automatic selection; downstream use must retain repeat, abstention,
  and explicit-confirmation safeguards.

## Reproduction

After preparing all required subject artifacts, run `make p300-research-baseline` for the
classical research model, `make p300-eegnet-research` for the original-task publication
comparator, or `make p300-eegnet` for EEGNet plus chronological adaptation. Use
`make p300-replay P300_REPLAY_ARGS="<epoch-directory> --checkpoint <run-directory>"` for a
virtual-clock JSONL replay with either checkpoint type. No dataset or model artifact is downloaded
during CI. Use `make counterfactual-fusion COUNTERFACTUAL_FUSION_ARGS="--input
<prepared-input.json>"` only after preparing candidate-aligned flash probabilities and explicit
language, adapter, and retrieval evidence.
