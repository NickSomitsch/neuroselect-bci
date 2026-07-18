# Dataset card: bigP3BCI Study P

## Dataset and intended use

NeuroSelect pins Study P from **bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based
Brain-Computer Interface Dataset**, version 1.0.0. The version DOI is
[`10.13026/0byy-ry86`](https://doi.org/10.13026/0byy-ry86), and the files are published by
[PhysioNet](https://physionet.org/content/bigp3bci/1.0.0/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Study P contains 228 EDF+ spelling-block recordings from 19 subjects, two sessions per subject,
32 EEG channels, and 256 Hz sampling. It compares predictive and non-predictive 9×8 P300 spelling.
NeuroSelect uses it for original-task target/non-target decoding and later offline replay. It must
not be described as open-vocabulary thought decoding, a clinical validation, or a live selection
study.

## Immutable source identity

The source release remains outside Git under `data/raw/bigp3bci/1.0.0/`. The official
`SHA256SUMS.txt` inventory is pinned to:

```text
75ce052ae8626a73b43887c994c4c0d17e5b0d775ad3083f759af20028e32fbb
```

Every selected EDF is verified against that inventory before it is opened. The full release is
44.6 GB uncompressed, so the preparation command downloads only explicitly named Study P
subjects. Setup and tests never download data.

## Participant metadata and discrepancy

On 2026-07-18, NeuroSelect read the fixed patient-header range of one inventory-listed EDF for
each of P_01 through P_19. All 19 headers contain `NonALS`. The original Study P publication
describes able-bodied participants, and MOABB's machine-readable configuration currently sets
`has_als: false`.

There is a secondary-source defect: MOABB's generated overview and class docstring say “19 ALS
subjects,” while its participant panel says “healthy.” This card preserves only the source EDF
label `NonALS`; that label is not evidence that every participant was healthy, nor should it be
used to infer clinical suitability. Demographic fields are not copied into decoder artifacts.

## Source fields and standardized format

Each EDF is one spelling block. NeuroSelect retains:

- subject (`P_01`–`P_19`), session (`SE001`/`SE002`), run, spelling condition, and the author's
  block label;
- EEG channel name, type, unit, and standard-10/20 position;
- flash onset from `StimulusBegin` and target/non-target label from `StimulusType`;
- `StimulusCode`, `CurrentTarget`, and `SelectedTarget` when available;
- an enclosing selection-trial ID derived from `PhaseInSequence` segments;
- source URL, source SHA-256, inventory identity, DOI, version, and license.

Standardized recordings are unfiltered EEG-only MNE FIF files with typed JSON metadata and an
output checksum sidecar. Decoder-ready epochs are compressed NumPy arrays with separate typed
epoch provenance, preprocessing configuration, artifact-rejection report, and checksums.

## Preprocessing

The versioned CPU recipe applies:

1. 60 Hz notch filtering and 0.5–20 Hz band-pass filtering.
2. Average EEG reference.
3. Epoching from -0.1 through 0.8 seconds around each flash.
4. Baseline correction over -0.1 through 0.0 seconds.
5. Rejection above 150 µV peak-to-peak or below 0.5 µV peak-to-peak, plus MNE annotation-based
   rejection.
6. Resampling accepted epochs to 128 Hz.

Rejected epochs remain represented in the preprocessing report with their source event ID and
reason. Production imports require exactly 32 EEG channels at 256 Hz; labeled blocks must contain
both target classes.

The real-file smoke audit found that at least one author Test block has a zero-only
`StimulusType` stream. NeuroSelect marks every event in such a block `unknown` (`-1`) rather than
silently treating all flashes as non-target. Author Train blocks are required to contain both
classes. Unknown-label epochs may support timed offline replay, but they are excluded from
supervised decoder fitting and accuracy/calibration denominators.

## Splits and leakage controls

The primary split holds out entire subjects: 13 train, 3 validation, and 3 test subjects, fixed by
seed `20260718`. No epoch, character-selection trial, recording, or subject may cross partitions.
The exact IDs are tracked in
[`configs/datasets/bigp3bci_study_p.yaml`](../configs/datasets/bigp3bci_study_p.yaml).

The dataset's `Train` and `Test` folder names are source task/block and label-availability fields;
they are explicitly not NeuroSelect model-evaluation splits. Supervised samples come from labeled
blocks inside the subject-separated partitions. Two cross-session folds support drift evaluation,
but Study P counterbalances predictive/non-predictive condition order between subjects. Session
findings must therefore report condition order and cannot be attributed solely to time or
electrode drift.

## Preparation

Review CC BY 4.0, then prepare a small subject slice:

```bash
uv run python scripts/prepare_study_p.py \
  --download --accept-license --subjects P_01 --source-partitions train --limit-files 1
```

Remove `--limit-files` to prepare both sessions and all blocks for the selected subject. Add more
subject IDs explicitly for a larger import. The command writes only ignored `data/` paths.

The Step 9 real-file smoke verified P_01/SE001 against the official inventory and round-tripped
both artifact formats. PredictiveSpelling Train01 contained 840 labeled flashes across 7 selection
trials; 725 epochs passed the pinned rejection recipe and retained both numeric classes. Test07
contained 2,232 unlabeled flashes across 31 selection trials; 2,021 epochs passed and every output
label remained `-1`. These are ingestion checks, not decoder-performance results.

## Limitations

- Target/non-target imbalance varies by block and subject.
- Offline replay can preserve evidence and timing, but mapping recorded flashes to new candidate
  words is counterfactual; the original participant did not select those words.
- Artifact thresholds are a transparent baseline, not a claim of optimal physiological cleaning.
- Subject-level generalization, person-specific adaptation, and session drift require separate
  reporting.
- This data layer provides no accuracy, calibration, speed, safety, or clinical-performance claim.
