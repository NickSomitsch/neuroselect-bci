# NeuroSelect

NeuroSelect is a research project for reducing the number of non-invasive BCI selections needed to compose a message through safe, personalized language prediction.

It is not a mind-reading system, an open-vocabulary EEG decoder, a medical device, or a diagnostic tool. Language-model suggestions are kept separate from neural evidence, and a message cannot be finalized without explicit confirmation.

## Current status

The repository currently contains the implementation foundation and the first public
synthetic evaluation assets:

- Python 3.12 and React/TypeScript project scaffolding.
- Typed domain contracts for candidates, neural evidence, sessions, and run manifests.
- A deterministic session-state transition model.
- Four explicitly synthetic personas with permissioned, expiring knowledge records.
- A deterministic held-out message benchmark with 5,600 generated messages.
- A seeded, call-order-independent neural probability simulator.
- A structured deterministic candidate generator with deduplication, explicit controls, and an
  application-owned versioned lexical risk policy.
- A local SQLite personal-knowledge store with safe lexical retrieval and provenance.
- A transparent fusion ranker with LM-dominance, repeat, abstention, and risk safeguards.
- A loopback session API with manual/simulated rounds and explicit confirmation boundaries.
- An accessible local web interface with candidate provenance, configurable round controls, and
  exact-text final confirmation.
- A deterministic held-out simulation experiment matrix with paired baselines, safety ablations,
  calibration and communication metrics, and checksum-addressed JSON artifacts.
- A pinned bigP3BCI Study P inventory/import layer with MNE preprocessing, artifact reports,
  standardized FIF/NumPy artifacts, and subject/session leakage controls.
- A CPU xDAWN/shrinkage-LDA baseline with held-subject Platt calibration, original-task metrics,
  checksum-addressed checkpoints, and deterministic virtual-clock P300 replay.
- A compact CPU/MPS EEGNet with held-subject temperature scaling, frozen-feature subject adapters,
  weights-only checkpoints, and strictly chronological SE001-to-SE002 drift reports.
- A checksum-addressed flash-to-tile aggregator and paired offline counterfactual A–F fusion
  runner with dependency-gated personalization, safety ablations, and hierarchical intervals.
- Locked research-scope, dataset, and local-first architecture decisions.
- Linting, type checking, tests, and continuous integration.

No model or EEG dataset is downloaded by setup. The interface is an operational simulated/manual
research demo, not a live or prerecorded-EEG BCI demo.

## Development

Prerequisites are `uv`, Node.js 22, and `pnpm` 10.

```bash
make setup
make verify
```

Run the simulated interface in two terminals with:

```bash
make api
```

```bash
pnpm --dir ui dev
```

Then open `http://127.0.0.1:5173`. The interface supports full keyboard navigation, large
candidate targets, 4/6/8/12-target rounds, adjustable focus-scan timing, high contrast, reduced
motion, visible evidence provenance, explicit rejection and correction controls, and one-time
final-message confirmation. The development server proxies only the local API routes to
`127.0.0.1:8000`.

Generate the benchmark JSONL files and checksum manifest under the ignored artifact
directory with:

```bash
make synthetic-data
```

Inspect the deterministic candidate contract without changing any message state:

```bash
uv run python scripts/generate_candidates.py "Could you"
```

Create a reproducible local knowledge store from the 20 tracked synthetic records:

```bash
make synthetic-knowledge
```

Personal records are profile-scoped, revisioned, permissioned, and physically deletable.
Disabled, expired, not-yet-valid, and prompt-injection-flagged records cannot influence
suggestions. Retrieved records retain their source and matched terms for visible explanations.

Run the current CPU-only candidate → RAG → simulated-neural → fusion slice with:

```bash
make fusion-smoke
```

The ranker records every normalized input and weighted contribution. A displayed top candidate
is still provisional: automatic selection is forbidden and explicit confirmation remains
required.

Run the tracked held-out simulated experiment matrix with:

```bash
make simulated-evaluation
```

This writes ignored `trials.jsonl`, `metrics.json`, and `manifest.json` artifacts under
`artifacts/evaluation/simulated-vertical-slice-v2/`. The controlled protocol keeps the intended
span visible to isolate fusion behavior; it does not measure unconstrained language-generation
recall or real EEG performance. Planned LoRA and complete calibrated-system conditions are listed
but cannot run until those components have real held-out implementations. Full conversation-
context removal is likewise dependency-gated until a context-sensitive language backend exists;
the runnable proxy removes confirmed context from retrieval queries and labels that scope exactly.

Review the Study P dataset card and explicitly prepare a one-recording smoke slice with:

```bash
make study-p-data STUDY_P_ARGS="--download --accept-license --subjects P_01 --source-partitions train --limit-files 1"
```

The command verifies the pinned PhysioNet checksum inventory and every EDF before use, preserves
raw files under ignored `data/raw/`, and emits EEG-only MNE FIF plus decoder-ready epochs and typed
checksum sidecars under ignored `data/processed/`. Remove `--limit-files` to process every block
for the named subject. Dataset-author `Train`/`Test` folder labels are never treated as model
splits; the tracked primary split holds out entire subjects. SE001-to-SE002 is the primary
chronological drift fold; its reverse is retained only as separately labeled condition/order
sensitivity analysis. Setup and CI remain offline.

After preparing recordings for all subjects in the tracked split, train and evaluate the CPU
classical baseline with:

```bash
make p300-baseline
```

The xDAWN/LDA feature model is fitted only on training subjects; its Platt calibrator is fitted
only on validation subjects. Unknown-label source blocks receive predictions for replay but never
enter supervised metrics. Runs write ignored checkpoint, evaluation, and environment-provenance
artifacts under `artifacts/models/p300-xdawn-lda-v1/`.
Epoch artifacts prepared before schema 1.1 must first be regenerated with `make study-p-data` and
`--overwrite` so exact source onset and stimulus-code metadata are available.

Train the compact EEGNet comparator, its held-subject temperature scaler, and per-test-subject
SE001 linear-head adapters with:

```bash
make p300-eegnet
```

This writes a tensor-only checkpoint plus original-task and chronological SE002 drift reports
under `artifacts/models/p300-eegnet-v1/`. Feature layers are hash-checked before and after every
adaptation. Subjects without enough labeled SE001 trials retain the subject-independent decoder
and are marked for conservative abstention.

Replay one prepared recording on a deterministic virtual clock with:

```bash
make p300-replay P300_REPLAY_ARGS="<epoch-directory> --checkpoint <run-directory> --speed 2"
```

Replay preserves chronological source timestamps and unknown labels and supports programmatic
pause, seek, reset, and speed changes. It is offline replay, not live EEG acquisition.

Run the paired counterfactual fusion matrix from an explicitly prepared JSON input with:

```bash
make counterfactual-fusion COUNTERFACTUAL_FUSION_ARGS="--input <prepared-input.json>"
```

The runner preserves the source flash stream and changes only which visible tile occupies the
recorded target signature. It writes result JSON, trial/mapping JSONL, metric/interval CSV, and a
checksum manifest under `artifacts/evaluation/counterfactual-fusion-v1/`. Conditions D–F require
an adapter ID, adapter checksum, and held-out personalization evidence. Controlled fixtures can
exercise those mechanics but are always marked non-claim-eligible. No real counterfactual result
is bundled because Study P data and trained artifacts remain local and no language-model LoRA is
implemented in this checkout.

Run the local research API at the configured loopback address with:

```bash
make api
```

The versioned `/api/v1` routes expose public synthetic profile summaries and support session
creation, configurable candidate rounds, select/reject/repeat, back, clear, other, cancel, manual
debug text, enhanced selection confirmation, and one-time final-message confirmation. Simulator
ground truth and confirmation secrets are not exposed in ordinary session views.

The tracked recipe emits exactly 1,000 training, 150 validation, and 250 test messages
for each of four personas. Templates and topics are disjoint across splits, every target
span is at most four whitespace-delimited tokens, and identical source files produce
byte-identical artifacts regardless of checkout location.

Research artifacts, downloaded datasets, and model weights belong under the ignored `artifacts/`,
`data/`, or `models/` directories. Every experiment must emit a machine-readable run manifest
containing immutable data/model/policy identifiers and checksums, package versions, and device
provenance.

## Research boundaries

The first real-data integration will use offline P300 replay. Recorded target evidence may be mapped to visible candidate tiles for counterfactual system evaluation, but the resulting words were not selected live by the original participants. Original-task decoder results, counterfactual replay results, and controlled simulation results must always be reported separately.

See [the research-scope ADR](docs/adr/0001-research-scope-and-claims.md), [the dataset ADR](docs/adr/0002-p300-and-study-p.md), [the Study P dataset card](docs/dataset-card.md), [the P300 model card](docs/model-card.md), [the synthetic benchmark ADR](docs/adr/0004-synthetic-benchmark-and-simulation.md), [the candidate-generation ADR](docs/adr/0005-structured-candidate-generation.md), [the personal-retrieval ADR](docs/adr/0006-local-personal-retrieval.md), [the transparent-fusion ADR](docs/adr/0007-transparent-fusion-and-abstention.md), [the session API ADR](docs/adr/0008-explicit-confirmation-session-api.md), [the accessible-interface ADR](docs/adr/0009-accessible-local-research-interface.md), [the controlled-evaluation ADR](docs/adr/0010-controlled-simulated-evaluation.md), [the pinned Study P data-layer ADR](docs/adr/0011-pinned-study-p-data-layer.md), [the classical decoder/replay ADR](docs/adr/0012-classical-p300-decoder-and-replay.md), [the EEGNet/adaptation ADR](docs/adr/0013-eegnet-adaptation-and-session-drift.md), and [the counterfactual-fusion ADR](docs/adr/0014-counterfactual-fusion-evaluation.md) before contributing claims or experiment code.

## License

Source code is licensed under the MIT License. Datasets, model weights, and derived artifacts retain their own licenses and must be cited separately.
