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
- An optional revision-pinned Qwen/MLX backend that replaces model-reported support with complete
  phrase log-likelihood scoring, plus checksum-verified LoRA adapter loading.
- Leakage-separated synthetic style corpora, a local MLX QLoRA training wrapper, and
  candidate-aligned style/RAG evidence with a clearly non-claim-eligible controlled proxy.
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
- A verified, evidence-separated research-report builder with deterministic paired intervals and
  an automated release-readiness gate.
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

Run the complete Step 2 mechanics without downloading a model:

```bash
make language-smoke
```

This keeps the visible fixture candidates fixed and emits separate generic-language,
controlled-style, and synthetic-RAG signals. The style proxy is explicitly marked
`controlled_fixture` and cannot support a LoRA-benefit claim.

Prepare split-safe MLX completion corpora for all four synthetic profiles:

```bash
make language-personalization-data
```

To use the real local backend on Apple silicon, install the optional dependency, then explicitly
allow the first download of the pinned Qwen revision:

```bash
make local-language-sync
uv run --extra local-language python scripts/run_personalized_language.py \
  "I would like" --backend mlx --download
```

Train one QLoRA adapter from a prepared profile corpus:

```bash
make language-lora LANGUAGE_LORA_ARGS="\
  --corpus artifacts/language/personalization-v1/synthetic-concise \
  --output artifacts/models/language-lora/synthetic-concise \
  --download"
```

Training validates every corpus checksum, masks the prompt loss, evaluates the held-out synthetic
test split, and writes a checksum-addressed adapter manifest. Model weights, corpora, and adapters
remain ignored local artifacts; setup and CI never download them.

Run the Step 3 held-out natural-candidate development evaluation with:

```bash
make language-evaluation
```

The tracked development recipe selects one test message per synthetic profile, verifies every
corpus and adapter checksum, and generates candidates without inserting the intended span. For
non-empty contexts, Qwen selects from compact noun, deadline, or ending vocabularies derived
exclusively from train and validation messages; empty-message generation remains unconstrained.
The development display contains nine language candidates plus the three application controls.
The vocabulary checksum and generic-versus-personalized availability and ranking artifacts are
written under
`artifacts/evaluation/held-out-language-personalization-dev-v1/`. It expects adapters named
`<profile-id>-dev-v1` under `artifacts/models/language-lora/`. Limited runs and development
adapters are always non-claim-eligible. The artifact also reports narrowly repaired outer-JSON
closers separately from unrecoverable candidate-generation failures. Target availability means
that the exact held-out next phrase appeared as one of the visible language candidates; an absent
target cannot receive a generic or personalized rank.

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
recall or real EEG performance. LoRA and complete calibrated-system conditions remain unavailable
in this fixture recipe because it does not load a locally trained adapter or real decoder
artifacts. Full conversation-context removal remains dependency-gated in this recipe; its
retrieval-context proxy removes confirmed context only from retrieval queries and labels that
scope exactly.

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

Build the Step 4 development input from the current verified language and P300 artifacts with:

```bash
make counterfactual-input
```

This deterministically pairs one complete four-span language message with four distinct recorded
P300 selection trials and writes canonical input plus a checksum manifest under
`artifacts/evaluation/counterfactual-input-development-v1/`. It does not regenerate candidates,
load Qwen, or retrain an adapter or decoder. The development recipe uses one whole message because
the current seven labeled P300 trials cannot cover two four-span messages.

Run the supported development fusion matrix with:

```bash
make counterfactual-evaluation
```

This Step 5 command verifies the Step 4 input manifest, requires the exact embedded development
specification, evaluates A–F plus the four supported ablations, writes result JSON,
trial/mapping JSONL, metric/interval CSV and a checksum manifest under
`artifacts/evaluation/counterfactual-fusion-development-v1/`, then reads the artifacts back before
succeeding. Add `COUNTERFACTUAL_EVALUATION_ARGS="--overwrite"` when intentionally replacing an
existing local development result.

The runner preserves the source flash stream and changes only which visible tile occupies the
recorded target signature. Protocol v2 keeps the
synthetic profile and message-span identity separate from the recorded EEG subject, records
whether the intended phrase was actually visible, and derives replay duration from the source
flash onsets. Conditions D–F require
an adapter ID, adapter checksum, and held-out personalization evidence. Controlled fixtures can
exercise those mechanics but are always marked non-claim-eligible. No real counterfactual result
is bundled because Study P data, generated language candidates, and trained adapters remain local.
The development configuration excludes irrelevant-retrieval and no-context ablations because the
current language artifact does not contain those alternate evidence snapshots.

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

Build the Step 6 evidence-separated development report with:

```bash
make development-report
```

The tracked recipe requires the current controlled simulation, xDAWN original-task, and Step 5
counterfactual artifacts. It verifies every source and writes separate evidence tables under
`artifacts/reports/neuroselect-development-evidence-v1/`. Target availability and conditional
recall are shown explicitly beside counterfactual top-k values. Add
`DEVELOPMENT_REPORT_ARGS="--overwrite"` when intentionally replacing an existing local report.

## Steps 7–8 research evidence protocol

Audit the exact research-grade prerequisites before starting long model or data jobs:

```bash
make research-readiness RESEARCH_READINESS_ARGS="--allow-incomplete"
```

The readiness command writes a machine-readable result under `artifacts/reports/` and checks all
of the following together:

- four checksum-verified adapters trained with the research LoRA recipe, including validation and
  test evaluation;
- the complete 1,000-message held-out language test split, containing 3,990 target spans, with no
  development message limit;
- clean, checksum-verified decoder training over all 13 training subjects, calibration over all
  three validation subjects, and evaluation over held-out subjects `P_02`, `P_11`, and `P_13`;
- a preregistered counterfactual sample of 144 distinct P300 selections, with exactly 48 per EEG
  subject and 36 per synthetic profile; and
- the balanced research input recipe and primary A–F fusion matrix.

Without `--allow-incomplete`, the command exits with status 2 whenever any requirement is missing.
This fail-closed behavior prevents limited development artifacts from being relabeled as research
evidence.

Step 8 separates the evidence denominators instead of trying to pair all 3,990 language spans.
The full language result remains mandatory component evidence. Counterfactual replay deterministically
selects three complete four-span messages for every synthetic-profile/EEG-subject cell: four
profiles × three subjects × three messages × four spans = 144 trials. Each complete message stays
within one EEG subject, source selections are never reused, and seeded hashing fixes both language
and EEG selection.

The current local audit is expected to remain incomplete: the development decoder contains seven
usable selections from `P_02`, rather than 48 each from all three test subjects; it was trained
with only `P_01` and calibrated with only `P_06`; the four research adapters and full language
result are also absent. Step 9 must build the complete preregistered Study P decoder evidence.

The tracked `language-research-adapter`, `language-research-evaluation`,
`counterfactual-research-input`, and `counterfactual-research-evaluation` targets provide exact
building blocks for the protocol. `counterfactual-research-input` runs the strict readiness gate
first and never synthesizes missing P300 evidence.

## Research release workflow

Build the evidence-separated report from available verified artifacts with:

```bash
make research-report
```

The tracked recipe requires the controlled simulation artifact and lists xDAWN, EEGNet, and
counterfactual results as optional until they exist locally. The generated JSON and Markdown keep
controlled simulation, original-task EEG, and counterfactual replay in separate tables. A missing
required source or dirty source manifest makes the report non-release-ready.

Validate public documentation and version metadata and build the Python distributions with:

```bash
make release-check
make package
```

`make release` runs the full verification, package, controlled evaluation, report, and strict
report-readiness gate. It is expected to fail on a dirty worktree or while a required artifact is
missing. See the [reproducibility guide](docs/reproducibility.md) for the evidence hierarchy.

## Research boundaries

The first real-data integration uses offline P300 replay. Recorded target evidence may be mapped
to visible candidate tiles for counterfactual system evaluation, but the resulting words were not
selected live by the original participants. Original-task decoder results, counterfactual replay
results, and controlled simulation results must always be reported separately.

Before contributing claims or experiment code, review the [research-scope ADR](docs/adr/0001-research-scope-and-claims.md), [Study P dataset card](docs/dataset-card.md), [P300 model card](docs/model-card.md), [counterfactual-fusion ADR](docs/adr/0014-counterfactual-fusion-evaluation.md), [input-preparation ADR](docs/adr/0019-language-p300-counterfactual-input-preparation.md), [balanced-sampling ADR](docs/adr/0023-balanced-counterfactual-research-sampling.md), and [research-release ADR](docs/adr/0015-research-release-and-reporting.md). Public-use boundaries are documented in the [privacy statement](docs/privacy.md), [limitations](docs/limitations.md), [responsible-use guidance](docs/responsible-use.md), [threat model](docs/threat-model.md), and [security policy](SECURITY.md). The remaining accepted decisions are indexed in [`docs/adr/`](docs/adr/).

## License

Source code is licensed under the MIT License. Datasets, model weights, and derived artifacts retain their own licenses and must be cited separately.
