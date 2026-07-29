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

Run the full 3,990-span Step 11 research evaluation on an MLX-compatible Colab GPU with the
tracked [Colab notebook](notebooks/neuroselect_step11_colab.ipynb). First create the portable
input archive locally:

```bash
make language-cloud-bundle LANGUAGE_CLOUD_BUNDLE_ARGS="--overwrite"
make language-cloud-verify \
  LANGUAGE_CLOUD_BUNDLE=artifacts/cloud/step11-language-inputs-v1.tar.gz
```

The approximately 113 MB ignored archive contains only the four final research adapters and
checksum-verified personalization corpora. It excludes intermediate adapter checkpoints and the
Qwen base model. After committing the implementation, create the private source bundle with
`make language-cloud-source-bundle LANGUAGE_CLOUD_SOURCE_BUNDLE_ARGS="--overwrite"`. Upload it as
`MyDrive/neuroselect-step11/neuroselect-step11-source.bundle` and upload the adapter archive as
`MyDrive/neuroselect-step11/step11-language-inputs-v1.tar.gz`. Paste the exact bundled commit SHA
into the notebook and run its cells in order. The notebook:

- rejects GPUs below CUDA compute capability 7.5 before installation;
- installs the locked Python 3.12 and `local-language-cuda` environment;
- safely verifies and extracts the adapter/corpus archive;
- downloads the exact pinned Qwen revision once into a persistent Drive cache and copies it to
  Colab's local SSD for inference;
- runs a short development-limit pilot with all four research adapters;
- enumerates only the applicable train/validation candidate vocabulary, batches Qwen likelihood
  scoring behind one shared prompt cache, and reuses identical context-only requests without
  inserting intended targets;
- writes active checkpoints to local SSD and atomically mirrors every 25 new research trials to
  Drive, resuming only when every input identity matches; and
- verifies all 3,990 ordered trials, clean Git provenance, checksums, and claim eligibility.

The canonical result and an export archive remain under `MyDrive/neuroselect-step11/`. On a Linux
CUDA host, the equivalent full command is:

```bash
make local-language-cuda-sync
make language-cuda-preflight
make language-model-cache LANGUAGE_MODEL_CACHE_ARGS="--download"
make language-research-evaluation-cuda LANGUAGE_RESEARCH_EVALUATION_ARGS="\
  --checkpoint-dir /persistent/neuroselect-step11/checkpoint-v1 \
  --resume --checkpoint-every 5 --progress-every 25 --overwrite"
make language-research-verify
```

`--resume` never mixes trials across code, protocol, model, adapter, corpus, or vocabulary
changes. The final artifact records CUDA device provenance. See
[the Step 11 Colab runbook](docs/step11-colab.md) for the exact handoff and recovery workflow.

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

For the publication-only original-task comparator, first run the bounded five-epoch pilot and then
the complete locked training split:

```bash
make p300-eegnet-pilot P300_EEGNET_PILOT_ARGS="--overwrite"
make p300-eegnet-research P300_EEGNET_RESEARCH_ARGS="--overwrite"
```

The research target skips chronological adaptation because the publication comparison is the
subject-independent original Study P task. It records elapsed time, peak process RSS, calibrated
event metrics, and explicit occurrence-level target-event ranking metrics.

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

## Step 9 full Study P EEG evidence

Prepare every labeled `Train` recording from both Study P sessions with:

```bash
make study-p-research-data
```

The target downloads from PhysioNet's public AWS mirror through sixteen bounded workers, verifies
every EDF against the pinned PhysioNet SHA-256 inventory, preprocesses 190 recordings, and audits
the exact fixed split: 13 training subjects, 3 validation subjects, and 3 held-out test subjects.
It requires five recordings per subject/session and at least 48 usable labeled selections from
each of `P_02`, `P_11`, and `P_13`.

Train the separate research xDAWN/LDA artifact and audit it with:

```bash
make p300-research-baseline
make p300-research-audit
```

The trainer refuses incomplete data. The audit verifies decoder metadata and evaluation checksums
without loading the executable joblib checkpoint, requires the exact train/calibration/test
subject sets, and checks replay capacity per held-out subject. Research evidence must originate
from a clean worktree; a functionally valid decoder trained while source changes are uncommitted
is correctly reported as dirty and must be regenerated after committing.

## Research release workflow

Step 13 builds the evidence-separated report from the four frozen, verified research inputs:

- the controlled simulation engineering matrix;
- the complete 3,990-span held-out language and personalization result;
- the full-split research xDAWN Study P original-task result; and
- the balanced 144-trial research counterfactual replay.

Build the report without rerunning or overwriting any source experiment:

```bash
make research-report
```

The tracked recipe requires all four artifacts. The generated JSON and Markdown keep controlled
simulation, held-out language, original-task EEG, and counterfactual replay in separate tables. A
missing required source or dirty source manifest makes the report non-release-ready. Protocol
claim eligibility means that an artifact satisfies its locked coverage and provenance rules; it
does not mean that the measured effect is large, statistically conclusive, or clinically useful.

After committing the Step 13 implementation so the report itself can have clean source provenance,
rebuild it and run the strict generated-report gate:

```bash
make research-report
make release-check RELEASE_CHECK_ARGS="--report artifacts/reports/neuroselect-research-release-v1"
make package
```

`make release` runs the full verification, package, controlled evaluation, report, and strict
report-readiness gate. It fails on a dirty worktree or while a required artifact is missing. See
the [reproducibility guide](docs/reproducibility.md) for the evidence hierarchy. The locked
offline-journal framing, immutable evidence inventory, venue order, and external submission gates
are recorded in the [publication strategy](paper/publication-strategy.md). Validate them without
rerunning an experiment with:

```bash
make publication-protocol-check
make publication-analysis PUBLICATION_ANALYSIS_ARGS="--overwrite"
```

The publication analysis verifies every frozen source manifest, pins the optional EEGNet
comparator, uses 10,000 message- or selection-clustered resamples, and writes canonical JSON plus
CSV estimates and intervals under `artifacts/publication/offline-methods-v1/`. Language resampling
preserves complete messages within fixed profile strata; counterfactual resampling preserves
complete messages within held-out EEG subjects.

Run the specified exploratory candidate-generation v2 comparison with:

```bash
make candidate-generation-v2 CANDIDATE_GENERATION_V2_ARGS="--overwrite"
```

This CPU-only analysis fits a profile-conditioned contextual phrase bank exclusively from
synthetic train and validation messages, fixes the visible language budget at nine, and compares
all 3,990 spans with the frozen v1 artifact. The generator API receives only profile, confirmed
context, and round number; intended test spans are used afterward for scoring. Output is written
under `artifacts/publication/candidate-generation-v2-exploratory-v1/` and is an exploratory,
test-exposed supplement rather than replacement primary evidence.

Run the locked Step 4 ablations and held-out-combination robustness benchmark with:

```bash
make candidate-generation-step4 CANDIDATE_GENERATION_STEP4_ARGS="--overwrite"
```

Step 4 reproduces full v2, removes profile conditioning, removes grammar routing, retains only
frequency scoring, and evaluates a two-stage opening interface. The second dataset holds out exact
opening stem-action combinations while exposing each component only through train/validation.
Generation remains target-blind; stage two sees the simulated selected stem, not the intended
action. The CPU-only run writes checksummed trials, metrics, paired 95% intervals, candidate banks,
and generated robustness splits under `artifacts/publication/candidate-generation-step4-v1/`.
The benchmark is synthetic and developer-authored, so this remains exploratory evidence.

Run the harder hierarchical opening-generalization experiment with:

```bash
make opening-generalization OPENING_GENERALIZATION_ARGS="--overwrite"
```

This locked CPU-only experiment expands the bottleneck beyond one menu: 24 fitted stems and 48
content words cover requests, preferences, clarifications, and status openings. It evaluates 288
held-out combinations plus 384 openings from paraphrase families absent from fitting. Complete
phrase retrieval, two-stage stem/content composition, and three-stage intent/stem/content
composition share a maximum menu size of nine. Outputs include exact coverage, paired intervals,
menus reached, candidate exposures, and coverage per required selection under
`artifacts/publication/opening-generalization-v1/`.

Build the checksum-pinned manuscript tables and figures with:

```bash
make publication-display PUBLICATION_DISPLAY_ARGS="--overwrite"
```

The display recipe verifies the clean primary analysis, candidate-generation v2, Step 4, and
opening-generalization manifests before reading any estimate. It emits ten CSV/Markdown tables,
five figures in SVG, 300-dpi PNG, and PDF, a caption sheet, an inventory, and a manifest under
`artifacts/publication/paper-display-v1/`. Primary language, primary original-task EEG, the
secondary EEGNet comparator, primary counterfactual replay, and exploratory candidate-generation
evidence remain visibly labeled and are never pooled into a single score. The command fails on a
dirty worktree for the final bundle; `--allow-dirty` exists only for implementation-time visual
validation.

Assemble the journal-neutral manuscript after the clean display exists:

```bash
make manuscript MANUSCRIPT_ARGS="--overwrite"
```

The assembler verifies the exact display manifest, all ten table and five figure markers, fifteen
references, and the quantitative claim ledger before writing synchronized DOCX, LaTeX, BibTeX,
compiled PDF, and rendered Markdown forms under `artifacts/publication/manuscript-v1/`. The LaTeX
bundle is self-contained and includes the five exact PDF figures. Tectonic is required for the
compiled PDF (`brew install tectonic` on macOS).

The journal-editable source and bibliography are tracked under `paper/latex/`. After changing the
canonical Markdown, references, tables, or figures, synchronize them during development with:

```bash
make manuscript-latex-sync MANUSCRIPT_ARGS="--overwrite --allow-dirty"
```

The ordinary `manuscript` target fails if those tracked LaTeX files are stale. `assembly_ready`
means that all document formats and their inputs came from clean verified source; it does not mean
`submission_ready`. During implementation, `--allow-dirty` permits visual QA while keeping
assembly readiness false.

The next paper step is independent scientific review and journal-specific submission preparation.
Use the [submission checklist](paper/submission-checklist.md) to obtain BCI review, institutional
secondary-use wording, open-access confirmation, and final author metadata before adapting the
journal-neutral LaTeX source to a venue template.

## Research boundaries

The first real-data integration uses offline P300 replay. Recorded target evidence may be mapped
to visible candidate tiles for counterfactual system evaluation, but the resulting words were not
selected live by the original participants. Original-task decoder results, counterfactual replay
results, and controlled simulation results must always be reported separately.

Before contributing claims or experiment code, review the [research-scope ADR](docs/adr/0001-research-scope-and-claims.md), [Study P dataset card](docs/dataset-card.md), [P300 model card](docs/model-card.md), [counterfactual-fusion ADR](docs/adr/0014-counterfactual-fusion-evaluation.md), [input-preparation ADR](docs/adr/0019-language-p300-counterfactual-input-preparation.md), [balanced-sampling ADR](docs/adr/0023-balanced-counterfactual-research-sampling.md), [full Study P evidence ADR](docs/adr/0024-full-study-p-research-evidence.md), [research-release ADR](docs/adr/0015-research-release-and-reporting.md), [final evidence-synthesis ADR](docs/adr/0025-final-research-evidence-synthesis.md), [offline-publication ADR](docs/adr/0026-offline-journal-publication-strategy.md), [exploratory candidate-generation ADR](docs/adr/0027-exploratory-target-blind-candidate-generation-v2.md), [opening-robustness ADR](docs/adr/0028-candidate-generation-ablation-and-opening-robustness.md), [hierarchical-opening ADR](docs/adr/0029-hierarchical-opening-generalization.md), [publication-display ADR](docs/adr/0030-publication-tables-and-figures.md), and [manuscript-assembly ADR](docs/adr/0031-verifiable-manuscript-assembly.md). Public-use boundaries are documented in the [privacy statement](docs/privacy.md), [limitations](docs/limitations.md), [responsible-use guidance](docs/responsible-use.md), [threat model](docs/threat-model.md), and [security policy](SECURITY.md). The remaining accepted decisions are indexed in [`docs/adr/`](docs/adr/).

## License

Source code is licensed under the MIT License. Datasets, model weights, and derived artifacts retain their own licenses and must be cited separately.
