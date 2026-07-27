# Reproducibility guide

## Environment and source

Use Python 3.12, `uv`, Node.js 22, and `pnpm` 10. Record the exact Git revision and keep the
worktree clean for release evidence. Install only from the tracked lockfiles:

```bash
make setup
make verify
make release-check
make package
```

CI performs the same formatting, lint, type, Python/UI test, release-metadata, package-build, and
frontend-build gates without downloading EEG or model artifacts.

## Experiment identity

Every experiment has a tracked YAML recipe, fixed seeds, immutable data/model/policy identifiers,
and a `RunManifest` containing runtime versions, device, output checksums, and dirty-tree state.
Never compare files before verifying their manifests. Hardware and library differences may still
change PyTorch optimization; do not silently pool runs from different environments.

Run the public controlled matrix with `make simulated-evaluation`. Study P and decoder commands are
documented in the README and require explicit local data preparation. Data-author `Train`/`Test`
folders are not model splits. Unknown labels never enter supervised metrics.

Prepare deterministic synthetic personalization corpora with
`make language-personalization-data`. Each profile directory contains MLX-compatible train,
validation, and test files plus a checksum manifest. The default `make language-smoke` path uses a
controlled style proxy and must not be reported as LoRA evidence. Real local runs require
`make local-language-sync`, the exact pinned Qwen revision, and a separately checksum-verified
adapter produced by `make language-lora`; keep the base-model and adapter identities in every
downstream artifact.

Build the local development counterfactual input with `make counterfactual-input`. The builder
verifies the held-out language and decoder manifests, pairs only complete language messages with
distinct labeled EEG selection trials, and records all source checksums. Run it with
`configs/experiments/counterfactual_fusion_development.yaml`; this limited evidence is always
non-claim-eligible.

Run Step 5 with `make counterfactual-evaluation`. The command refuses any fusion configuration
that differs from the specification embedded in the prepared input, requires complete paired
trial and interval coverage, and reads the written artifacts back through the strict verifier.
Use `COUNTERFACTUAL_EVALUATION_ARGS="--overwrite"` only when intentionally replacing the local
development result.

Build Step 6 with `make development-report`. Its tracked recipe requires the controlled
simulation, xDAWN original-task, and development counterfactual manifests and renders them as
separate evidence tables. The report shows target availability and conditional counterfactual
recall and never pools the three evidence tiers. Use
`DEVELOPMENT_REPORT_ARGS="--overwrite"` only to replace an existing local report.

Audit the Steps 7–8 research protocol with:

```bash
make research-readiness RESEARCH_READINESS_ARGS="--allow-incomplete"
```

The tracked research-expansion recipe computes the complete held-out demand from source data rather
than accepting a manually entered count. It currently resolves to 1,000 messages and 3,990 target
spans. It then verifies the research LoRA recipe, all four profile adapters, the unlimited language
result, clean decoder provenance, complete training/calibration/test subject coverage, balanced
P300 capacity, the sampled input builder, and the primary A–F fusion matrix. Omit
`--allow-incomplete` in automation that must fail unless every prerequisite is satisfied.

Step 8 preregisters 144 counterfactual trials separately from the complete 3,990-span language
evaluation. For every combination of four synthetic profiles and three held-out EEG subjects,
seeded sampling selects three complete four-span messages. This produces 36 trials per profile and
48 per EEG subject. Every message maps to selections from one subject; every recorded selection is
used at most once. The source manifest records exact profile and subject counts.

The current development decoder contributes seven selections from one held-out subject. The
readiness gate therefore remains blocked until every held-out subject contributes at least 48
usable selections and the decoder metadata verifies all preregistered training and calibration
subjects. Counterfactual intervals remain descriptive hierarchical bootstrap summaries; three EEG
test subjects do not support population or clinical generalization.

Run Step 9 only after accepting the Study P CC-BY-4.0 license:

```bash
make study-p-research-data
make p300-research-baseline
make p300-research-audit
```

The data target uses bounded parallel downloads but retains per-file official checksum
verification and atomic writes. Its post-download audit verifies all 190 labeled `Train`
recordings without trusting directory presence alone. The research decoder is written separately
under `artifacts/models/p300-xdawn-lda-research-v1/`; development results under
`p300-xdawn-lda-v1/` are not overwritten or promoted.

The Step 9 audit reads only checksum-verified JSON metadata and predictions. It requires the exact
13/3/3 train/validation/test subject split, at least 48 replay trials per held-out subject, and a
clean source manifest. The joblib checkpoint is not loaded by the audit.

## Research report

Step 13 consumes the frozen outputs from the controlled simulation, Step 11 full held-out language
evaluation, Step 9 research xDAWN evaluation, and Step 12 research counterfactual evaluation. It
does not retrain a model, download data, or rerun a source experiment. Run:

```bash
make research-report
```

The tracked report recipe requires all four research artifacts. It verifies every output checksum,
does not execute checkpoint files, reconstructs simulation result digests, and emits canonical
JSON, readable Markdown, and a report manifest. `release_ready` is false if any required source is
missing or if the tracked clean-source policy is violated. The report target deliberately does not
depend on `simulated-evaluation`, because rebuilding a frozen source while preparing the report
would change its identity and could make its provenance dirty.

Simulation intervals resample synthetic profiles and then paired trials. Counterfactual intervals
resample subjects and then trials. The language table reports unconditional candidate recall
beside recall conditional on target availability; it does not invent an interval absent from the
source result. Evidence tiers are rendered in separate tables and must remain separate in
publications.

The report generator records the current source-tree state in its own manifest. Therefore, after
committing a Step 13 implementation, regenerate and strictly validate the final artifact with:

```bash
make research-report
make release-check RELEASE_CHECK_ARGS="--report artifacts/reports/neuroselect-research-release-v1"
```

## Evidence hierarchy

1. Controlled simulation: engineering behavior only.
2. Held-out language component: synthetic teacher-forced candidate generation and ranking only.
3. Original-task EEG: source P300 target/non-target decoding only.
4. Counterfactual replay: candidate-system behavior over remapped recorded evidence.
5. Future live or participant study: unavailable and requiring a separate protocol.

Absence of a higher tier cannot be filled by relabeling a lower tier.
