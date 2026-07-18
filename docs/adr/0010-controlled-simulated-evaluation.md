# ADR 0010: Controlled simulated evaluation before real EEG

- Status: Accepted
- Date: 2026-07-18

## Context

The manual and seeded-simulation vertical slice now works, but it needs a reproducible experiment
framework before real EEG data is introduced. The framework must distinguish implemented
conditions from planned LoRA and calibrated-decoder conditions, exercise the actual generator,
retriever, simulator, and transparent ranker, and avoid presenting controlled simulation as EEG
evidence.

The fixture language backend is not trained to recall every span in the synthetic held-out message
benchmark. Measuring fusion only on spans it happens to propose would confound target availability
with ranking behavior and create selection bias.

## Decision

Add a controlled, paired fusion protocol over the synthetic test split. For every target span, the
protocol supplies that target and deterministic held-out distractors to the real candidate policy,
which still performs normalization, filtering, deduplication, identifier generation, and control
injection. This makes target availability a protocol invariant and explicitly does **not** measure
free-form candidate-generation recall.

Each condition receives the same candidate set and base simulated neural draw. The tracked matrix
includes:

- A: simulated BCI evidence only;
- B: generic language support only;
- C: simulated neural plus generic language support;
- current neural/language/RAG fusion without abstention;
- current transparent fusion with repeat and abstention;
- uniform and deterministically shuffled neural evidence;
- removed, shuffled, and deliberately irrelevant retrieval evidence; and
- removed confirmed conversation context.

The planned D, E, and F conditions remain in the machine-readable catalog but are unavailable.
They require a real held-out LoRA evaluation and, for F, a calibrated P300 decoder. The runner
rejects configurations that request an unavailable condition.

Every fourth controlled trial creates a language conflict by making a distractor linguistically
most likely while preserving the intended option. The system still forbids automatic selection,
and tests verify that strong neural evidence can keep the target first without silently confirming
it.

The one-pass evaluation counts a span as completed only when the displayed top recommendation is
the intended span. An ideal explicit confirmer rejects incorrect recommendations, so an incorrect
top recommendation increases correction burden but never becomes an unintended output. This is a
conservative automated protocol, not a participant usability model.

Reported time is a versioned interaction-time model covering candidate presentation and explicit
actions. It is not wall-clock code performance or measured participant latency. Results include
overall and per-synthetic-profile top-1/top-3 recall, completion, exact-message accuracy,
correct selections and words per modeled minute, selections per completed message, correction,
repeat, abstention, unintended-word safety, top-label ECE, multiclass Brier score, and conflict
slices.

## Reproducibility and artifacts

The runner uses only the synthetic test split, stable message ordering, fixed seeds, deterministic
probability transformations, a fixed evaluation time, and canonical JSON serialization. It writes:

- `trials.jsonl` with one auditable record per condition and target span;
- `metrics.json` with the condition catalog, aggregate metrics, assumptions, and result digest; and
- `manifest.json` with Git/config/data/protocol identifiers and output checksums.

Identical source inputs, configuration, and Git revision produce byte-identical artifacts in
different output directories. When the working tree is dirty, the manifest records a digest over
the Git diff and non-ignored untracked files in addition to the base Git revision.

## Consequences

Step 8 can detect regressions and compare current components without downloading EEG data or
claiming real-world communication performance. Results are suitable for engineering validation
and experiment-design review only. They cannot establish participant speed, EEG decoding quality,
LoRA benefit, clinical utility, or generalization to a real BCI session.

The deliberately irrelevant-retrieval condition bypasses normal retrieval relevance to create a
labeled stress input for the ranker. It does not weaken the production retriever's permission,
expiry, profile, or injection filters.
