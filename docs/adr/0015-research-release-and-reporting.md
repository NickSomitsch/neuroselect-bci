# ADR 0015: Research release and evidence-separated reporting

- Status: Accepted
- Date: 2026-07-20

## Context

NeuroSelect now emits simulation, original-task decoder, and counterfactual artifacts, but a public
release needs one reproducible reporting path without encouraging those evidence types to be
pooled. A report consumer must verify artifacts before displaying metrics and must not execute an
untrusted joblib or PyTorch checkpoint merely to read results. Missing real data and LoRA evidence
must remain visible rather than being replaced with fixture numbers.

## Decision

Introduce a tracked research-report recipe whose sources declare an expected run kind, path,
requirement status, and optional paired reference condition. Verify the source manifest and every
declared output checksum before parsing results. Reconstruct simulation results from `metrics.json`
and `trials.jsonl` and verify their count and canonical digest. Read original-task JSON metadata and
evaluation without loading checkpoint payloads. Reuse the stricter counterfactual artifact reader.

Render controlled simulation, original-task EEG, and counterfactual replay in separate statistical
tables with an evidence-scope statement, source Git and manifest identity, claim-scope eligibility,
and limitations. Never compute a combined performance estimate. Simulation comparisons use a
deterministic paired hierarchical bootstrap over profile then trial; counterfactual reports retain
their subject-then-trial intervals. All intervals are descriptive.

Emit canonical JSON, Markdown, and a checksum-addressed report manifest. Under the tracked release
policy, a required missing source or any dirty source run makes `release_ready=false`. Optional
real-data sources remain explicitly listed as missing. The release gate also requires citation,
contribution, security, privacy, limitations, responsible-use, threat-model, reproducibility,
dataset-card, and model-card files, aligned version metadata, the complete test suite, and a
successful source/wheel build.

## Consequences

The repository can produce an auditable engineering report today and incorporate real local
artifacts later without changing their evidence meaning. A dirty or incomplete report is still
useful for development, but cannot pass the release-ready gate. No all-subject Study P, real LoRA,
counterfactual, live-BCI, clinical, or human-subject result is created by this reporting layer.
