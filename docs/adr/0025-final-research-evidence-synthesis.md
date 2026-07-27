# ADR 0025: Final research evidence synthesis

- Status: Accepted
- Date: 2026-07-27

## Context

Steps 9–12 produced clean, checksum-addressed artifacts for the full-split Study P xDAWN decoder,
four research LoRA adapters and their complete 3,990-span held-out evaluation, and a balanced
144-trial counterfactual replay. The earlier research-report recipe predated those artifacts. It
treated development xDAWN, EEGNet, and counterfactual paths as optional, had no explicit language
component table, and reran the controlled simulation whenever a report was built.

That behavior cannot define the final evidence package. A report must not silently omit a completed
research component, promote a development path, or mutate a frozen source while assembling it.
Language target availability also cannot be hidden behind conditional ranking values.

## Decision

Step 13 requires exactly four verified source tiers:

1. controlled simulation;
2. complete held-out language and personalization evaluation;
3. research xDAWN original-task Study P evaluation; and
4. research counterfactual fusion replay.

Add `language_component` as a separate report evidence kind. The table verifies the held-out
language result, trials, metrics, manifest metadata, adapter coverage, and checksums through the
existing strict artifact reader. It reports overall and per-profile target availability, generic
and personalized unconditional top-k recall, recall conditional on availability, reciprocal rank,
and message accuracy. It adds no inferential interval that was not present in the source result.

The canonical report recipe points only to the research xDAWN, full held-out language, and research
counterfactual paths, and every source is required. Report construction consumes frozen artifacts
directly; it no longer has a Make dependency that reruns simulation. Controlled simulation,
language, original-task EEG, and counterfactual replay remain separate tables and are never pooled.

`claim_eligible` continues to mean that a source satisfies its locked protocol, coverage, and clean
provenance rules. It does not mean that a result is positive, statistically conclusive, clinically
useful, or representative of live BCI communication.

## Consequences

The final report fails closed if any Step 13 source is absent, checksum-invalid, or dirty. Its
machine-readable and Markdown forms expose the weak absolute language availability and the limited
three-subject counterfactual denominator instead of smoothing them into a single system score.

The evidence package is reproducible and suitable for transparent engineering or pilot-paper
reporting. It does not create a human-subject end-to-end result. Stronger efficacy claims require a
new candidate-generation study and ultimately a preregistered live participant protocol.
