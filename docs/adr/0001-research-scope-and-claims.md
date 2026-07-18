# ADR 0001: Research scope and claims

- Status: accepted
- Date: 2026-07-17

## Context

NeuroSelect investigates whether personalized language prediction and retrieval can reduce selections needed for non-invasive BCI communication without increasing unintended or weakly supported messages.

Scalp EEG does not provide unrestricted, reliable open-vocabulary thought decoding. A language-model prior must not be presented as neural evidence or silently converted into an asserted user message.

## Decision

- The first release is a reproducible systems study, not a clinical or human-subject study.
- Neural evidence, generic language probability, LoRA influence, RAG influence, and explicit user confirmation remain separate in schemas, logs, evaluation, and UI.
- No candidate is finalized as a message without explicit confirmation.
- Original-task EEG evaluation, counterfactual replay, and controlled simulation are reported separately.
- The project makes no mind-reading, medical-device, diagnostic, treatment, or clinical-efficacy claims.

## Consequences

The repository may demonstrate language-assisted candidate selection, but it must not claim that generated words were decoded from private thoughts. Any future live or human-subject work requires a separate protocol, consent process, and ethics review.
