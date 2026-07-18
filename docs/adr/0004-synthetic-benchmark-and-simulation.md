# ADR 0004: Synthetic benchmark and neural simulation

- Status: accepted
- Date: 2026-07-17

## Decision

- Track four clearly labeled synthetic persona specifications, not generated messages or
  private communication data.
- Keep stable writing style separate from permissioned knowledge records. Temporary facts
  use the `current_event` kind and require an explicit expiry time.
- Generate exactly 1,000 training, 150 validation, and 250 test messages per persona from a
  versioned seed and compact YAML sources.
- Keep template identifiers and topics disjoint across train, validation, and test. Reject
  duplicate messages within a persona split and assert that persona messages do not overlap
  across splits.
- Represent each message as short selectable target spans of no more than four
  whitespace-delimited tokens.
- Store generated JSONL files under ignored `artifacts/`; emit a manifest containing source
  and artifact SHA-256 digests so runs can identify their exact inputs.
- Simulate neural evidence from the global seed plus stable session, round, intended-target,
  and candidate-order identifiers. Do not consume mutable global random state.
- Label simulator rounds as target-supported, distractor-lapse, or ambiguous and keep the
  known intended candidate alongside the resulting evidence.

## Consequences

The benchmark is reproducible across checkout locations and can be regenerated without
shipping a large derived corpus. Held-out results test unseen wording structures and topics,
but they remain synthetic and cannot establish clinical utility or real-world BCI accuracy.
The seeded simulator supports deterministic system tests; it is not an EEG model and its
probabilities must never be reported as recorded neural performance.
