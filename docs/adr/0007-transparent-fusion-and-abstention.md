# ADR 0007: Transparent fusion and abstention

- Status: accepted
- Date: 2026-07-18

## Decision

- Combine separately visible neural, generic-language, personalization, retrieval, diversity,
  and risk signals. Do not ask an LLM to perform an opaque fusion decision.
- Require neural and generic-language inputs to be normalized distributions over their declared
  domains. Personalization is a bounded lift in `[-1, 1]`; retrieval is a bounded relevance
  score in `[0, 1]`; diversity is a bounded negative Jaccard-overlap adjustment; risk is one of
  none, elevated, or high.
- Use a hand-set versioned safety baseline with neural weight `0.65` and combined non-neural
  weight no greater than `0.35`. All positive signal weights form a convex combination. Risk is
  an explicit subtractive penalty outside that combination.
- Preserve every visible candidate, including Other, Back, and Cancel. Controls receive neural
  support but no language, personalization, or retrieval score.
- Treat missing neural evidence as `None`, assign it no weighted contribution, and abstain rather
  than falling back to language-only ranking. Request a repeated neural selection when top neural
  support or margin is too low, or when the fused and neural top candidates conflict.
- Detect when the generic-language favorite's combined non-neural contribution exceeds its
  neural contribution. Expose an LM-dominance flag even when the neural floor successfully keeps
  the intended candidate first.
- Require enhanced confirmation for elevated/high-risk top candidates and neural-language
  conflicts. A ranking result may display a provisional top candidate, but it structurally
  forbids automatic selection and never constitutes message confirmation.
- Require the generator's application-owned risk taxonomy to match the ranker's elevated/high
  taxonomy at service construction, preventing configured confirmation categories from becoming
  unreachable or silently ignored.
- For the P300 MVP, neural probabilities map directly to the identifiers of visible candidate
  tiles. Semantic mapping from broad EEG classes to open-vocabulary candidates is out of scope
  until a paradigm that requires it is evaluated.

## Weight and threshold tuning

The initial weights and thresholds are policy defaults, not fitted scientific results. Later
tuning must use only training/validation subjects and sessions, optimize a preregistered utility
that penalizes unintended selections, and lock one policy revision before held-out testing.
Report sensitivity across weight grids, candidate counts, missing-neural conditions, and
neural-language conflicts. Never select weights on the final test set.

## Consequences

The ranker is deterministic, auditable, and suitable for adversarial system tests before a real
decoder or LoRA exists. The hand-set policy cannot establish optimal communication speed or
safety. The current lexical diversity penalty is intentionally simple, and the risk taxonomy is
a confirmation policy rather than a medical or legal classifier.
