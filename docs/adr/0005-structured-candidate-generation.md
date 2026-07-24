# ADR 0005: Structured candidate-language generation

- Status: accepted
- Date: 2026-07-17

## Decision

- Use strict structured JSON proposals for the MVP rather than relying on direct tokenizer
  probabilities. The required unit is a visible word or short phrase, which may span multiple
  model tokens and must remain portable across local backends.
- Treat backend support values as relative generic-language evidence. Normalize them over the
  displayed language candidates, but do not describe them as calibrated intent probabilities.
- Interpret `candidate_count` as the complete visible set. Reserve three positions for
  application-owned Other, Back, and Cancel controls; a language model cannot create or score
  these controls.
- Normalize Unicode and whitespace, deduplicate case- and punctuation-insensitively, reject
  control characters and overlong phrases, and fail closed if too few safe unique candidates
  remain.
- Apply an application-owned, versioned lexical risk policy after normalization. Generated model
  output cannot assign or remove these tags. Tag matches feed the ranking and confirmation policy;
  they are conservative safeguards, not a complete medical, legal, or privacy classifier.
- Derive context, candidate-set, and candidate identifiers from canonical confirmed text and
  versioned generator inputs. Record backend, model, generator, prompt, and risk-policy revisions
  on every result.
- Keep generation pure with respect to the message session. A candidate remains provisional
  until a separate explicit selection action confirms it.
- Ship a deterministic YAML fixture backend for tests and CPU-only development. ADR 0017 adds the
  optional Qwen/MLX adapter while retaining this fixture as the offline default.

## Consequences

The fixture backend is not a language-model quality baseline, but it makes candidate policy,
API contracts, and later ranking experiments reproducible without downloading model weights.
Structured prompting may not expose exact token likelihoods; later model adapters must document
how relative support is obtained and must retain an LM-only baseline for comparison.
ADR 0017 resolves that requirement by rescoring each complete proposed phrase with mean
continuation token log likelihood instead of trusting support numbers emitted in JSON.
