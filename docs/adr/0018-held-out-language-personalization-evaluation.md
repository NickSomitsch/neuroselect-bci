# ADR 0018: Held-out natural-candidate language evaluation

- Status: accepted
- Date: 2026-07-24

## Decision

- Evaluate only the synthetic benchmark test split and use the ground-truth preceding spans as
  teacher-forced confirmed context.
- Generate candidates naturally with the pinned generic model. Never insert the intended span.
  Record target absence separately from generation failure and never count the `Other` control as
  successful phrase recall.
- Treat the model as an autocomplete engine: proposals must append directly to confirmed text and
  must not be replies, paraphrases, or interface commands. For non-empty contexts, constrain Qwen
  to compact noun, deadline, or ending phrase classes derived structurally from train and
  validation messages only. Record the vocabulary checksum. Leave empty-context generation
  unconstrained rather than exposing test-only opening patterns.
- Require strict candidate objects. When MLX emits valid candidate objects but corrupts only the
  outer JSON collection closers, recover that single structural defect and record the repaired
  round rate. Reject malformed candidate objects, prose, markdown, and arbitrary trailing content.
- Keep the visible candidate set fixed while scoring it with the profile-matched, checksum-verified
  LoRA. Report generic and personalized top-1/top-3 recall, conditional recall, reciprocal rank,
  message completion, target availability, and rank change.
- Export candidate-aligned generic support, personalization support and lift, and lexical retrieval
  evidence for later counterfactual preparation. Retrieval remains separate from language ranks.
- Verify the source benchmark, per-profile corpus manifest, adapter weights, base-model revision,
  and training configuration provenance before evaluation.
- Mark limited runs and development adapters as non-claim-eligible. A claim-eligible run requires
  every held-out test message and adapters trained with the tracked research configuration.

## Consequences

The project can now measure whether the intended phrase was naturally proposed before interpreting
ranking quality, and it can distinguish generic-model ranking from adapter ranking on exactly the
same visible candidates. The development recipe uses nine language candidates plus three controls
and evaluates one deterministically selected message per profile so the real MLX path can be
checked quickly; it is not a personalization-benefit result.

Preparing paired counterfactual inputs still requires calibrated Study P decoder artifacts and a
locked mapping from these language rounds to recorded selection trials.
