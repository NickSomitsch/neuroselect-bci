# ADR 0017: Local language scoring and synthetic personalization

- Status: accepted
- Date: 2026-07-23

## Decision

- Keep the deterministic fixture backend as the default setup and CI path. Provide MLX-LM through
  an explicit optional dependency and never download model weights without an explicit command
  flag.
- Pin `Qwen/Qwen3-4B-MLX-4bit` to an exact Hugging Face revision. Load only that cached revision
  unless the operator explicitly allows its first download.
- Use structured JSON only to propose candidate text. Ignore support values emitted by the model
  and rescore every complete word or phrase with mean continuation-token log likelihood. Softmax
  those finite scores over the candidate set as relative language evidence; do not call them
  calibrated intent probabilities.
- Keep the visible candidate set fixed while computing personalization. A verified LoRA scores the
  same candidates, and personalization lift is exactly personal support minus generic support.
  Candidate-level lexical RAG remains a separate signal with record provenance.
- Generate MLX completion corpora independently for each synthetic profile from the tracked
  train, validation, and test message splits. Train only on `train.jsonl`, mask prompt loss, use
  `valid.jsonl` during training, and evaluate `test.jsonl` after training.
- Bind each adapter to the synthetic profile, pinned base model, corpus-manifest checksum, trainer
  revision, complete training-configuration checksum, validation/test status, and learned-weight
  checksum. Reject missing, mismatched, or tampered adapter artifacts.
- Provide a deterministic controlled style proxy so end-to-end mechanics and A–F input alignment
  can be tested without model weights. Label it `controlled_fixture` and prohibit claim
  eligibility; it is not a fake LoRA.

## Consequences

The repository now contains a runnable local language and personalization implementation without
making model downloads part of setup or CI. Researchers can prepare four split-safe synthetic
corpora, train profile-specific QLoRA adapters on Apple silicon, and export generic, personal, and
RAG signals over one candidate set.

No trained adapter is committed, and implementing a trainer is not evidence that personalization
improves communication. A paper claim still requires running the held-out language benchmark,
recording candidate availability and ranking metrics, preparing candidate-aligned counterfactual
trials, and keeping controlled-proxy results separate from trained-adapter results.
