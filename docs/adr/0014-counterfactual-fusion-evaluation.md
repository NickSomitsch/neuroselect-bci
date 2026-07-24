# ADR 0014: Counterfactual P300 fusion evaluation

- Status: Accepted
- Date: 2026-07-19

## Context

The calibrated P300 decoders emit an event-level target probability for each recorded flash, but
the NeuroSelect ranker requires a probability distribution over the candidates visible in one
round. Study P participants selected characters in the source task; they did not select generated
NeuroSelect words or phrases. The experiment therefore needs a reproducible bridge that preserves
the source evidence without representing counterfactual candidate text as an observed intention.

The comparison also needs to run every condition on the same candidate set and recorded flash
stream. Personalization and retrieval inputs must remain separately inspectable, and missing
held-out evidence must fail explicitly rather than being replaced with an unlabelled heuristic.

## Decision

For a fixed flash layout, aggregate calibrated event probabilities using a candidate-wise binary
log likelihood. For candidate `c`, sum `log(p)` when the flashed stimulus code belongs to `c`'s
signature and `log(1-p)` otherwise, then apply a temperature-controlled softmax across candidates.
Probability clipping, temperature, and the minimum repetition count are locked in the experiment
configuration and included in its checksum. Incomplete stimulus-code rounds are rejected.

Counterfactual replay changes only the tile-to-stimulus signature mapping. The visible intended
candidate receives the recorded target signature, with the displaced tile receiving its former
signature. Event order, event identifiers, onset times, stimulus codes, calibrated probabilities,
subject, session, and recorded target codes are unchanged. If the intended candidate is not
visible, the recorded target is mapped to the explicit `Other` control. This is an `Other`-tile
decision, not successful language-candidate recall.

Every prepared input retains source decoder-manifest and original-task-evaluation checksums. Each
result additionally records the input/config checksums, source and mapped layout checksums,
ordered event identifiers and onsets, adapter IDs and checksums, and normalized ranking records.
The runner evaluates paired conditions on every prepared trial:

- A: recorded neural evidence only, without language or safeguards.
- B: generic language support only.
- C: recorded neural plus generic language support.
- D: C plus explicitly supplied personalization lift and adapter provenance.
- E: D plus the recorded retrieval snapshot.
- F: the complete transparent ranker with repeat, abstention, risk, and confirmation safeguards.

The same runner implements uniform-neural, cross-trial shuffled-neural, remove-RAG,
candidate-rotated retrieval, irrelevant-retrieval, remove-context, and remove-retrieval-context
ablations. Shuffling neural evidence uses another complete trial posterior; shuffling retrieval
moves each evidence bundle to a different visible language candidate. Required alternate language
and retrieval snapshots are input artifacts, not values synthesized inside the runner.

Conditions D–F and their complete-system ablations require a non-empty personalization lift plus a
specific adapter ID and SHA-256 for every trial. A controlled fixture may exercise mechanics, but
its result is forcibly marked non-claim-eligible. A claim-eligible configuration identifies the
evidence as a held-out adapter; that label is a protocol eligibility check, not proof of benefit.

Overall and per-subject metrics use the established machine-readable evaluation schema. When F is
present, deterministic paired hierarchical bootstrap intervals resample subjects and then trials
within subjects for top-1 recall and completion-rate differences against F. These intervals are
descriptive and do not establish non-inferiority, clinical utility, or statistical significance.

Write result JSON, trial and mapping JSONL, condition-metric and paired-interval CSV, and a run
manifest. Verify every table checksum on read and cross-check the recorded dataset, aggregation,
ranking-policy, and personalization-adapter identities against the result. Original-task decoder,
counterfactual replay, and controlled simulation artifacts always use separate run kinds and
tables.

## Consequences

The repository can now execute the paired A–F fusion mechanics from an explicitly prepared input,
audit exactly how source flashes were mapped, and prevent controlled personalization fixtures from
supporting research claims. The implementation does not convert Study P into observed word-level
intent and never permits automatic selection.

No complete real-data counterfactual result is checked into the repository: Study P data and model
artifacts are intentionally local, and no trained language-model LoRA is bundled. ADR 0017
provides the local training and verified-loading path without turning an unrun workflow into
evidence.
Personalization-data-quantity, candidate-count, fusion-weight, stale-retrieval, unseen-vocabulary,
and profile-mismatch sweeps therefore remain dependency-gated follow-on analyses. Results from
those sweeps must use explicit tracked configurations. ADR 0016 amends the trial and metric
provenance for protocol v2; v1 artifacts remain readable but must not be used for absent-target
availability claims.
