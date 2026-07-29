# ADR 0029: Hierarchical opening generalization

- Status: Accepted
- Date: 2026-07-28

## Context

Step 4 showed that a two-stage stem-action interface could cover every opening in a constrained
three-stem, nine-action benchmark. Because both component vocabularies fit within their menus,
that result could not establish performance under a genuine nine-candidate bottleneck. It also did
not test preferences, clarifications, status openings, or surface families absent from fitting.

## Decision

Add a separate, locked exploratory experiment with four discourse intents: request, preference,
clarification, and status. Its fitting data contain 24 stems and 48 content words. Exact
stem-content pairs are partitioned deterministically into train, validation, and a 288-opening
combination test. A second 384-opening test uses eight paraphrase-family stems that are completely
absent from fitting while retaining observed content words. There is no exact fit/test opening
overlap.

Compare three target-blind interfaces at the same maximum menu size of nine:

1. one-stage retrieval of a complete opening;
2. global stem selection followed by intent-compatible content selection; and
3. intent selection followed by intent-specific stem and content selection.

Downstream menus receive only teacher-forced selections that would have been observable from the
preceding interface stage. No generator accepts the intended opening or intended content.
Alongside exact availability, report planned selections, menus reached, total candidate
exposures, availability per required selection, and paired 10,000-resample intervals within fixed
synthetic-profile strata.

## Consequences

For held-out combinations, one-stage retrieval achieved 0% availability, two-stage composition
25.0% (difference from one-stage +25.0 percentage points, 95% interval +20.1 to +29.9), and
three-stage intent conditioning 66.7% (+66.7 points, 95% interval +61.5 to +72.2). Three-stage
availability exceeded two-stage by 41.7 points (95% interval +36.1 to +47.6).

The added selections did not erase that difference under the prespecified descriptive efficiency
measure. Availability per required selection was 0 for one stage, 0.125 for two stages, and 0.222
for three stages. The three-minus-two difference was +0.097 (95% interval +0.073 to +0.121).
Three-stage trials exposed 19 candidates across their three menus on average, compared with 12.4
across reached two-stage menus and nine for one-stage retrieval.

Every method achieved 0% on the unseen paraphrase-family challenge. This negative result identifies
a closed-vocabulary boundary: hierarchy composes observed components but does not create new
surface stems. Addressing that boundary would require a separately locked generative or
token-compositional method rather than further tuning on these exposed tests.

The benchmark is synthetic and developer-authored, and its protocol was locked immediately before
execution rather than independently preregistered. Results remain exploratory interface evidence,
not a live communication-rate or participant-benefit claim. The complete CPU-only run took under
one second of experiment time and about 0.63 GiB peak resident memory on the supported MacBook Air.
