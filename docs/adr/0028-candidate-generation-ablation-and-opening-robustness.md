# ADR 0028: Candidate-generation ablation and opening robustness

- Status: Accepted
- Date: 2026-07-28

## Context

Exploratory candidate-generation v2 raised exact-span availability from 28.7% to 52.5%, but the
existing test set had already been inspected and opening availability remained 4.8%. That result
does not show which v2 components caused the change, and it does not test whether complete opening
phrases generalize to unseen combinations.

## Decision

Lock Step 4 as an exploratory supplement with two distinct analyses:

1. Re-evaluate the existing, explicitly test-exposed benchmark using full v2, no profile
   conditioning, no grammar routing, frequency-only scoring with grammar routing retained, and a
   two-stage opening method.
2. Apply the same five frozen methods to a deterministic synthetic robustness benchmark. Its
   train, validation, and test templates and topics are disjoint. The nine exact test opening
   stem-action combinations are absent from train and validation, while every component is
   observable there.

All one-stage methods retain the nine-candidate budget. The opening method first offers up to nine
stems and, after an observed stem selection, offers up to nine actions. It never accepts an
intended target. The offline replay uses the correct stem as a teacher-forced simulated selection
before generating stage two; this is recorded as an additional selection, not treated as free.
Later spans use unchanged full v2.

The source configuration, source digest, method order, 10,000 message-clustered bootstrap
resamples, and outcome-based omission prohibition were fixed before a successful execution.
Artifacts retain both candidate banks, all generated robustness splits, trial-level candidates,
paired contrasts, checksums, and dirty-tree provenance.

## Consequences

On the existing benchmark, removing grammar routing reduced overall availability from 52.5% to
12.9%, while removing profile conditioning reduced it to 47.3%. The two-stage opening method
raised opening availability from 4.8% to 18.8% (paired delta +14.0 percentage points, 95% interval
+12.1 to +16.0) and complete-message availability from 2.1% to 9.1% (+7.0 points, 95% interval
+5.5 to +8.5).

On the held-out-combination benchmark, all one-stage methods had 0% opening and complete-message
availability. Two-stage composition reached 100% opening availability because all three stems and
all nine actions fit within their respective menus, and complete-message availability was 32.3%
(paired delta +32.3 points, 95% interval +29.4 to +35.2). This is a designed compositional stress
test, not evidence that arbitrary natural openings will attain 100% coverage.

The robustness data remain synthetic and developer-authored; the protocol was locked before
execution but was not independently preregistered. The findings support an algorithmic explanation
and a concrete interface tradeoff, not participant benefit or unrestricted language generation.
The CPU-only run needs no Qwen model or adapter and completed on the supported MacBook Air in about
75 seconds with 0.80 GiB peak resident memory.
